"""Validation: gate execution, caching, and the security scanners."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from forge.config import Config, SandboxConfig
from forge.kernel.ledger import Ledger
from forge.validation import gates as _builtin_gates  # noqa: F401  (registers gates)
from forge.validation.gate import Gate, GateContext, gate_registry
from forge.validation.runner import GateRunner
from forge.validation.types import Severity, ValidationReport, Verdict
from forge.workspace.sandbox import LocalSandbox


@pytest.fixture
def gate_ctx(tmp_path: Path) -> GateContext:
    root = tmp_path / "wt"
    root.mkdir()
    return GateContext(
        sandbox=LocalSandbox(SandboxConfig(command_timeout=20), root),
        root=root,
        artifacts_dir=tmp_path / "artifacts",
        toolchain={"commands": {}},
        node_id="node_test",
    )


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


def test_failing_verdict_keeps_the_tail_of_long_output() -> None:
    verdict = Verdict.failing("unit", "tests failed", evidence="x" * 10_000 + "THE ACTUAL ERROR")
    rendered = verdict.render(max_evidence=500)
    assert "THE ACTUAL ERROR" in rendered
    assert len(rendered) < 1200


def test_skipped_gates_do_not_block() -> None:
    report = ValidationReport(verdicts=[Verdict.skip("types", "no type checker")])
    assert report.passed


def test_errored_gates_are_distinguished_from_failures() -> None:
    report = ValidationReport(verdicts=[Verdict.error("browser", "playwright crashed")])
    assert not report.passed and report.errors


# --------------------------------------------------------------------------
# Built-in gates
# --------------------------------------------------------------------------


def test_schema_gate_finds_malformed_json(gate_ctx: GateContext) -> None:
    (gate_ctx.root / "package.json").write_text('{"name": "x",}')
    verdict = gate_registry.create("schema").run(gate_ctx)
    assert not verdict.passed
    assert verdict.issues[0].path == "package.json"


def test_secret_scanner_flags_a_real_key(gate_ctx: GateContext) -> None:
    (gate_ctx.root / "config.js").write_text('const key = "AKIAQ7RZ3JKLMN4PXWTY";\n')
    verdict = gate_registry.create("secrets").run(gate_ctx)
    assert not verdict.passed
    assert verdict.issues[0].severity == Severity.CRITICAL


def test_structured_matches_are_not_suppressed_by_word_heuristics(gate_ctx: GateContext) -> None:
    """A real key that happens to contain 'test' must still be caught."""
    (gate_ctx.root / "config.js").write_text('const k = "ghp_TESTaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";\n')
    assert not gate_registry.create("secrets").run(gate_ctx).passed


def test_documented_vendor_example_keys_are_ignored(gate_ctx: GateContext) -> None:
    (gate_ctx.root / "README.md").write_text("Set it to AKIAIOSFODNN7EXAMPLE for the tutorial.\n")
    assert gate_registry.create("secrets").run(gate_ctx).passed


@pytest.mark.parametrize(
    "line",
    [
        'const key = "your-api-key-here";',
        'const key = process.env.API_KEY;',
        'password = "example-placeholder";',
    ],
)
def test_secret_scanner_ignores_placeholders_and_env_reads(gate_ctx: GateContext, line: str) -> None:
    """A noisy secret scanner gets ignored, which is worse than no scanner."""
    (gate_ctx.root / "config.js").write_text(line + "\n")
    assert gate_registry.create("secrets").run(gate_ctx).passed


def test_secret_scanner_skips_lockfiles(gate_ctx: GateContext) -> None:
    (gate_ctx.root / "package-lock.json").write_text('{"k": "AKIAIOSFODNN7EXAMPLE"}')
    assert gate_registry.create("secrets").run(gate_ctx).passed


def test_unit_gate_extracts_failing_test_names(gate_ctx: GateContext) -> None:
    gate = gate_registry.create("unit")
    issues = gate.parse("FAILED tests/test_a.py::test_x - AssertionError: nope\n1 failed", gate_ctx)
    assert issues and "tests/test_a.py::test_x" in issues[0].message


def test_command_gate_skips_when_the_project_has_no_such_command(gate_ctx: GateContext) -> None:
    assert gate_registry.create("unit").run(gate_ctx).skipped


def test_command_gate_runs_a_configured_command(gate_ctx: GateContext) -> None:
    gate_ctx.settings = {"command": "exit 0"}
    assert gate_registry.create("lint").run(gate_ctx).passed

    gate_ctx.settings = {"command": "echo 'src/a.ts:12: error: boom' >&2; exit 1"}
    verdict = gate_registry.create("lint").run(gate_ctx)
    assert not verdict.passed


def test_a_missing_tool_skips_rather_than_fails(gate_ctx: GateContext) -> None:
    """The most expensive bug found in live testing.

    An absent `ruff` failed a node four times, escalated it to the costliest
    rung on every retry, and finally blocked the project -- over a linter that
    was never installed. A missing tool is not a failing check.
    """
    gate_ctx.settings = {"command": "definitely-not-installed-xyz check ."}
    verdict = gate_registry.create("lint").run(gate_ctx)

    assert verdict.skipped, "a missing binary must skip"
    assert verdict.ok, "and must not block the node"
    assert "not installed" in verdict.summary


def test_a_missing_tool_makes_the_gate_inapplicable(gate_ctx: GateContext) -> None:
    gate_ctx.settings = {"command": "definitely-not-installed-xyz ."}
    assert not gate_registry.create("lint").applicable(gate_ctx)


def test_exit_127_from_a_wrapper_also_skips(gate_ctx: GateContext) -> None:
    """The binary exists but what it invokes does not -- npx, make, poetry run."""
    gate_ctx.settings = {"command": "sh -c 'echo nope: command not found >&2; exit 127'"}
    verdict = gate_registry.create("lint").run(gate_ctx)
    assert verdict.skipped


def test_uninstalled_packages_skip_rather_than_fail(gate_ctx: GateContext) -> None:
    """Exit 1 from npx before `npm install` is a toolchain fact, not a type error.

    Verbatim output from a live run: the scaffold node had written package.json
    and tsconfig.json but nothing had fetched node_modules yet. The types gate
    "failed", and the node escalated local -> codex -> claude over a compiler
    that was simply not installed. Exit 1 rather than 127, and no wording a
    shell would recognise, so neither existing check caught it.
    """
    message = 'npm error npx canceled due to missing packages and no YES option: [tsc@2.0.4]'
    gate_ctx.settings = {"command": f"sh -c {shlex.quote(f'echo {shlex.quote(message)} >&2; exit 1')}"}
    verdict = gate_registry.create("types").run(gate_ctx)

    assert verdict.skipped, "an uninstalled package must skip"
    assert verdict.ok, "and must not burn the node's attempts on cloud rungs"


def test_a_real_type_error_still_fails_when_the_compiler_is_installed(gate_ctx: GateContext) -> None:
    """The skip is narrow: real tsc output must still fail the gate."""
    message = "src/game.ts(12,5): error TS2322: Type string is not assignable to type number."
    gate_ctx.settings = {"command": f"sh -c {shlex.quote(f'echo {shlex.quote(message)}; exit 1')}"}
    verdict = gate_registry.create("types").run(gate_ctx)

    assert not verdict.skipped
    assert not verdict.passed


def test_a_real_failure_from_an_installed_tool_still_fails(gate_ctx: GateContext) -> None:
    """The skip must not swallow genuine failures."""
    gate_ctx.settings = {"command": "sh -c 'echo src/a.py:1: error: bad >&2; exit 1'"}
    verdict = gate_registry.create("lint").run(gate_ctx)
    assert not verdict.passed and not verdict.skipped


def test_a_failing_gate_that_merely_mentions_a_missing_command_still_fails(
    gate_ctx: GateContext,
) -> None:
    """`_looks_missing` is a text match, and it was not gated on failure.

    A test suite whose own assertion output quotes "command not found", or a
    build relaying a child process's error, was reported SKIPPED. A skipped
    verdict counts as ok, so a genuinely failing gate passed validation and the
    node was marked succeeded on broken code.
    """
    message = "FAIL src/shell.test.ts > reports a missing binary: command not found"
    gate_ctx.settings = {
        "command": f"sh -c {shlex.quote(f'echo {shlex.quote(message)}; exit 1')}"
    }
    verdict = gate_registry.create("unit").run(gate_ctx)

    assert not verdict.skipped, "the command ran; it just failed"
    assert not verdict.ok, "and a failing gate must block"


def test_a_passing_gate_is_never_reported_as_skipped(gate_ctx: GateContext) -> None:
    message = "ok - handles 'command not found' gracefully"
    gate_ctx.settings = {
        "command": f"sh -c {shlex.quote(f'echo {shlex.quote(message)}; exit 0')}"
    }
    verdict = gate_registry.create("unit").run(gate_ctx)
    assert verdict.passed and not verdict.skipped


def test_the_coverage_floor_is_enforced_on_a_run_that_passed(gate_ctx: GateContext) -> None:
    """The only case where a coverage floor means anything, and the only one it missed.

    It read the percentage from `verdict.evidence`, which `CommandGate.run`
    populates only when the command *failed*. So the floor was enforced exactly
    when the test suite was already broken for some other reason, and never on
    a green run -- which is to say never.
    """
    # Verbatim shape of a vitest/v8 summary: the percent sign is in the header
    # row, never on the totals row.
    report = "All files |   41.2 |    30.1 |     100 |   41.2 |"
    gate_ctx.settings = {
        "floor": 80.0,
        "command": f"sh -c {shlex.quote(f'echo {shlex.quote(report)}; exit 0')}",
    }
    verdict = gate_registry.create("coverage").run(gate_ctx)

    assert verdict.score == pytest.approx(41.2)
    assert not verdict.passed, "41.2% is below the 80% floor"
    assert "below" in verdict.summary


def test_coverage_above_the_floor_passes(gate_ctx: GateContext) -> None:
    report = "All files |   91.5 |    88.0 |     100 |   91.5 |"
    gate_ctx.settings = {
        "floor": 80.0,
        "command": f"sh -c {shlex.quote(f'echo {shlex.quote(report)}; exit 0')}",
    }
    verdict = gate_registry.create("coverage").run(gate_ctx)
    assert verdict.passed and verdict.score == pytest.approx(91.5)


def test_the_browser_timeout_reaches_the_gate_context(tmp_path: Path, ledger: Ledger) -> None:
    """It was `browser_timeout if False else command_timeout`: dead config.

    Nothing read it, so `page.goto` inherited the 900s command ceiling and a
    page that never loaded hung a gate for fifteen minutes.
    """
    from forge.config import Config

    config = Config()
    config.sandbox.command_timeout = 900.0
    config.validation.browser_timeout = 45.0
    runner = GateRunner(config, ledger)
    ctx = runner.build_context(
        root=tmp_path,
        sandbox=LocalSandbox(SandboxConfig(command_timeout=900), tmp_path),
        toolchain={"commands": {}},
    )

    assert ctx.browser_timeout == 45.0
    assert ctx.timeout == 900.0, "everything else keeps the sandbox ceiling"


def test_smoke_flow_resolves_relative_routes_and_normalises_selector_values() -> None:
    from forge.validation.gates.browser import SmokeFlowGate

    class Page:
        url = "http://127.0.0.1:5173/"

        class Keyboard:
            def __init__(self) -> None:
                self.events: list[tuple[str, str]] = []

            def down(self, key: str) -> None:
                self.events.append(("down", key))

            def up(self, key: str) -> None:
                self.events.append(("up", key))

        def __init__(self) -> None:
            self.gotos: list[str] = []
            self.selectors: list[str] = []
            self.waits: list[float] = []
            self.sizes: list[dict[str, int]] = []
            self.keyboard = self.Keyboard()

        def goto(self, target: str, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self.gotos.append(target)

        def wait_for_selector(self, selector: str, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self.selectors.append(selector)

        def wait_for_timeout(self, duration: float) -> None:
            self.waits.append(duration)

        def set_viewport_size(self, size: dict[str, int]) -> None:
            self.sizes.append(size)

    page = Page()
    SmokeFlowGate._run_step(page, {"action": "goto", "value": "/?screenshot=1"})
    SmokeFlowGate._run_step(
        page,
        {"action": "wait_for", "value": "selector: #game-canvas"},
    )
    SmokeFlowGate._run_step(page, {"action": "wait", "value": "0.5"})
    SmokeFlowGate._run_step(page, {"action": "key_down", "value": "Space"})
    SmokeFlowGate._run_step(page, {"action": "key_up", "value": "Space"})
    SmokeFlowGate._run_step(page, {"action": "resize", "value": "800x600"})

    assert page.gotos == ["http://127.0.0.1:5173/?screenshot=1"]
    assert page.selectors == ["#game-canvas"]
    assert page.waits == [500.0]
    assert page.keyboard.events == [("down", "Space"), ("up", "Space")]
    assert page.sizes == [{"width": 800, "height": 600}]


def test_visual_baselines_ignore_smoke_failure_captures(gate_ctx: GateContext) -> None:
    from forge.validation.gates.visual import _screenshots

    directory = gate_ctx.artifacts_dir / gate_ctx.node_id
    directory.mkdir(parents=True)
    for name in (
        "screenshot_.png",
        "smoke_failure_1.png",
        "smoke_final.png",
        "diff_screenshot_.png",
    ):
        (directory / name).write_bytes(b"png")

    assert [path.name for path in _screenshots(gate_ctx)] == ["screenshot_.png"]


def test_dangerous_patterns_are_advisory_not_blocking(gate_ctx: GateContext) -> None:
    (gate_ctx.root / "a.js").write_text("el.innerHTML = userInput;\n")
    verdict = gate_registry.create("dangerous_patterns").run(gate_ctx)
    assert verdict.passed, "advisory gates report without stalling the run"
    assert verdict.issues


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def test_runner_orders_cheap_gates_first(config, ledger: Ledger, gate_ctx: GateContext) -> None:
    runner = GateRunner(config, ledger)
    resolved = runner.resolve(["browser", "schema", "unit", "secrets"])
    assert [g.name for g in resolved] == ["schema", "secrets", "unit", "browser"]


def test_runner_survives_a_gate_that_raises(config, ledger: Ledger, gate_ctx: GateContext) -> None:
    """A broken gate must degrade to an error verdict, never kill the run."""

    class Exploding(Gate):
        name = "exploding"
        cacheable = False

        def run(self, ctx: GateContext) -> Verdict:
            raise RuntimeError("gate is broken")

    gate_registry.register(Exploding)
    report = GateRunner(config, ledger).run(["exploding"], gate_ctx)
    assert report.errors and "gate is broken" in report.verdicts[0].summary


def test_passing_verdicts_are_cached_by_tree_content(config, ledger: Ledger, gate_ctx: GateContext) -> None:
    runs: list[int] = []

    class Counting(Gate):
        name = "counting"
        cacheable = True

        def run(self, ctx: GateContext) -> Verdict:
            runs.append(1)
            return Verdict.passing(self.name, "ok")

    gate_registry.register(Counting)
    runner = GateRunner(config, ledger)

    runner.run(["counting"], gate_ctx)
    runner.run(["counting"], gate_ctx)
    assert len(runs) == 1, "an unchanged tree must not re-run a deterministic gate"

    (gate_ctx.root / "new.txt").write_text("changed")
    runner.run(["counting"], gate_ctx)
    assert len(runs) == 2, "a changed tree must invalidate the cached verdict"


def test_failing_verdicts_are_never_cached(config, ledger: Ledger, gate_ctx: GateContext) -> None:
    """Caching a failure would leave a gate permanently red after the fix."""
    runs: list[int] = []

    class AlwaysFails(Gate):
        name = "always_fails"
        cacheable = True

        def run(self, ctx: GateContext) -> Verdict:
            runs.append(1)
            return Verdict.failing(self.name, "nope")

    gate_registry.register(AlwaysFails)
    runner = GateRunner(config, ledger)
    runner.run(["always_fails"], gate_ctx)
    runner.run(["always_fails"], gate_ctx)
    assert len(runs) == 2


def test_unknown_gates_are_skipped_not_fatal(config, ledger: Ledger, gate_ctx: GateContext) -> None:
    report = GateRunner(config, ledger).run(["schema", "no_such_gate"], gate_ctx)
    assert [v.gate for v in report.verdicts] == ["schema"]


def test_gate_outcomes_reach_the_ledger(config, ledger: Ledger, gate_ctx: GateContext) -> None:
    GateRunner(config, ledger).run(["schema"], gate_ctx)
    types = {e.type for e in ledger.read()}
    assert "gate.passed" in types


def test_fail_fast_stops_before_runtime_and_advisory_phases(
    config, ledger: Ledger, gate_ctx: GateContext
) -> None:
    """A compiler failure must not launch browsers or performance checks."""
    ran: list[str] = []

    class StaticFailure(Gate):
        name = "test_static_failure"
        order = 30
        cacheable = False

        def run(self, ctx: GateContext) -> Verdict:
            ran.append(self.name)
            return Verdict.failing(self.name, "compile failed")

    class RuntimeGate(Gate):
        name = "test_runtime_after_failure"
        order = 110
        cacheable = False

        def run(self, ctx: GateContext) -> Verdict:
            ran.append(self.name)
            return Verdict.passing(self.name)

    class AdvisoryGate(Gate):
        name = "test_advisory_after_failure"
        order = 150
        blocking = False
        cacheable = False

        def run(self, ctx: GateContext) -> Verdict:
            ran.append(self.name)
            return Verdict.passing(self.name)

    for gate in (StaticFailure, RuntimeGate, AdvisoryGate):
        gate_registry.register(gate)

    report = GateRunner(config, ledger).run(
        [StaticFailure.name, RuntimeGate.name, AdvisoryGate.name],
        gate_ctx,
        fail_fast=True,
    )

    assert [verdict.gate for verdict in report.verdicts] == [StaticFailure.name]
    assert ran == [StaticFailure.name]


def test_runtime_exclusive_gates_follow_successful_static_checks(
    config, ledger: Ledger, gate_ctx: GateContext
) -> None:
    """Exclusive means serial, not 'run before lower-order checks'."""
    ran: list[str] = []

    class StaticPass(Gate):
        name = "test_static_before_runtime"
        order = 20
        cacheable = False

        def run(self, ctx: GateContext) -> Verdict:
            ran.append(self.name)
            return Verdict.passing(self.name)

    class BrowserNamedGate(Gate):
        # Use the built-in exclusive name temporarily; restore it immediately.
        name = "browser"
        order = 110
        cacheable = False

        def run(self, ctx: GateContext) -> Verdict:
            ran.append(self.name)
            return Verdict.passing(self.name)

    original = type(gate_registry.create("browser"))
    gate_registry.register(StaticPass)
    gate_registry.register(BrowserNamedGate)
    try:
        GateRunner(config, ledger).run(
            [BrowserNamedGate.name, StaticPass.name], gate_ctx, fail_fast=True
        )
    finally:
        gate_registry.register(original)

    assert ran == [StaticPass.name, BrowserNamedGate.name]


def test_tsc_errors_parse_into_structured_issues(gate_ctx: GateContext) -> None:
    """tsc writes `file.ts(line,col):` where the generic parser expects colons.

    Without a dedicated pattern every TypeScript error arrived as unstructured
    text and the issue count read zero on a genuinely failing gate -- which
    costs the fixing agent both the file/line and roughly an order of magnitude
    in tokens over the raw output.
    """
    output = (
        "src/game.ts(12,5): error TS2322: Type 'string' is not assignable to type 'number'.\n"
        "src/table.ts(3,18): error TS2304: Cannot find name 'Vec2'.\n"
    )
    issues = gate_registry.create("types").parse(output, gate_ctx)

    assert [(i.path, i.line) for i in issues] == [("src/game.ts", 12), ("src/table.ts", 3)]
    assert "TS2322" in issues[0].message
    assert all(i.severity == Severity.HIGH for i in issues)


def test_non_tsc_type_output_still_parses(gate_ctx: GateContext) -> None:
    """mypy uses `path:line: message`; the generic parser must still apply."""
    issues = gate_registry.create("types").parse(
        "forge/models/client.py:42: error: Incompatible return value type\n", gate_ctx
    )
    assert issues and issues[0].path == "forge/models/client.py" and issues[0].line == 42


def test_a_failed_gate_records_why_not_only_that(config, ledger: Ledger, gate_ctx: GateContext) -> None:
    """The ledger said `{"exit_code": 2, "command": "npm run typecheck"}` and no more.

    The evidence was already assembled and already bounded -- `Verdict.render`
    feeds it to the repair prompt -- so the model could see the reason while an
    operator reading the ledger could not. Diagnosing a zero-byte source file
    meant leaving Forge and running tsc by hand.
    """
    from forge.kernel.events import EventType

    class Failing(Gate):
        name = "failing"
        cacheable = False

        def run(self, ctx: GateContext) -> Verdict:
            return Verdict.failing(
                "failing",
                "`npm run typecheck` failed with exit 2",
                evidence="src/engine/collide.ts(1,1): error TS2306: File is not a module.",
            )

    gate_registry.register(Failing)
    GateRunner(config, ledger).run(["failing"], gate_ctx)

    failed = ledger.read(types=[EventType.GATE_FAILED])
    assert failed, "the gate should have failed"
    assert "TS2306" in failed[-1].payload.get("evidence", ""), (
        "a failure has to record why, or the ledger cannot answer the only question asked of it"
    )


def test_a_passing_gate_does_not_carry_evidence(config, ledger: Ledger, gate_ctx: GateContext) -> None:
    """A pass has nothing to explain, and every gate on every node would bloat it."""
    from forge.kernel.events import EventType

    class Passing(Gate):
        name = "passing_quietly"
        cacheable = False

        def run(self, ctx: GateContext) -> Verdict:
            return Verdict(gate="passing_quietly", passed=True, evidence="lots of noise" * 500)

    gate_registry.register(Passing)
    GateRunner(config, ledger).run(["passing_quietly"], gate_ctx)

    passed = ledger.read(types=[EventType.GATE_PASSED])
    assert passed and "evidence" not in passed[-1].payload


def test_gate_evidence_in_the_ledger_is_bounded(config, ledger: Ledger, gate_ctx: GateContext) -> None:
    """A long test log must not become a database row."""
    from forge.kernel.events import EventType
    from forge.validation.runner import _LEDGER_EVIDENCE_CHARS

    class Verbose(Gate):
        name = "verbose_failure"
        cacheable = False

        def run(self, ctx: GateContext) -> Verdict:
            return Verdict.failing("verbose_failure", "failed", evidence="x" * 100_000)

    gate_registry.register(Verbose)
    GateRunner(config, ledger).run(["verbose_failure"], gate_ctx)

    failed = ledger.read(types=[EventType.GATE_FAILED])
    assert len(failed[-1].payload["evidence"]) == _LEDGER_EVIDENCE_CHARS


def test_a_compiler_error_survives_truncation() -> None:
    """It kept only the tail, which is right for a test runner and wrong for tsc.

    A compiler lists errors in file order: the first is the cause and the rest
    are cascade. Showing a model only the tail hands it the downstream noise and
    truncates away the line it needs. That is what "could not find the unbalanced
    brace in three rounds" looked like from the model's side -- it was never
    shown the brace.
    """
    root = "src/engine/collide.ts(12,1): error TS1005: '}' expected."
    cascade = "\n".join(f"src/other{i}.ts: cascade error {i}" for i in range(400))
    rendered = Verdict.failing("types", "tsc failed", evidence=f"{root}\n{cascade}").render(
        max_evidence=600
    )

    assert "TS1005" in rendered, "the root cause must survive"
    assert "collide.ts(12,1)" in rendered, "and so must its location"
    assert "cascade error 399" in rendered, "the tail still matters for test runners"


def test_an_authored_smoke_flow_outlives_the_node_that_wrote_it(
    config: Config, ledger: Ledger
) -> None:
    """The only behavioural gate must not be applicable exactly once.

    `browser_qa` wrote its flow into one node's spec, so on a real project the
    smoke gate ran twice and skipped 144 times with "not applicable to this
    project" while every other gate passed a game nobody could play.
    """
    from forge.validation.runner import SMOKE_FLOW_KEY, GateRunner

    runner = GateRunner(config, ledger)

    bare = runner.build_context(root=config.project_dir, sandbox=None, toolchain={})
    assert not bare.setting("smoke", {}).get("steps")

    ledger.kv_set(SMOKE_FLOW_KEY, {"steps": [{"action": "press", "value": "Space"}]})

    inherited = runner.build_context(root=config.project_dir, sandbox=None, toolchain={})
    assert inherited.setting("smoke")["steps"] == [{"action": "press", "value": "Space"}]

    # A node that brings its own flow still wins.
    override = runner.build_context(
        root=config.project_dir,
        sandbox=None,
        toolchain={},
        settings={"smoke": {"steps": [{"action": "click", "selector": "#play"}]}},
    )
    assert override.setting("smoke")["steps"] == [{"action": "click", "selector": "#play"}]
