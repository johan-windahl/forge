"""Agent-level behaviour that is not visible from the orchestrator.

Currently the file-request protocol: the one channel a coding model has for
saying "I am guessing at this interface" before it writes 500 lines against a
guess.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.agents.coding import (
    _TRANSIENT_OPENCODE_ERROR,
    MAX_REQUESTED_FILES,
    OPENCODE_PROMPT_BUDGET,
    CodingAgent,
)
from forge.agents.goal import _goal_images
from forge.agents.planning import (
    ARCHITECTURE_SCHEMA,
    _architecture_markdown,
    _attach_reference_images,
)
from forge.agents.reviewing import _whole_review_paths
from forge.agents.shipping import _documentation_files
from forge.memory.context import ContextBuilder, reference_images
from forge.models.types import estimate_tokens
from forge.validation.types import ValidationReport, Verdict
from forge.workspace.git import Repo
from forge.workspace.patch import EDIT_PLAN_SCHEMA, EditPlan


@dataclass
class _Ctx:
    """The only thing `_grant_files` touches."""

    root: Path


class _Coding(CodingAgent):
    """CodingAgent is abstract; `_grant_files` does not care what `run` does."""

    kind = "t_grant"

    def run(self, ctx):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _agent() -> _Coding:
    return _Coding()


def test_whole_review_paths_include_source_contracts_and_tests(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "docs" / "architecture.md").write_text("contract")
    (tmp_path / "src" / "game.ts").write_text("export {}")
    (tmp_path / "tests" / "game.test.ts").write_text("test")
    (tmp_path / "package.json").write_text("{}")
    (tmp_path / "node_modules" / "vendor.ts").write_text("ignored")

    paths = _whole_review_paths(tmp_path)

    assert paths[:3] == [
        "docs/architecture.md",
        "src/game.ts",
        "tests/game.test.ts",
    ]
    assert "package.json" in paths
    assert not any("node_modules" in path for path in paths)


def test_documentation_context_includes_existing_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("existing handoff")
    (tmp_path / "package.json").write_text("{}")

    assert _documentation_files(tmp_path) == ["README.md", "package.json"]


def test_goal_images_use_reference_and_latest_capture_set(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    references = tmp_path / ".forge" / "references"
    references.mkdir(parents=True)
    (references / "nightmare.png").write_bytes(b"reference")
    artifacts = tmp_path / "artifacts"
    older = artifacts / "node_old"
    newer = artifacts / "node_new"
    older.mkdir(parents=True)
    newer.mkdir()
    (older / "screenshot_.png").write_bytes(b"old")
    (newer / "screenshot_.png").write_bytes(b"new")
    (newer / "screenshot_?screenshot=1.png").write_bytes(b"fixed")
    older.touch()
    newer.touch()

    refs, candidates = _goal_images(root, artifacts)

    # References now travel with the operator's description of each; undeclared
    # files found by directory scan carry an empty one.
    assert [path.name for path, _ in refs] == ["nightmare.png"]
    assert [path.parent.name for path in candidates] == ["node_new", "node_new"]


@dataclass
class _DiffRepo:
    patch: str

    def diff(self, **_kwargs) -> str:  # type: ignore[no-untyped-def]
        return self.patch


@dataclass
class _ScopeNode:
    kind: str = "implement"


@dataclass
class _ScopeCtx:
    repo: _DiffRepo
    node: _ScopeNode


@dataclass
class _WipCommit:
    sha: str
    subject: str
    node_id: str | None


@dataclass
class _WipDiffRepo:
    head_patch: str
    baseline_patch: str
    commits: list[_WipCommit]

    def log(self, **_kwargs) -> list[_WipCommit]:  # type: ignore[no-untyped-def]
        return self.commits

    def diff(self, ref=None, **_kwargs) -> str:  # type: ignore[no-untyped-def]
        return self.baseline_patch if ref == "wip-1^" else self.head_patch


@dataclass
class _WipScopeNode:
    id: str = "node-1"
    attempts: int = 2
    kind: str = "implement"


@dataclass
class _GateNode:
    gates: list[str]


@dataclass
class _GateCtx:
    node: _GateNode
    config: object | None = None


@dataclass
class _CleanRepo:
    def is_dirty(self) -> bool:
        return False


@dataclass
class _RepairNode:
    title: str
    gates: list[str]
    kind: str = "implement"


@dataclass
class _RepairCtx:
    node: _RepairNode
    spec: dict
    repo: _CleanRepo

    class _Logger:
        def info(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

    def logger(self) -> _Logger:
        return self._Logger()


# --------------------------------------------------------------------------
# What the model may ask for
# --------------------------------------------------------------------------


def test_a_requested_file_that_exists_is_granted(tmp_path: Path) -> None:
    """The case this exists for.

    The pinball run's edit plan said, in its summary field because there was
    nowhere else to put it: "Need to see tuning.ts first for physics constants."
    It then wrote 500 lines against a guessed interface.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src/tuning.ts").write_text("export const GRAVITY = 9.81;\n")

    granted = _agent()._grant_files(_Ctx(tmp_path), ["src/tuning.ts"])
    assert granted == ["src/tuning.ts"]


def test_a_file_the_model_supposedly_already_has_is_still_granted(tmp_path: Path) -> None:
    """Asking for it *is* the evidence that it was not visible.

    haiku correctly asked for vec2.ts, tuning.ts and types.ts. All three were in
    the intended path list, so the old "already in context" refusal granted
    nothing, and the throwaway placeholder it had written alongside the request
    got committed as the deliverable. The caller pins these into their own
    section rather than refusing them.
    """
    (tmp_path / "seen.ts").write_text("x")
    assert _agent()._grant_files(_Ctx(tmp_path), ["seen.ts"]) == ["seen.ts"]


def test_a_request_for_something_that_does_not_exist_is_ignored(tmp_path: Path) -> None:
    """Silently: the request is advisory, and failing a node over a bad hint
    would be worse than dropping the hint."""
    assert _agent()._grant_files(_Ctx(tmp_path), ["nope.ts"]) == []


def test_a_request_cannot_escape_the_workspace(tmp_path: Path) -> None:
    """A request is not authority."""
    (tmp_path / "inside.ts").write_text("x")
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("credentials")

    granted = _agent()._grant_files(
        _Ctx(tmp_path),
        ["../secret.txt", "/etc/passwd", "../../etc/hosts", "inside.ts"],
    )
    assert granted == ["inside.ts"]


def test_a_traversal_is_not_normalised_into_a_file_that_happens_to_exist(tmp_path: Path) -> None:
    """`lstrip("./")` takes a character set, not a prefix.

    It turned "../secret.txt" into "secret.txt", so the traversal check never saw
    a "..", and a same-named file inside the workspace was served in answer to a
    request that pointed outside it.
    """
    (tmp_path / "secret.txt").write_text("the workspace's own file")
    (tmp_path.parent / "secret.txt").write_text("someone else's file")

    assert _agent()._grant_files(_Ctx(tmp_path), ["../secret.txt"]) == []
    # The plain relative form is still fine.
    assert _agent()._grant_files(_Ctx(tmp_path), ["secret.txt"]) == ["secret.txt"]


def test_a_leading_dot_slash_is_still_accepted(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.ts").write_text("x")
    assert _agent()._grant_files(_Ctx(tmp_path), ["./src/a.ts"]) == ["src/a.ts"]


def test_forge_state_and_git_internals_are_never_granted(tmp_path: Path) -> None:
    for directory in (".git", ".forge"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "secrets").write_text("x")

    granted = _agent()._grant_files(_Ctx(tmp_path), [".git/secrets", ".forge/secrets"])
    assert granted == []


def test_a_directory_is_not_a_file(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    assert _agent()._grant_files(_Ctx(tmp_path), ["src"]) == []


def test_a_request_is_capped_so_it_cannot_swamp_the_context(tmp_path: Path) -> None:
    """A model that asks for the whole repository would push out the task."""
    wanted = []
    for i in range(MAX_REQUESTED_FILES + 5):
        (tmp_path / f"f{i}.ts").write_text("x")
        wanted.append(f"f{i}.ts")

    granted = _agent()._grant_files(_Ctx(tmp_path), wanted)
    assert len(granted) == MAX_REQUESTED_FILES


def test_no_request_grants_nothing(tmp_path: Path) -> None:
    assert _agent()._grant_files(_Ctx(tmp_path), []) == []


def test_opencode_prompt_leaves_room_for_tools_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenCode discovers source with tools; preloading a native-sized context
    caused the live local server to reject the request before any edit."""
    builder = ContextBuilder(45_000)
    builder.add("Oversized memory", "context " * 80_000)
    agent = _agent()
    monkeypatch.setattr(agent, "builder", lambda _ctx: builder)

    prompt = agent._opencode_prompt(
        object(),  # type: ignore[arg-type]
        "Implement the focused task",
        report=None,
        advice="",
        round_index=0,
        extra_sections=None,
        include_paths=None,
    )

    assert builder.budget == OPENCODE_PROMPT_BUDGET
    assert estimate_tokens(prompt) < OPENCODE_PROMPT_BUDGET + 500


@pytest.mark.parametrize(
    "message",
    [
        'decode() failed: vk::Queue::submit: ErrorDeviceLost',
        "connection reset by peer",
        "503 service unavailable",
    ],
)
def test_transient_opencode_provider_failures_are_recognised(message: str) -> None:
    assert _TRANSIENT_OPENCODE_ERROR.search(message)


def test_context_errors_are_not_mistaken_for_transient_provider_failures() -> None:
    assert not _TRANSIENT_OPENCODE_ERROR.search("Context size has been exceeded")


def test_architecture_requires_executable_production_workflows() -> None:
    assert "production_workflows" in ARCHITECTURE_SCHEMA["required"]
    workflow = {
        "discipline": "Graphics",
        "target": "A readable reference-driven table",
        "tools": ["Canvas2D", "Playwright screenshots"],
        "method": "Trace composition primitives, render a spike, then compare.",
        "artifacts": ["docs/references/table.png", "visual-baselines/table.png"],
        "validation": ["Fresh-context side-by-side critique"],
        "reference_use": "Match hierarchy and density, not copyrighted artwork.",
    }

    markdown = _architecture_markdown(
        {
            "overview": "A game.",
            "stack": [],
            "modules": [],
            "production_workflows": [workflow],
            "conventions": [],
        }
    )

    assert "## Production workflows" in markdown
    assert "Playwright screenshots" in markdown
    assert "Fresh-context side-by-side critique" in markdown


def test_reference_images_are_discovered_and_attached_to_planning(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "docs" / "references"
    directory.mkdir(parents=True)
    (directory / "b.jpg").write_bytes(b"jpeg")
    (directory / "a.png").write_bytes(b"png")
    (directory / "notes.md").write_text("not an image")
    (tmp_path / "unrelated.png").write_bytes(b"not a reference")

    found = reference_images(tmp_path)
    assert [path.name for path in found] == ["a.png", "b.jpg"]

    builder = ContextBuilder()
    _attach_reference_images(builder, type("Ctx", (), {"root": tmp_path})())
    messages = builder.build(system_prompt="plan", task="make a plan")
    assert [image.label for image in messages[-1].images] == [
        "reference:a.png",
        "reference:b.jpg",
    ]


def test_operator_references_do_not_need_to_dirty_the_worktree(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_references = tmp_path / ".forge" / "references"
    state_references.mkdir(parents=True)
    image = state_references / "quality-bar.png"
    image.write_bytes(b"png")

    assert reference_images(workspace) == [image]


def test_accidental_export_deletion_is_rejected_before_validation() -> None:
    patch = """\
diff --git a/src/types.ts b/src/types.ts
--- a/src/types.ts
+++ b/src/types.ts
@@ -1,2 +1 @@
-export interface InputState { left: boolean }
-export type BootConfig = { screenshot: boolean }
+export interface InputState { left: boolean; right: boolean }
"""
    violation = _agent()._semantic_scope_violation(
        _ScopeCtx(_DiffRepo(patch), _ScopeNode()),
        "Add right-flipper state to InputState",
    )
    assert "BootConfig" in violation
    assert "InputState" not in violation, "a modified declaration is not a deletion"


def test_intentional_api_removal_is_allowed() -> None:
    patch = "-export interface LegacyApi { old: boolean }\n"
    violation = _agent()._semantic_scope_violation(
        _ScopeCtx(_DiffRepo(patch), _ScopeNode()),
        "Remove the deprecated LegacyApi interface",
    )
    assert violation == ""


def test_scope_check_uses_integrated_baseline_before_preserved_wip() -> None:
    repo = _WipDiffRepo(
        head_patch="-export function speculativeStepWorld() {}\n",
        baseline_patch="+export function wantedApi() {}\n",
        commits=[_WipCommit("wip-1", "wip: preserve attempt 1 for task", "node-1")],
    )

    violation = _agent()._semantic_scope_violation(
        _ScopeCtx(repo, _WipScopeNode()),  # type: ignore[arg-type]
        "Implement the physics loop",
    )

    assert violation == "", "a WIP-only API is not part of the integrated contract"


def test_leaf_repairs_defer_runtime_and_quality_gates_to_integration() -> None:
    ctx = _GateCtx(
        _GateNode(
            [
                "schema",
                "lint",
                "types",
                "build",
                "unit",
                "browser",
                "visual",
                "load_perf",
                "project_contract",
            ]
        )
    )
    assert _agent()._coding_gate_names(ctx) == [
        "schema",
        "lint",
        "types",
        "build",
        "unit",
        "project_contract",
    ]


def test_obsolete_diagnostic_repair_skips_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def passing_gates(self, ctx, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs["gate_names"])
        return ValidationReport(verdicts=[Verdict.passing("types")])

    monkeypatch.setattr(_Coding, "run_gates", passing_gates)
    ctx = _RepairCtx(
        node=_RepairNode(
            "Fix TS5097 import extension error",
            ["types", "browser", "load_perf"],
        ),
        spec={"objective": "Remove the .ts extension so compilation passes"},
        repo=_CleanRepo(),
    )

    result = _agent()._already_satisfied_repair(ctx, "Fix the compiler diagnostic")

    assert result is not None and result.success
    assert result.data == {"already_satisfied": True, "preflight": True}
    assert calls == [["types"]], "preflight uses leaf gates, not browser/performance"


def test_green_generic_gates_do_not_skip_feature_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def should_not_run(self, ctx, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        return ValidationReport(verdicts=[Verdict.passing("types")])

    monkeypatch.setattr(_Coding, "run_gates", should_not_run)
    ctx = _RepairCtx(
        node=_RepairNode("Implement touch controls", ["types", "unit"]),
        spec={"objective": "Add multitouch flipper and plunger controls"},
        repo=_CleanRepo(),
    )

    assert _agent()._already_satisfied_repair(ctx, "Implement touch controls") is None
    assert not called, "green generic gates cannot prove a feature exists"


def test_clean_verify_node_can_finish_without_manufacturing_a_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def passing_gates(self, ctx, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs["gate_names"])
        return ValidationReport(verdicts=[Verdict.passing("types")])

    monkeypatch.setattr(_Coding, "run_gates", passing_gates)
    ctx = _RepairCtx(
        node=_RepairNode(
            "Verify no Math.random in rendering and input paths",
            ["lint", "types", "browser"],
        ),
        spec={"objective": "Confirm deterministic screenshot rendering"},
        repo=_CleanRepo(),
    )

    report = _agent()._verify_unchanged_audit(ctx)

    assert report is not None and report.passed
    assert calls == [["lint", "types"]]


def test_implementation_node_cannot_claim_a_clean_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def should_not_run(self, ctx, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        return ValidationReport(verdicts=[Verdict.passing("types")])

    monkeypatch.setattr(_Coding, "run_gates", should_not_run)
    ctx = _RepairCtx(
        node=_RepairNode("Implement deterministic screenshot mode", ["types"]),
        spec={"objective": "Build the screenshot feature"},
        repo=_CleanRepo(),
    )

    assert _agent()._verify_unchanged_audit(ctx) is None
    assert not called


def test_decomposed_parent_validates_children_instead_of_replaying_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], bool]] = []

    def passing_gates(self, ctx, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((kwargs["gate_names"], kwargs["fail_fast"]))
        return ValidationReport(verdicts=[Verdict.passing("browser")])

    monkeypatch.setattr(_Coding, "run_gates", passing_gates)
    children = {
        "leaf-a": SimpleNamespace(status="succeeded"),
        "leaf-b": SimpleNamespace(status="succeeded"),
    }
    ctx = SimpleNamespace(
        node=_RepairNode(
            "Implement deterministic screenshot mode",
            ["lint", "types", "browser"],
        ),
        spec={
            "decomposed": True,
            "decomposition_children": ["leaf-a", "leaf-b"],
        },
        repo=_CleanRepo(),
        graph=SimpleNamespace(get=children.__getitem__),
        logger=lambda: SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )

    result = _agent()._already_satisfied_decomposition(ctx)

    assert result is not None and result.success
    assert result.data == {"already_satisfied": True, "decomposition": True}
    assert calls == [(["lint", "types", "browser"], False)]


def test_decomposed_parent_waits_for_every_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _Coding,
        "run_gates",
        lambda *_args, **_kwargs: pytest.fail("gates ran before children finished"),
    )
    children = {
        "leaf-a": SimpleNamespace(status="succeeded"),
        "leaf-b": SimpleNamespace(status="pending"),
    }
    ctx = SimpleNamespace(
        node=_RepairNode("Implement feature", ["types"]),
        spec={
            "decomposed": True,
            "decomposition_children": ["leaf-a", "leaf-b"],
        },
        repo=_CleanRepo(),
        graph=SimpleNamespace(get=children.__getitem__),
    )

    assert _agent()._already_satisfied_decomposition(ctx) is None


def test_eslint_parser_repair_is_a_diagnostic_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _Coding,
        "run_gates",
        lambda self, ctx, **kwargs: ValidationReport(
            verdicts=[Verdict.passing("lint"), Verdict.passing("types")]
        ),
    )
    ctx = _RepairCtx(
        node=_RepairNode(
            "Add config test to the TypeScript include array",
            ["lint", "types"],
        ),
        spec={
            "objective": "Make ESLint recognize the file",
            "acceptance": ["eslint reports 0 parser errors"],
        },
        repo=_CleanRepo(),
    )

    result = _agent()._already_satisfied_repair(ctx, "Resolve the parser error")

    assert result is not None and result.data["already_satisfied"]


def test_browser_harness_repair_rechecks_the_failed_gate_before_editing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def passing_gates(self, ctx, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs["gate_names"])
        return ValidationReport(verdicts=[Verdict.passing("smoke")])

    monkeypatch.setattr(_Coding, "run_gates", passing_gates)
    ctx = _RepairCtx(
        node=_RepairNode("Fix browser failures in QA", ["browser", "smoke"]),
        spec={"objective": "The application fails when driven in a real browser"},
        repo=_CleanRepo(),
    )

    result = _agent()._already_satisfied_repair(ctx, "Fix browser failures")

    assert result is not None and result.data["already_satisfied"]
    assert calls == [["browser", "smoke"]]


# --------------------------------------------------------------------------
# The protocol carrying it
# --------------------------------------------------------------------------


def test_the_schema_offers_the_field_without_requiring_it() -> None:
    """A model that has what it needs must not have to say so."""
    assert "need_files" in EDIT_PLAN_SCHEMA["properties"]
    assert "need_files" not in EDIT_PLAN_SCHEMA["required"]


def test_a_plan_without_the_field_asks_for_nothing() -> None:
    plan = EditPlan.from_payload(
        {"summary": "s", "edits": [{"path": "a.ts", "op": "write", "content": "x"}]}
    )
    assert plan.need_files == []


# --------------------------------------------------------------------------
# Provisional edits must never become the deliverable
# --------------------------------------------------------------------------


def test_the_refusal_message_distinguishes_withheld_from_nonexistent() -> None:
    """It told the model "every file available to you is already included above"
    when the files were grantable and merely withheld by the request budget.

    A message the model cannot act on is answered by asking again, which costs
    another full-context call on a frontier rung. This asserts the two branches
    exist and say different things.
    """
    source = inspect.getsource(CodingAgent.implement)
    assert "nothing further will be provided" in source, "the withheld case must say so"
    assert "You already have the full contents of" in source, (
        "and must name what the model does have, or it simply asks again"
    )
    assert "exists in" in source, "the nonexistent case must say so"
    assert "already included above" not in source, (
        "the old message claimed files were present when they had been withheld"
    )


# --------------------------------------------------------------------------
# Acceptance criteria that name tests
# --------------------------------------------------------------------------


@dataclass
class _Node:
    acceptance: list[str]


@dataclass
class _NodeCtx:
    node: _Node


def test_criteria_demanding_tests_are_not_met_by_code_alone() -> None:
    """The plunger node shipped 52 lines and no test at all.

    Three of its six criteria said "asserted by test". Every gate passed --
    they measure lint, types, build and whether the *existing* tests still run --
    and the node was marked succeeded having implemented about a third of what it
    was asked for. The graph then builds on it.
    """
    ctx = _NodeCtx(_Node([
        "A minimum-charge release does not clear the gate, so a weak launch is "
        "possible and the ball drains back - asserted by test.",
    ]))
    message = _agent()._unmet_test_requirement(ctx, ["src/game/plunger.ts"])
    assert "adds none" in message
    assert "asserted by test" in message, "it must quote the criterion it is enforcing"


def test_adding_a_test_file_satisfies_it() -> None:
    ctx = _NodeCtx(_Node(["behaviour is asserted by test"]))
    assert _agent()._unmet_test_requirement(
        ctx, ["src/game/plunger.ts", "src/game/plunger.test.ts"]
    ) == ""


def test_a_node_never_asked_for_tests_is_left_alone() -> None:
    """Narrow on purpose: it must not argue with a node that has no such criteria."""
    ctx = _NodeCtx(_Node(["The README documents the controls."]))
    assert _agent()._unmet_test_requirement(ctx, ["README.md"]) == ""

    assert _agent()._unmet_test_requirement(_NodeCtx(_Node([])), ["a.ts"]) == ""


def test_python_and_directory_test_layouts_count_too() -> None:
    ctx = _NodeCtx(_Node(["asserted by test"]))
    for path in ("tests/test_thing.py", "src/spec/thing.spec.ts", "pkg/thing_test.py"):
        assert _agent()._unmet_test_requirement(ctx, [path]) == "", path


# --------------------------------------------------------------------------
# Files every coding node needs regardless of what the planner declared
# --------------------------------------------------------------------------


@dataclass
class _PathCtx:
    root: Path
    spec: dict
    node: object
    graph: object = None


@dataclass
class _DeplessNode:
    deps: list


def test_build_manifests_are_included_without_being_asked_for(tmp_path: Path) -> None:
    """A node spent its last file request on `package.json` and `tsconfig.json`.

    It was refused -- the budget counts requests, not files -- and returned no
    edits at all, losing the round. Together those files are a few hundred
    tokens, and whether the project is ESM, what `strict` is set to and where
    the test globs point are not discoveries. They are the ground rules.
    """
    (tmp_path / "package.json").write_text('{"type": "module"}')
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {"strict": true}}')
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.ts").write_text("x")

    ctx = _PathCtx(root=tmp_path, spec={"paths": ["src/a.ts"]}, node=_DeplessNode([]))
    paths = _agent()._relevant_paths(ctx)

    assert "package.json" in paths and "tsconfig.json" in paths
    assert paths[0] == "src/a.ts", "the declared paths still come first"


def test_manifests_that_do_not_exist_are_not_invented(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.ts").write_text("x")
    ctx = _PathCtx(root=tmp_path, spec={"paths": ["src/a.ts"]}, node=_DeplessNode([]))
    assert _agent()._relevant_paths(ctx) == ["src/a.ts"]


def test_work_survives_a_spent_request_budget_but_a_phantom_request_does_not() -> None:
    """Two cases that look alike and must not be treated alike.

    A node was handed a 1489-character diagnosis, asked for one more file, was
    refused, had its edits discarded and escalated -- the advice was bought and
    thrown away. But the earlier false success was also `need_files` plus edits:
    haiku's `// Placeholder` reached a commit that way.

    `granted` separates them. A model that has spent its requests got what it
    asked for and is now working; a model whose request was never grantable is
    asking for files that do not exist, which is the confusion that produced the
    placeholder.
    """
    source = inspect.getsource(CodingAgent.implement)
    assert "plan.edits and granted >= MAX_FILE_REQUESTS" in source, (
        "keeping edits must be gated on a spent budget, not merely on having edits"
    )
    keep = source.index("keeping the edits it did make")
    discard = source.index("no file requests left and no edits to keep")
    assert keep < discard, "the keep branch must precede the discard branch"


def test_opencode_retry_recognises_committed_work_preserved_on_node_branch(
    tmp_path: Path,
) -> None:
    """A failed integrated gate resets main, not the persistent node branch."""
    main = Repo(tmp_path / "workspace").init()
    node = main.ensure_worktree(
        tmp_path / "node-worktree", "forge/node/preserved", base="HEAD"
    )
    (node.path / "feature.ts").write_text("export const preserved = true;\n")
    node.commit("feat: preserved result")
    ctx = SimpleNamespace(
        repo=node,
        config=SimpleNamespace(workspace_dir=main.path),
    )

    assert CodingAgent._worktree_changes(ctx) == ["feature.ts"]


def test_unresolved_high_severity_findings_are_gaps_not_noise() -> None:
    """A reviewer that said `request_changes` outranks a model saying "done".

    Appearance and behaviour have no deterministic gate behind them, so
    `report.passed` means only that the code compiled and booted. Without this
    the model's boolean was the whole verdict, and it returned `complete` with
    36 findings still open, including "missing plunger lane, outlanes,
    drop-target bank, ramps, multiplier lane and drain" on a pinball table.
    """
    from forge.agents.goal import _blocking_findings, _merge_finding_gaps
    from forge.memory.store import finding

    critical = finding("Incomplete Table Geometry", "no plunger lane or drain",
                       severity="critical", source="visual:node_x")
    low = finding("Baseline name is unclear", "cosmetic", severity="low",
                  source="visual:node_x")

    blocking = _blocking_findings([critical, low])
    assert [record.title for record in blocking] == ["Incomplete Table Geometry"]

    gaps = _merge_finding_gaps([], blocking)
    assert [gap["what"] for gap in gaps] == ["Incomplete Table Geometry"]
    assert gaps[0]["essential"] is True
    # Carries the id so closing the work also closes the finding.
    assert gaps[0]["finding_id"] == critical.id


def test_a_gap_the_model_already_named_is_not_duplicated() -> None:
    """Otherwise the gap list doubles every round and no-progress never trips."""
    from forge.agents.goal import _merge_finding_gaps
    from forge.memory.store import finding

    record = finding("Missing drain", "the ball never drains", severity="high")
    gaps = _merge_finding_gaps([{"what": "Missing drain", "essential": True}], [record])

    assert len(gaps) == 1


def test_stale_gate_findings_do_not_block_the_goal() -> None:
    """A gate re-runs; its old records are a log, not a backlog.

    The project this was written against carried 30 unresolved `gate:types`
    findings like "'add' is declared but its value is never read" against 4
    real review findings. Blocking on all of them buries the four that matter
    under compile errors that no longer reproduce, and `report.passed` already
    covers the ones that do.
    """
    from forge.agents.goal import _blocking_findings
    from forge.memory.store import finding

    stale = finding("types: error TS6133: 'add' is declared but never read",
                    "", severity="high", source="gate:types")
    real = finding("Presentation Layer Unapplied", "no lighting or ball shadow",
                   severity="critical", source="visual:node_x")

    assert [r.title for r in _blocking_findings([stale, real])] == [
        "Presentation Layer Unapplied"
    ]


def test_a_gap_cannot_be_closed_by_writing_documentation() -> None:
    """Observed live: three gaps closed by editing README.md.

    "Missing Modern Presentation & Visual Effects", "Broken Game Flow & HUD"
    and "Incomplete Table Geometry & Scoring Architecture" were routed to the
    documentation agent, which changed README.md and docs/project-memory.md,
    reported "documentation updated (2 file(s))" and marked all three
    succeeded. The game was untouched and the gaps counted as closed.
    """
    from forge.agents.goal import _gap_kind

    assert _gap_kind({"what": "Missing lighting", "kind": "document"}) == "implement"
    assert _gap_kind({"what": "Missing lighting", "kind": "review"}) == "implement"
    assert _gap_kind({"what": "Missing lighting"}) == "implement"
    # A specialist that does change the product is left alone.
    assert _gap_kind({"what": "Flipper physics wrong", "kind": "debug"}) == "debug"
    assert _gap_kind({"what": "No coverage", "kind": "test_author"}) == "test_author"
