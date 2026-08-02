"""Gates that run a command and read its exit code.

Compilation, tests, linting, type checking and formatting are all the same
gate with a different command and a different output parser. Factoring them this
way means a new language is supported by adding entries to a command table, not
by writing a new gate.

Output parsing deserves the attention it gets here. The difference between
handing a model 200 lines of raw pytest output and handing it eight structured
issues with file and line is roughly an order of magnitude in tokens, and a
noticeably higher chance of a correct fix on the first attempt.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ...errors import CommandTimeout, SandboxError
from ...obs.log import get_logger
from ..gate import Gate, GateContext, register
from ..types import Issue, Severity, Verdict

log = get_logger("validation.command")


class CommandGate(Gate):
    """Base for anything that shells out.

    Subclasses supply :meth:`command` and optionally :meth:`parse`.
    """

    #: Config key under ``toolchain.commands`` holding the command, if any.
    command_key: str = ""
    #: Fallbacks tried in order when the toolchain has no explicit command.
    fallbacks: tuple[tuple[str, str], ...] = ()
    timeout: float = 900.0

    def command(self, ctx: GateContext) -> str | None:
        explicit = ctx.setting("command")
        if explicit:
            return str(explicit)
        commands = ctx.toolchain.get("commands", {})
        if self.command_key and commands.get(self.command_key):
            return str(commands[self.command_key])
        for marker, command in self.fallbacks:
            if ctx.sandbox.exists(marker):
                return command
        return None

    def applicable(self, ctx: GateContext) -> bool:
        command = self.command(ctx)
        if command is None:
            return False
        return _tool_available(ctx, command)

    def run(self, ctx: GateContext) -> Verdict:
        command = self.command(ctx)
        if command is None:
            return Verdict.skip(self.name, "no command configured for this project")
        if not _tool_available(ctx, command):
            return Verdict.skip(self.name, f"{_binary_of(command)!r} is not installed on this host")
        try:
            result = ctx.sandbox.exec(
                command, shell=True, timeout=min(self.timeout, ctx.timeout)
            )
        except CommandTimeout as exc:
            return Verdict.failing(
                self.name,
                f"{self.name} timed out after {min(self.timeout, ctx.timeout):.0f}s",
                evidence=str(exc.context.get("tail", "")),
                detail={"timeout": True},
            )
        except SandboxError as exc:
            return Verdict.error(self.name, f"could not run {self.name}: {exc}")

        # Exit 127 is the shell's "command not found". A missing tool is not a
        # failing check, and treating it as one is expensive out of all
        # proportion: observed live, an absent `ruff` failed a node four times,
        # escalated it to the most costly rung on every retry, and finally
        # blocked the project -- over a linter that was never installed.
        # `_looks_missing` is a text match, so it has to be gated on the command
        # having actually failed -- as the `_looks_uninstalled` branch below
        # already is. A test suite whose own output quotes "command not found",
        # or a build log relaying a child process's error, was reported as
        # SKIPPED; a skipped verdict counts as ok, so a real failure passed
        # validation. Exit 127 is the shell speaking and needs no such guard.
        if result.returncode == 127 or (not result.ok and _looks_missing(result.combined)):
            return Verdict.skip(
                self.name, f"{_binary_of(command)!r} is not installed on this host"
            )
        if not result.ok and _looks_uninstalled(result.combined):
            return Verdict.skip(
                self.name,
                f"`{command}` cannot run yet: its packages are not installed "
                "(dependencies have not been fetched for this project)",
            )

        issues = self.parse(result.combined, ctx)
        passed = result.ok and not any(i.severity in (Severity.HIGH, Severity.CRITICAL) for i in issues)
        summary = (
            f"`{command}` succeeded"
            if passed
            else f"`{command}` failed with exit {result.returncode}"
        )
        return Verdict(
            gate=self.name,
            passed=passed,
            summary=summary,
            evidence=result.combined if not passed else "",
            issues=issues,
            duration=result.duration,
            detail={"exit_code": result.returncode, "command": command},
        )

    def parse(self, output: str, ctx: GateContext) -> list[Issue]:
        return []


# --------------------------------------------------------------------------
# Concrete gates
# --------------------------------------------------------------------------

#: Shell wording for a missing executable, across the shells we might hit.
#: Cheap pre-filter only. The trailing colons used to be doing the discriminating,
#: and they excluded dash -- `sh: 1: ruff: not found`, which is what /bin/sh
#: prints on Debian and therefore what most of these actually look like. The
#: shape check in `_looks_missing` is what separates a shell's diagnostic from a
#: program quoting one, so these can be plain substrings.
_MISSING_MARKERS = ("command not found", "not found", "no such file or directory", "is not recognized")

#: The same condition one level up: the *launcher* is installed but the package
#: it should run is not. `npx --no-install tsc` exits 1, not 127, and says
#: nothing a shell would recognise, so neither the exit code nor the probe on
#: `npx` catches it. Observed live: a scaffolded project has package.json but no
#: node_modules yet, the types gate "failed", and the node escalated local ->
#: codex -> claude over a toolchain that was simply not installed yet.
_MISSING_PACKAGE_MARKERS = (
    "npx canceled due to missing packages",
    "could not determine executable to run",
    "no matching version found for",
    "cannot find module 'typescript'",
    "this is not the tsc command you are looking for",
    "is not recognized as an internal or external command",
)


def _binary_of(command: str) -> str:
    """The executable a shell command starts with, ignoring env prefixes."""
    for token in command.strip().split():
        if "=" in token and not token.startswith("-"):
            continue  # VAR=value prefix
        return token
    return command.strip()


def _tool_available(ctx: GateContext, command: str) -> bool:
    """Is the command's executable present in the sandbox?

    Checked before running, so a project that has no linter installed reports a
    clear skip rather than a failure the agents then try to "fix" by rewriting
    perfectly good code.
    """
    binary = _binary_of(command)
    # Shell builtins and constructs are always available.
    if binary in ("true", "false", "exit", "echo", "cd", ":") or not binary:
        return True
    try:
        probe = ctx.sandbox.exec(f"command -v {binary} >/dev/null 2>&1", shell=True, timeout=30)
    except Exception:  # pragma: no cover - sandbox unavailable
        return True
    return probe.ok


#: A shell saying a program is absent, as opposed to a program saying the words.
#: The shape is the whole point: `sh: 1: ruff: not found` is a diagnostic, and
#: `FAIL src/shell.test.ts > handles a missing binary: command not found` is a
#: test name. A substring search cannot tell them apart, and got it wrong in the
#: direction that matters -- a failing gate reported SKIPPED, which counts as
#: ok, which marks a node succeeded on code that does not work.
_SHELL_DIAGNOSTIC = re.compile(
    # POSIX: `sh: 1: ruff: not found`, `bash: line 2: ruff: command not found`
    r"^\s*(?:[\w./+-]+:\s*)*(?:line \d+:\s*)?[\w./+-]+:\s*"
    r"(?:command not found|not found|no such file or directory)\s*$"
    # cmd.exe: `'ruff' is not recognized as an internal or external command,`
    r"|^\s*'?[\w./+-]+'?\s+is not recognized[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _looks_missing(output: str) -> bool:
    lowered = output.lower()
    if not any(marker in lowered for marker in _MISSING_MARKERS):
        return False
    return bool(_SHELL_DIAGNOSTIC.search(output))


def _looks_uninstalled(output: str) -> bool:
    """The launcher ran but the package it needed is not installed."""
    lowered = output.lower()
    return any(marker in lowered for marker in _MISSING_PACKAGE_MARKERS)


_GENERIC_LOCATION = re.compile(
    r"^(?P<path>[\w./\-]+\.\w+):(?P<line>\d+)(?::(?P<col>\d+))?[:\s]+(?P<rest>.+)$"
)


def _parse_locations(output: str, *, severity: str = Severity.HIGH, limit: int = 40) -> list[Issue]:
    """Pull ``path:line: message`` triples out of arbitrary tool output.

    Deliberately generic. Writing a bespoke parser per tool is a maintenance
    burden that pays off only for the tools whose output is genuinely
    structured; for everything else this recovers the location, which is the
    part that matters.
    """
    issues: list[Issue] = []
    for line in output.splitlines():
        match = _GENERIC_LOCATION.match(line.strip())
        if not match:
            continue
        rest = match.group("rest").strip()
        level = severity
        lowered = rest.lower()
        if lowered.startswith("warning") or " warning" in lowered[:30]:
            level = Severity.LOW
        issues.append(
            Issue(
                message=rest[:300],
                severity=level,
                path=match.group("path"),
                line=int(match.group("line")),
            )
        )
        if len(issues) >= limit:
            break
    return issues


@register
class FormatGate(CommandGate):
    name = "format"
    description = "Code formatting is consistent"
    order = 10
    blocking = False  # formatting is auto-fixable; never stall a node on it
    command_key = "format_check"
    fallbacks = (
        ("pyproject.toml", "ruff format --check . 2>&1 || true"),
        ("package.json", "npx --no-install prettier --check . 2>&1 || true"),
    )
    timeout = 180.0


@register
class LintGate(CommandGate):
    name = "lint"
    description = "Static analysis passes"
    order = 20
    command_key = "lint"
    fallbacks = (
        ("pyproject.toml", "ruff check ."),
        ("Cargo.toml", "cargo clippy -- -D warnings"),
    )
    timeout = 300.0

    def parse(self, output: str, ctx: GateContext) -> list[Issue]:
        return _parse_locations(output, severity=Severity.MEDIUM)


@register
class TypeGate(CommandGate):
    name = "types"
    description = "Type checking passes"
    order = 30
    command_key = "types"
    fallbacks = (("tsconfig.json", "npx --no-install tsc --noEmit"),)
    timeout = 600.0

    def parse(self, output: str, ctx: GateContext) -> list[Issue]:
        # tsc writes `file.ts(line,col): error TSxxxx: message`, with
        # parentheses where the generic parser expects colons -- so every
        # TypeScript error arrived as unstructured text and the issue count
        # read zero on a genuinely failing gate.
        issues = [
            Issue(
                path=match.group("path"),
                line=int(match.group("line")),
                message=match.group("rest").strip(),
                severity=Severity.HIGH,
            )
            for match in _TSC_LOCATION.finditer(output)
        ]
        return issues or _parse_locations(output, severity=Severity.HIGH)


#: `src/game.ts(12,5): error TS2322: Type 'string' is not assignable ...`
_TSC_LOCATION = re.compile(
    r"^(?P<path>[\w./\-]+\.\w+)\((?P<line>\d+),(?P<col>\d+)\):\s*(?P<rest>.+)$",
    re.MULTILINE,
)


@register
class BuildGate(CommandGate):
    name = "build"
    description = "The project compiles or bundles"
    order = 40
    command_key = "build"
    fallbacks = (
        ("Cargo.toml", "cargo build"),
        ("go.mod", "go build ./..."),
        ("Makefile", "make"),
    )
    timeout = 1200.0

    def parse(self, output: str, ctx: GateContext) -> list[Issue]:
        return _parse_locations(output, severity=Severity.CRITICAL)


@register
class UnitTestGate(CommandGate):
    name = "unit"
    description = "Unit tests pass"
    order = 50
    command_key = "unit"
    fallbacks = (
        ("pyproject.toml", "python -m pytest -q"),
        ("Cargo.toml", "cargo test"),
        ("go.mod", "go test ./..."),
    )
    timeout = 1800.0

    def parse(self, output: str, ctx: GateContext) -> list[Issue]:
        """Extract failing test names.

        A model fixing a test failure needs to know *which* tests failed far
        more than it needs the stack traces, which it can usually reconstruct
        from the code. Naming them first makes the evidence section that
        follows much cheaper to use.
        """
        issues: list[Issue] = []
        patterns = (
            re.compile(r"^FAILED\s+(?P<name>[\w./:\[\]\-]+)(?:\s+-\s+(?P<msg>.*))?$"),  # pytest
            re.compile(r"^\s*✕\s+(?P<name>.+)$"),  # vitest / jest
            re.compile(r"^---- (?P<name>[\w:]+) stdout ----$"),  # cargo
            re.compile(r"^\s*--- FAIL: (?P<name>\w+)"),  # go
        )
        for line in output.splitlines():
            for pattern in patterns:
                match = pattern.match(line.rstrip())
                if match:
                    name = match.group("name").strip()
                    message = (match.groupdict().get("msg") or "").strip()
                    issues.append(
                        Issue(
                            message=f"test failed: {name}" + (f" -- {message[:200]}" if message else ""),
                            severity=Severity.CRITICAL,
                            rule="test",
                        )
                    )
                    break
            if len(issues) >= 40:
                break
        return issues


@register
class IntegrationTestGate(CommandGate):
    name = "integration"
    description = "Integration tests pass"
    order = 60
    command_key = "integration"
    timeout = 2400.0

    def parse(self, output: str, ctx: GateContext) -> list[Issue]:
        return UnitTestGate().parse(output, ctx)


@register
class CoverageGate(CommandGate):
    """Enforces a coverage floor, if one is configured.

    Off by default. A coverage floor imposed on a project that does not want
    one produces tests written to satisfy the number rather than to catch bugs,
    which is worse than no floor at all.
    """

    name = "coverage"
    description = "Test coverage meets the configured floor"
    _output = ""
    order = 70
    blocking = False
    command_key = "coverage"
    timeout = 1800.0

    def applicable(self, ctx: GateContext) -> bool:
        return bool(ctx.setting("floor", 0)) and super().applicable(ctx)

    def parse(self, output: str, ctx: GateContext) -> list[Issue]:
        # The percentage is in the output, and `CommandGate.run` keeps evidence
        # only when the command *failed*. Reading it from the verdict therefore
        # worked in exactly the case that does not matter: the floor was never
        # once enforced against a test run that passed. A fresh gate instance is
        # created per validation run, so holding it here is safe.
        self._output = output
        return super().parse(output, ctx)

    def run(self, ctx: GateContext) -> Verdict:
        self._output = ""
        verdict = super().run(ctx)
        floor = float(ctx.setting("floor", 0))
        percent = _extract_coverage(self._output or verdict.evidence or verdict.summary)
        if percent is None:
            verdict.summary += " (coverage percentage not found in output)"
            return verdict
        verdict.score = percent
        if percent < floor:
            verdict.passed = False
            verdict.summary = f"coverage {percent:.1f}% is below the {floor:.1f}% floor"
        return verdict


_COVERAGE = re.compile(r"(?:TOTAL|All files)[^\n]*?(\d+(?:\.\d+)?)\s*%")

#: The istanbul/v8 table, which is what every JS project prints and which
#: carries no `%` on the row itself -- the sign is in the *header* (`% Stmts`).
#: The percent-anchored pattern above therefore never matched it, so on the one
#: stack Forge builds for, a configured floor did nothing at all.
_COVERAGE_TABLE = re.compile(r"^\s*All files\s*\|\s*(\d+(?:\.\d+)?)", re.MULTILINE)


def _extract_coverage(output: str) -> float | None:
    match = _COVERAGE.search(output)
    if match:
        return float(match.group(1))
    match = _COVERAGE_TABLE.search(output)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)%\s+(?:coverage|covered)", output, re.IGNORECASE)
    return float(match.group(1)) if match else None


@register
class SchemaGate(Gate):
    """Validates that declared project metadata files are well-formed.

    Cheap, instant, and catches a whole class of failure that otherwise only
    surfaces when a build breaks minutes later: a model writing a ``package.json``
    with a trailing comma.
    """

    name = "schema"
    description = "JSON and config files parse"
    order = 5
    inputs = ("*.json", "*.jsonc")

    def run(self, ctx: GateContext) -> Verdict:
        issues: list[Issue] = []
        checked = 0
        for path in sorted(ctx.root.rglob("*.json")):
            rel = path.relative_to(ctx.root)
            if any(part in {"node_modules", ".git", ".forge", "dist", "build"} for part in rel.parts):
                continue
            checked += 1
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                issues.append(
                    Issue(
                        message=f"invalid JSON: {exc.msg}",
                        severity=Severity.CRITICAL,
                        path=rel.as_posix(),
                        line=exc.lineno,
                    )
                )
            except OSError:  # pragma: no cover
                continue
        return Verdict(
            gate=self.name,
            passed=not issues,
            summary=f"{checked} JSON file(s) checked, {len(issues)} invalid",
            issues=issues,
        )


def custom_command_gate(name: str, command: str, **kwargs: Any) -> type[Gate]:
    """Build and register a gate from a command string.

    Used by the scaffolding agent when it discovers a project-specific check
    (``npm run e2e``), so project-defined validation becomes a first-class gate
    with caching and reporting rather than an ad-hoc shell call.
    """

    class _Custom(CommandGate):
        pass

    _Custom.name = name
    _Custom.description = kwargs.get("description", f"custom gate: {command}")
    _Custom.order = kwargs.get("order", 80)
    _Custom.blocking = kwargs.get("blocking", True)
    _Custom.timeout = kwargs.get("timeout", 900.0)
    _Custom.command = lambda self, ctx: command  # type: ignore[assignment]
    _Custom.applicable = lambda self, ctx: True  # type: ignore[assignment]
    return register(_Custom)
