"""Gate verdicts.

A verdict is designed to be useful to three different readers:

* the **scheduler**, which needs one bit -- did it pass -- plus whether retrying
  could help;
* the **model**, which needs the smallest piece of output that explains the
  failure, because handing it 4000 lines of test log wastes most of the context
  budget on scrollback;
* the **human**, who needs a one-line summary and a path to the full artefact.

So a verdict carries a summary, a bounded ``evidence`` string sized for a
prompt, and a pointer to the complete output on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..util.proc import clamp_output


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


#: Severities that must block a node from succeeding.
BLOCKING = frozenset({Severity.HIGH, Severity.CRITICAL})


@dataclass(slots=True)
class Issue:
    """One specific problem found by a gate."""

    message: str
    severity: str = Severity.MEDIUM
    path: str | None = None
    line: int | None = None
    rule: str | None = None

    def render(self) -> str:
        location = ""
        if self.path:
            location = f"{self.path}:{self.line}" if self.line else self.path
            location = f"{location}: "
        rule = f" [{self.rule}]" if self.rule else ""
        return f"{location}{self.message}{rule}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "severity": self.severity,
            "path": self.path,
            "line": self.line,
            "rule": self.rule,
        }


@dataclass(slots=True)
class Verdict:
    """The result of running one gate."""

    gate: str
    passed: bool
    summary: str = ""
    #: Bounded excerpt suitable for a prompt. Never the whole log.
    evidence: str = ""
    issues: list[Issue] = field(default_factory=list)
    #: Numeric result where one exists: coverage %, pixel difference, ms.
    score: float | None = None
    duration: float = 0.0
    #: Files produced: screenshots, videos, reports.
    artifacts: list[str] = field(default_factory=list)
    #: True when the gate could not run at all, as opposed to running and
    #: failing. Errors are retried; failures are fixed.
    errored: bool = False
    skipped: bool = False
    skip_reason: str = ""
    cached: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking_issues(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity in BLOCKING]

    @property
    def ok(self) -> bool:
        return self.passed or self.skipped

    def render(self, *, max_evidence: int = 4000) -> str:
        """The form handed to a model that must fix the failure."""
        status = "SKIPPED" if self.skipped else ("PASS" if self.passed else "FAIL")
        lines = [f"### {self.gate}: {status}"]
        if self.summary:
            lines.append(self.summary)
        if self.score is not None:
            lines.append(f"score: {self.score}")
        if self.issues:
            lines.append("")
            lines += [f"- {issue.render()}" for issue in self.issues[:40]]
            if len(self.issues) > 40:
                lines.append(f"- ... and {len(self.issues) - 40} more")
        if self.evidence and not self.passed:
            # Both ends, not the tail. The tail is right for a test runner, which
            # summarises at the end, and wrong for a compiler, which lists errors
            # in file order so the *first* one is the cause and the rest are
            # cascade. Showing a model only the tail of a `tsc` run hands it the
            # downstream noise and truncates away the line it needs -- which is
            # what "could not find the unbalanced brace in three rounds" actually
            # looked like from the model's side.
            #
            # `clamp_output` in util/proc already states this rule; this was the
            # one place that contradicted it.
            excerpt, _ = clamp_output(self.evidence, max_evidence)
            lines += ["", "```", excerpt.strip(), "```"]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "summary": self.summary,
            "score": self.score,
            "duration": round(self.duration, 3),
            "issues": [issue.to_dict() for issue in self.issues],
            "artifacts": self.artifacts,
            "errored": self.errored,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "cached": self.cached,
            "detail": self.detail,
        }

    @classmethod
    def passing(cls, gate: str, summary: str = "", **kwargs: Any) -> Verdict:
        return cls(gate=gate, passed=True, summary=summary, **kwargs)

    @classmethod
    def failing(cls, gate: str, summary: str, **kwargs: Any) -> Verdict:
        return cls(gate=gate, passed=False, summary=summary, **kwargs)

    @classmethod
    def skip(cls, gate: str, reason: str) -> Verdict:
        return cls(gate=gate, passed=True, skipped=True, skip_reason=reason, summary=f"skipped: {reason}")

    @classmethod
    def error(cls, gate: str, summary: str, **kwargs: Any) -> Verdict:
        return cls(gate=gate, passed=False, errored=True, summary=summary, **kwargs)


@dataclass(slots=True)
class ValidationReport:
    """Every verdict from one validation pass."""

    verdicts: list[Verdict] = field(default_factory=list)
    duration: float = 0.0

    @property
    def passed(self) -> bool:
        return all(v.ok for v in self.verdicts)

    @property
    def failures(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.ok]

    @property
    def errors(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.errored]

    def render(self, *, failures_only: bool = True) -> str:
        selected = self.failures if failures_only else self.verdicts
        if not selected:
            return "All gates passed."
        return "\n\n".join(v.render() for v in selected)

    def summary_line(self) -> str:
        passed = sum(1 for v in self.verdicts if v.ok)
        return f"{passed}/{len(self.verdicts)} gates passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "duration": round(self.duration, 3),
            "summary": self.summary_line(),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }
