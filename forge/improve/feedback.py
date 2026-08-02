"""Platform feedback: finding defects in Forge itself, not in what it builds.

Forge already learns two things. The routing policy learns which model handles
which task class, and the lessons library accumulates transferable knowledge
about building software. Both are about *the work*.

Neither can see a defect in Forge. This module is the missing third channel, and
it exists because of a specific, humbling observation: the lessons library
already contained

    "A gate whose tool is not installed must skip, never fail"

while the types gate was failing nodes over an uninstalled TypeScript compiler
and escalating them to the costliest rung. The knowledge was present and
structurally unable to act, because a lesson is advice to a model and the fix
was a change to a tuple of strings. No amount of model self-improvement reaches
that. It has to leave the system and land in the repository.

So the detectors here look for shapes that mean "the platform is malfunctioning"
rather than "this project is hard":

* a node failing repeatedly with an *identical* error signature -- a real
  problem varies its symptoms; a bug repeats verbatim
* a rung that has never once succeeded for a task class
* a gate that has never once passed
* deterministic work discarded and redone on a more expensive rung
* a sustained cloud fraction far above the configured target
* retry storms: many failures in a short window on one node

Every detector is arithmetic over the ledger. Nothing here asks a model whether
something is wrong, because a model asked to grade its own platform is exactly
the wrong instrument -- and because these signals need to be trustworthy enough
to act on without a human re-deriving them.

Findings accumulate in a cross-project store, since the most valuable signal is
precisely the one that shows up in *every* project and is therefore invisible
from inside any single one.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..kernel.events import EventType
from ..kernel.ledger import Ledger
from ..util.clock import Clock, default_clock
from ..util.hashing import content_hash

#: A finding needs to clear this to be worth an operator's attention. Set from
#: live observation: the codex schema bug produced 42 identical failures, and
#: the types gate misfired on every node it touched. Genuine difficulty does not
#: repeat this cleanly.
IDENTICAL_FAILURE_THRESHOLD = 3
NEVER_SUCCEEDED_MIN_SAMPLES = 3
RETRY_STORM_WINDOW = 600.0
RETRY_STORM_COUNT = 6

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass(slots=True)
class Finding:
    """One suspected defect in the platform."""

    kind: str
    severity: str
    title: str
    detail: str
    #: What to look at in the Forge source. Not a fix -- a starting point.
    where: str = ""
    occurrences: int = 1
    evidence: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def signature(self) -> str:
        """Stable identity, so the same defect merges across runs and projects."""
        return content_hash(self.kind, self.title)[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "kind": self.kind,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "where": self.where,
            "occurrences": self.occurrences,
            "evidence": self.evidence[:5],
            "projects": self.projects,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        return cls(
            kind=data["kind"],
            severity=data.get("severity", "medium"),
            title=data["title"],
            detail=data.get("detail", ""),
            where=data.get("where", ""),
            occurrences=int(data.get("occurrences", 1)),
            evidence=list(data.get("evidence", [])),
            projects=list(data.get("projects", [])),
            first_seen=float(data.get("first_seen", 0.0)),
            last_seen=float(data.get("last_seen", 0.0)),
        )


# --------------------------------------------------------------------------
# Error normalisation
# --------------------------------------------------------------------------

#: Identifiers, ids, paths, numbers and timestamps differ between two instances
#: of the same bug. Stripping them is what lets identical defects cluster and
#: genuinely different failures stay apart.
_VOLATILE = re.compile(
    r"""
      \b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b  # uuid
    | \b(?:node|run|lease|mem|proj|thread)_[0-9A-Za-z]{6,}\b            # forge ids
    | \b\d{4}-\d{2}-\d{2}[T ][\d:.]+Z?\b                                # timestamps
    | \b0x[0-9a-f]+\b
    | \b\d+\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def error_signature(text: str, *, keep: int = 220) -> str:
    """Reduce an error to a form that is equal for two instances of one bug."""
    cleaned = _VOLATILE.sub("#", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:keep]


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------


def _project_name(ledger: Ledger) -> str:
    rows = ledger.read(types=[EventType.PROJECT_CREATED])
    return str(rows[0].payload.get("name", "")) if rows else ""


def detect_identical_failures(ledger: Ledger) -> list[Finding]:
    """The same node failing the same way is a bug, not a hard problem.

    A genuinely difficult task fails differently each time: a different test,
    a different assertion, a different missing piece. Byte-identical repetition
    means the system is doing something that cannot work and cannot notice.
    """
    clusters: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for event in ledger.read(types=[EventType.NODE_FAILED]):
        error = event.payload.get("error") or ""
        if not error:
            continue
        clusters[(event.node_id or "?", error_signature(error))].append(event)

    findings: list[Finding] = []
    for (node_id, signature), events in clusters.items():
        if len(events) < IDENTICAL_FAILURE_THRESHOLD:
            continue
        retryable = sum(
            1 for e in events if (e.payload.get("result") or {}).get("disposition") == "retry"
        )
        severity = "critical" if retryable >= len(events) - 1 else "high"
        findings.append(
            Finding(
                kind="identical_failure_loop",
                severity=severity,
                title=f"Node retried {len(events)}x on an identical error",
                detail=(
                    f"Node {node_id} failed {len(events)} times with the same normalised "
                    f"error. {retryable} of those were classified retryable. An error that "
                    "reproduces byte-for-byte will not resolve itself; it should be terminal "
                    "and reroute or surface, not consume a backoff loop."
                ),
                where="forge/models/cli_provider.py::_classify_failure, forge/kernel/recovery.py",
                occurrences=len(events),
                evidence=[signature],
                first_seen=events[0].ts,
                last_seen=events[-1].ts,
            )
        )
    return findings


def detect_retry_storms(ledger: Ledger) -> list[Finding]:
    """Many failures on one node inside a short window."""
    by_node: dict[str, list[float]] = defaultdict(list)
    for event in ledger.read(types=[EventType.NODE_FAILED]):
        by_node[event.node_id or "?"].append(event.ts)

    findings: list[Finding] = []
    for node_id, stamps in by_node.items():
        stamps.sort()
        for i in range(len(stamps)):
            window = [t for t in stamps[i:] if t - stamps[i] <= RETRY_STORM_WINDOW]
            if len(window) >= RETRY_STORM_COUNT:
                rate = len(window) / max(1.0, window[-1] - window[0]) * 60
                findings.append(
                    Finding(
                        kind="retry_storm",
                        severity="high",
                        title=f"{len(window)} failures on one node in "
                              f"{window[-1] - window[0]:.0f}s",
                        detail=(
                            f"Node {node_id} failed {len(window)} times at roughly "
                            f"{rate:.1f}/minute. Backoff is either not applied or reset by "
                            "each attempt, so a stuck node spins instead of yielding."
                        ),
                        where="forge/kernel/recovery.py (backoff schedule)",
                        occurrences=len(window),
                        first_seen=window[0],
                        last_seen=window[-1],
                    )
                )
                break
    return findings


def detect_dead_rungs(ledger: Ledger) -> list[Finding]:
    """A rung that has never once succeeded for a task class is misconfigured."""
    tally: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for event in ledger.read(types=[EventType.ROUTE_DECIDED]):
        key = (str(event.payload.get("task_class", "?")), str(event.payload.get("tier", "?")))
        index = 0 if event.payload.get("outcome") == "success" else 1
        tally[key][index] += 1

    findings: list[Finding] = []
    for (task_class, tier), (ok, bad) in tally.items():
        total = ok + bad
        if total >= NEVER_SUCCEEDED_MIN_SAMPLES and ok == 0:
            findings.append(
                Finding(
                    kind="dead_rung",
                    severity="critical",
                    title=f"Rung '{tier}' has never succeeded at '{task_class}'",
                    detail=(
                        f"{total} attempts, 0 successes. A rung that cannot serve a task "
                        "class at all is a wiring fault -- an incompatible schema dialect, "
                        "a missing flag, a bad model id -- not a quality problem. The router "
                        "will keep selecting it, because it prices rungs by cost."
                    ),
                    where="forge/models/cli_provider.py, forge/models/policy.py",
                    occurrences=total,
                )
            )
    return findings


def detect_useless_gates(ledger: Ledger) -> list[Finding]:
    """A gate that never passes is misconfigured, not vigilant."""
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # pass, fail, skip
    summaries: dict[str, list[str]] = defaultdict(list)
    for event in ledger.read(types=[EventType.GATE_PASSED, EventType.GATE_FAILED]):
        gate = str(event.payload.get("gate", "?"))
        if event.type == EventType.GATE_PASSED:
            tally[gate][0] += 1
        else:
            tally[gate][1] += 1
            summary = str(event.payload.get("summary", ""))
            if summary and summary not in summaries[gate]:
                summaries[gate].append(summary)

    findings: list[Finding] = []
    for gate, (passes, failures, _) in tally.items():
        total = passes + failures
        if total >= NEVER_SUCCEEDED_MIN_SAMPLES and passes == 0:
            findings.append(
                Finding(
                    kind="gate_never_passes",
                    severity="high",
                    title=f"Gate '{gate}' has never passed in {total} runs",
                    detail=(
                        f"{failures} failures, 0 passes. A gate that never passes is "
                        "almost always detecting its own misconfiguration rather than a "
                        "defect in the code -- and every failure costs a node an attempt "
                        "and an escalation."
                    ),
                    where="forge/validation/gates/",
                    occurrences=total,
                    evidence=summaries[gate][:3],
                )
            )
    return findings


def detect_discarded_local_work(ledger: Ledger) -> list[Finding]:
    """Local success thrown away, then redone on a cloud rung.

    This is the expensive shape: the cheap rung did the job, something rejected
    the result, and the work was repeated somewhere costly. When it correlates
    with a failing gate it usually means the gate is wrong, not the model.
    """
    sequence: list[tuple[str, str, str]] = []
    for event in ledger.read(types=[EventType.ROUTE_DECIDED]):
        sequence.append(
            (
                event.node_id or "?",
                str(event.payload.get("tier", "?")),
                str(event.payload.get("outcome", "?")),
            )
        )

    wasted: dict[str, int] = defaultdict(int)
    seen_local_success: set[str] = set()
    for node_id, tier, outcome in sequence:
        if tier.startswith("local") and outcome == "success":
            seen_local_success.add(node_id)
        elif (
            node_id in seen_local_success
            and not tier.startswith("local")
            # Only a *successful* cloud call actually repeated the work. Failed
            # attempts are their own finding; counting them here would inflate
            # this one with the same incident and make the report untrustworthy.
            and outcome == "success"
        ):
            wasted[node_id] += 1

    if not wasted:
        return []
    return [
        Finding(
            kind="discarded_local_work",
            severity="high",
            title=f"{len(wasted)} node(s) succeeded locally, then were redone on a cloud rung",
            detail=(
                "The local rung produced an accepted result and the node still escalated. "
                "Something downstream -- most often a gate that cannot run in this "
                "environment -- is discarding good work and paying frontier prices to "
                "repeat it. This defeats the local-first design directly."
            ),
            where="forge/validation/gates/, forge/kernel/orchestrator.py",
            occurrences=sum(wasted.values()),
            evidence=[f"node {n}: {c} cloud redo(s)" for n, c in list(wasted.items())[:5]],
        )
    ]


def detect_cloud_fraction_breach(ledger: Ledger, target: float) -> list[Finding]:
    """Sustained cloud usage far above the configured target."""
    local = cloud = 0
    for event in ledger.read(types=[EventType.BUDGET_SPENT]):
        tokens = int(event.payload.get("output_tokens", 0) or 0)
        if event.payload.get("hosted") == "cloud":
            cloud += tokens
        else:
            local += tokens
    total = local + cloud
    if total < 1000 or target <= 0:
        return []
    fraction = cloud / total
    if fraction < min(0.95, target * 3):
        return []
    return [
        Finding(
            kind="cloud_fraction_breach",
            severity="high",
            title=f"Cloud tokens at {fraction:.0%} against a {target:.0%} target",
            detail=(
                f"{cloud:,} cloud output tokens versus {local:,} local. A breach this "
                "wide is a routing or validation fault rather than a run of hard tasks; "
                "check for a dead rung or a gate rejecting local output."
            ),
            where="forge/models/policy.py, forge/validation/",
            occurrences=1,
        )
    ]


def collect(ledger: Ledger, *, cloud_target: float = 0.0) -> list[Finding]:
    """Run every detector over one project's ledger."""
    findings: list[Finding] = []
    findings += detect_identical_failures(ledger)
    findings += detect_retry_storms(ledger)
    findings += detect_dead_rungs(ledger)
    findings += detect_useless_gates(ledger)
    findings += detect_discarded_local_work(ledger)
    findings += detect_cloud_fraction_breach(ledger, cloud_target)

    project = _project_name(ledger)
    for finding in findings:
        if project and project not in finding.projects:
            finding.projects.append(project)
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), -f.occurrences))
    return findings


# --------------------------------------------------------------------------
# Cross-project store
# --------------------------------------------------------------------------


class FeedbackStore:
    """Accumulates findings across projects, one JSON file per signature.

    Cross-project on purpose. A defect that appears in every project is both the
    most valuable to fix and the hardest to notice from inside any one of them,
    where it just looks like how the tool behaves.
    """

    def __init__(self, root: Path, *, clock: Clock | None = None) -> None:
        self.root = Path(root).expanduser()
        self._clock = clock or default_clock()

    def _path(self, signature: str) -> Path:
        return self.root / f"finding_{signature}.json"

    def all(self) -> list[Finding]:
        if not self.root.exists():
            return []
        out: list[Finding] = []
        for path in sorted(self.root.glob("finding_*.json")):
            try:
                out.append(Finding.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        out.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), -f.occurrences))
        return out

    def record(self, findings: list[Finding]) -> tuple[int, int]:
        """Merge findings into the store. Returns (new, updated)."""
        self.root.mkdir(parents=True, exist_ok=True)
        new = updated = 0
        for finding in findings:
            path = self._path(finding.signature)
            now = self._clock.now()
            if path.exists():
                try:
                    existing = Finding.from_dict(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError, KeyError):
                    existing = None
                if existing is not None:
                    existing.occurrences += finding.occurrences
                    existing.last_seen = finding.last_seen or now
                    for project in finding.projects:
                        if project not in existing.projects:
                            existing.projects.append(project)
                    for item in finding.evidence:
                        if item not in existing.evidence:
                            existing.evidence.append(item)
                    # A defect seen in several projects is a platform defect.
                    if len(existing.projects) > 1 and existing.severity == "high":
                        existing.severity = "critical"
                    finding = existing
                    updated += 1
                else:
                    new += 1
            else:
                finding.first_seen = finding.first_seen or now
                finding.last_seen = finding.last_seen or now
                new += 1
            path.write_text(json.dumps(finding.to_dict(), indent=2), encoding="utf-8")
        return new, updated

    def clear(self) -> int:
        if not self.root.exists():
            return 0
        paths = list(self.root.glob("finding_*.json"))
        for path in paths:
            path.unlink()
        return len(paths)


def render(findings: list[Finding], *, verbose: bool = False) -> str:
    """A report an operator can read, or paste to whoever maintains Forge."""
    if not findings:
        return "No platform anomalies detected."

    marks = {"critical": "!!", "high": " !", "medium": " ~", "low": " ."}
    lines = [f"{len(findings)} platform anomal{'y' if len(findings) == 1 else 'ies'} detected.", ""]
    for finding in findings:
        lines.append(f"{marks.get(finding.severity, '  ')} [{finding.severity}] {finding.title}")
        lines.append(f"     {finding.detail}")
        if finding.where:
            lines.append(f"     look at: {finding.where}")
        if finding.projects:
            lines.append(f"     seen in: {', '.join(finding.projects)}")
        if verbose and finding.evidence:
            for item in finding.evidence[:5]:
                lines.append(f"       - {item[:160]}")
        lines.append("")
    lines.append("These are defects in Forge, not in the project. Fixing them requires a")
    lines.append("code change; no amount of model self-improvement will reach them.")
    return "\n".join(lines)
