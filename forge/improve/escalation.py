"""Escalation pairs: what the weak model got wrong that the strong model got right.

An escalation is the most informative event in the system and Forge used to
discard it. When a node fails on the local rung and then succeeds on a frontier
one, the two attempts share everything -- same task, same context, same
instructions -- and differ only in the thing worth learning about. That is a
matched pair, and it is far better evidence than asking a model to speculate in
the abstract about what small models get wrong.

This module captures those pairs. It deliberately does *not* draw conclusions
from them. Extraction -- turning a pair into a rule, and preferably into a
deterministic check rather than a prompt -- is a separate step that should run
against a corpus large enough to distinguish a general failure mode from one bad
afternoon. Writing the corpus is cheap and the data is perishable; interpreting
it is neither.

Two properties the corpus needs, and the reason each is here:

**Cross-project.** A rule derived from one project is a guess. The same mistake
in three unrelated projects is a property of the model. Pairs therefore
accumulate outside any project, alongside lessons and feedback.

**Evidence, not just outputs.** The pair on its own says one attempt was
rejected; it does not say why. The gate verdict or validation error that did the
rejecting is what makes the difference diagnosable, so it is captured with it.

A caution worth recording where it will be read. During the run this was built
from, almost every "local failure" was a Forge defect rather than a model error:
a gate that could not run reported a missing compiler as a type error, and a
schema dialect mismatch failed a rung 67 times. Mining such pairs would teach a
model to work around bugs that no longer exist. Pairs carry the rejecting
evidence precisely so that extraction can filter these out, and any extraction
pass must do so.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..kernel.events import EventType
from ..kernel.ledger import Ledger
from ..obs.log import get_logger
from ..util.hashing import content_hash

log = get_logger("improve.escalation")

#: Ladder position by rung name, so "stronger" is a comparison rather than a
#: guess. Resolved from config where available; this is the fallback ordering.
DEFAULT_LADDER = ("local", "local_deep", "codex", "claude")

#: A corpus is only useful if it stays readable. Oldest pairs are dropped first.
MAX_PAIRS = 2000


@dataclass(slots=True)
class Attempt:
    """One model's answer to one node."""

    model: str
    tier: str
    task_class: str
    text: str = ""
    truncated: bool = False
    error: str = ""
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tier": self.tier,
            "task_class": self.task_class,
            "text": self.text,
            "truncated": self.truncated,
            "error": self.error,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attempt:
        return cls(
            model=data.get("model", ""),
            tier=data.get("tier", ""),
            task_class=data.get("task_class", ""),
            text=data.get("text", ""),
            truncated=bool(data.get("truncated", False)),
            error=data.get("error", ""),
            ts=float(data.get("ts", 0.0)),
        )


@dataclass(slots=True)
class Pair:
    """A weaker model's rejected answer beside a stronger model's accepted one."""

    node_id: str
    node_title: str
    project: str
    task_class: str
    weak: Attempt
    strong: Attempt
    #: What rejected the weak attempt: gate summaries, validation errors.
    #: Without this a pair shows a difference but not a fault.
    evidence: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    @property
    def signature(self) -> str:
        """Identity of the *situation*, so re-runs do not duplicate the corpus."""
        return content_hash(self.project, self.node_id, self.weak.model, self.strong.model)[:16]

    @property
    def diagnosable(self) -> bool:
        """Is there enough here to learn from?

        A pair with no weak output, or no rejecting evidence, records only that
        an escalation happened -- which the routing policy already knows.
        """
        return bool(self.weak.text and (self.evidence or self.weak.error))

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "node_id": self.node_id,
            "node_title": self.node_title,
            "project": self.project,
            "task_class": self.task_class,
            "weak": self.weak.to_dict(),
            "strong": self.strong.to_dict(),
            "evidence": self.evidence[:6],
            "paths": self.paths[:12],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pair:
        return cls(
            node_id=data.get("node_id", ""),
            node_title=data.get("node_title", ""),
            project=data.get("project", ""),
            task_class=data.get("task_class", ""),
            weak=Attempt.from_dict(data.get("weak", {})),
            strong=Attempt.from_dict(data.get("strong", {})),
            evidence=list(data.get("evidence", [])),
            paths=list(data.get("paths", [])),
        )


def _rank(ladder: tuple[str, ...] | list[str], model: str) -> int:
    try:
        return list(ladder).index(model)
    except ValueError:
        return len(ladder)  # unknown rungs sort as strongest, never as weakest


def find_pairs(
    ledger: Ledger,
    *,
    ladder: tuple[str, ...] | list[str] = DEFAULT_LADDER,
    project: str = "",
    titles: dict[str, str] | None = None,
) -> list[Pair]:
    """Recover every weak-failed/strong-succeeded pair from one ledger.

    Pure reading. The ledger already holds attempts, outcomes and gate verdicts;
    this correlates them by node rather than storing anything new during a run.
    """
    responses: dict[str, list[Attempt]] = defaultdict(list)
    for event in ledger.read(types=[EventType.MODEL_RESPONSE]):
        if not event.node_id:
            continue
        payload = event.payload
        responses[event.node_id].append(
            Attempt(
                model=str(payload.get("model", "")),
                tier=str(payload.get("tier", "")),
                task_class=str(payload.get("task_class", "")),
                text=str(payload.get("text", "")),
                truncated=bool(payload.get("text_truncated", False)),
                ts=event.ts,
            )
        )

    outcomes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for event in ledger.read(types=[EventType.ROUTE_DECIDED]):
        if event.node_id:
            outcomes[event.node_id].append(
                (str(event.payload.get("tier", "")), str(event.payload.get("outcome", "")))
            )

    errors: dict[str, list[str]] = defaultdict(list)
    for event in ledger.read(types=[EventType.NODE_FAILED]):
        if event.node_id and event.payload.get("error"):
            errors[event.node_id].append(str(event.payload["error"]))

    evidence: dict[str, list[str]] = defaultdict(list)
    for event in ledger.read(types=[EventType.GATE_FAILED]):
        if event.node_id:
            summary = str(event.payload.get("summary", ""))
            detail = str(event.payload.get("evidence", ""))[:400]
            line = f"{event.payload.get('gate', '?')}: {summary}"
            if detail:
                line += f"\n{detail}"
            if line not in evidence[event.node_id]:
                evidence[event.node_id].append(line)

    pairs: list[Pair] = []
    for node_id, results in outcomes.items():
        failed = {tier for tier, outcome in results if outcome != "success"}
        succeeded = [tier for tier, outcome in results if outcome == "success"]
        if not failed or not succeeded:
            continue
        weak_name = min(failed, key=lambda t: _rank(ladder, t))
        strong_name = max(succeeded, key=lambda t: _rank(ladder, t))
        if _rank(ladder, strong_name) <= _rank(ladder, weak_name):
            continue  # not an escalation: it eventually worked on the same rung

        attempts = responses.get(node_id, [])
        weak = next((a for a in attempts if a.model == weak_name), Attempt(weak_name, "", ""))
        strong = next((a for a in attempts if a.model == strong_name), Attempt(strong_name, "", ""))
        weak.error = errors[node_id][0] if errors[node_id] else ""

        pairs.append(
            Pair(
                node_id=node_id,
                node_title=(titles or {}).get(node_id, ""),
                project=project,
                task_class=weak.task_class or strong.task_class,
                weak=weak,
                strong=strong,
                evidence=evidence[node_id],
            )
        )
    pairs.sort(key=lambda p: p.weak.ts)
    return pairs


class EscalationCorpus:
    """Pairs accumulated across every project on the host.

    One file per pair, like lessons and feedback, so the corpus can be read,
    diffed and pruned by a human without a tool. Cross-project because a mistake
    seen once is an anecdote and the same mistake in three unrelated projects is
    a property of the model.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()

    def _path(self, signature: str) -> Path:
        return self.root / f"pair_{signature}.json"

    def all(self) -> list[Pair]:
        if not self.root.exists():
            return []
        out: list[Pair] = []
        for path in sorted(self.root.glob("pair_*.json")):
            try:
                out.append(Pair.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        return out

    def record(self, pairs: list[Pair], *, only_diagnosable: bool = True) -> int:
        """Add pairs to the corpus. Returns how many were newly written."""
        written = 0
        for pair in pairs:
            if only_diagnosable and not pair.diagnosable:
                continue
            path = self._path(pair.signature)
            if path.exists():
                continue
            self.root.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pair.to_dict(), indent=2), encoding="utf-8")
            written += 1
        if written:
            log.info("escalation pairs recorded", count=written, root=str(self.root))
        self._prune()
        return written

    def _prune(self) -> None:
        paths = sorted(self.root.glob("pair_*.json")) if self.root.exists() else []
        for path in paths[: max(0, len(paths) - MAX_PAIRS)]:
            path.unlink(missing_ok=True)

    def stats(self) -> dict[str, Any]:
        pairs = self.all()
        by_class: dict[str, int] = defaultdict(int)
        by_project: dict[str, int] = defaultdict(int)
        for pair in pairs:
            by_class[pair.task_class] += 1
            by_project[pair.project] += 1
        return {
            "pairs": len(pairs),
            "projects": len(by_project),
            "by_task_class": dict(by_class),
            "diagnosable": sum(1 for p in pairs if p.diagnosable),
        }


def render(pairs: list[Pair], *, verbose: bool = False) -> str:
    if not pairs:
        return "No escalation pairs recorded yet."
    lines = [f"{len(pairs)} escalation pair(s).", ""]
    for pair in pairs:
        mark = "  " if pair.diagnosable else "~ "
        lines.append(f"{mark}[{pair.task_class}] {pair.weak.model} failed -> {pair.strong.model} succeeded")
        if pair.node_title:
            lines.append(f"     {pair.node_title[:78]}")
        if pair.project:
            lines.append(f"     project: {pair.project}")
        if pair.evidence:
            lines.append(f"     rejected by: {pair.evidence[0].splitlines()[0][:70]}")
        elif not pair.diagnosable:
            lines.append("     (no rejecting evidence retained; not diagnosable)")
        if verbose and pair.weak.text:
            lines.append(f"     weak output: {pair.weak.text[:200]!r}")
        lines.append("")
    return "\n".join(lines)
