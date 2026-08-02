"""Promoting model reasoning into deterministic tooling.

This is the mechanism behind the brief's question "could deterministic tooling
replace model reasoning?", and it is answered by measurement rather than by
asking a model to speculate.

The observation: when a reviewer flags the same class of problem repeatedly, or
a debug node fixes the same category of failure over and over, that recurrence
is evidence that a rule exists. A rule can be a lint configuration, a test, a
new gate, or a line in the conventions -- all of which cost nothing per run and
never forget, unlike a model that must rediscover the problem each time.

Detection is deterministic: cluster recorded findings and fixes by normalised
text, and surface any cluster above a threshold. What to *do* about a cluster is
proposed by a model, because that requires knowing what tooling exists. But the
decision that a cluster is worth acting on is arithmetic.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..kernel.events import EventType
from ..kernel.ledger import Ledger
from ..memory.records import MemoryKind
from ..memory.store import MemoryStore


@dataclass(slots=True)
class PromotionCandidate:
    """A repeated pattern that deterministic tooling could handle instead."""

    signature: str
    occurrences: int
    examples: list[str] = field(default_factory=list)
    #: Where the pattern was observed: review, gate, debug.
    origin: str = "review"
    paths: list[str] = field(default_factory=list)
    #: Filled in by the improvement agent.
    proposal: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "occurrences": self.occurrences,
            "origin": self.origin,
            "examples": self.examples[:5],
            "paths": self.paths[:10],
            "proposal": self.proposal,
        }


#: Tokens stripped when computing a signature, so that "the render loop leaks a
#: listener in Player.ts" and "the update loop leaks a listener in Enemy.ts"
#: cluster together instead of looking like two unrelated findings.
_NOISE = re.compile(r"""["'`][^"'`]{1,80}["'`]|\b[A-Za-z_][A-Za-z0-9_]*\.(?:ts|tsx|js|jsx|py|rs|go)\b|\b\d+\b""")
_WORD = re.compile(r"[a-z]{3,}")

_STOP = frozenset(
    ["the", "this", "that", "with", "from", "into", "your", "there", "here", "should", "would", "could", "must", "have", "has", "been", "are", "was", "were", "will", "and", "but", "for", "not", "are", "its", "his", "her", "they", "them", "then", "than", "when", "where", "which", "what", "who", "how", "why", "all", "any", "some", "more", "most", "other", "such", "only", "own", "same", "very", "can", "just"]
)


def signature(text: str, *, keep: int = 6) -> str:
    """Reduce a finding to a comparable fingerprint.

    Keeps the most informative words -- the rarest ones are also the most
    specific to a single incident, so the *first* words of the normalised text
    are used instead, which in practice carry the category ("unhandled promise
    rejection") rather than the instance.
    """
    cleaned = _NOISE.sub(" ", text.lower())
    words = [w for w in _WORD.findall(cleaned) if w not in _STOP]
    return " ".join(words[:keep])


def detect_promotions(
    ledger: Ledger,
    memory: MemoryStore,
    *,
    threshold: int = 3,
    limit: int = 10,
) -> list[PromotionCandidate]:
    """Find repeated patterns worth turning into tooling."""
    clusters: dict[tuple[str, str], PromotionCandidate] = {}

    def add(origin: str, text: str, path: str | None = None) -> None:
        sig = signature(text)
        if len(sig.split()) < 3:
            return
        key = (origin, sig)
        candidate = clusters.get(key)
        if candidate is None:
            candidate = PromotionCandidate(signature=sig, occurrences=0, origin=origin)
            clusters[key] = candidate
        candidate.occurrences += 1
        if len(candidate.examples) < 8:
            candidate.examples.append(text[:200])
        if path and path not in candidate.paths:
            candidate.paths.append(path)

    # Review and gate findings recorded in memory.
    for record in memory.by_kind(MemoryKind.FINDING, limit=1000):
        add("review" if record.source.startswith(("review", "security", "visual")) else "gate",
            record.title, record.paths[0] if record.paths else None)
    # Findings already resolved still count: a problem fixed five times is
    # exactly the one worth preventing.
    for record in memory.by_kind(MemoryKind.FINDING, status="resolved", limit=1000):
        add("review", record.title, record.paths[0] if record.paths else None)

    # Recurring gate failures, from the log.
    gate_failures: dict[str, int] = defaultdict(int)
    for event in ledger.read(types=[EventType.GATE_FAILED]):
        summary = event.payload.get("summary", "")
        gate = event.payload.get("gate", "?")
        gate_failures[gate] += 1
        if summary:
            add("gate", f"{gate}: {summary}")

    # Bug fixes recorded by the debug agent.
    for record in memory.by_kind(MemoryKind.FACT, limit=1000):
        if "bugfix" in record.tags:
            add("debug", record.title)

    candidates = [c for c in clusters.values() if c.occurrences >= threshold]
    candidates.sort(key=lambda c: -c.occurrences)
    return candidates[:limit]


def routing_promotions(ledger: Ledger, *, min_samples: int = 8) -> list[str]:
    """Suggest ladder changes backed by observed success rates.

    Returned as sentences rather than applied automatically. A routing change
    based on eight samples can be wrong in a way that costs a lot; the
    retrospective proposes and the operator (or a future policy with more
    evidence) disposes.
    """
    suggestions: list[str] = []
    for row in ledger.conn.execute("SELECT * FROM routing_stats_v2 ORDER BY task_class, tier"):
        total = row["successes"] + row["failures"]
        if total < min_samples:
            continue
        rate = row["successes"] / total
        if rate >= 0.92 and row["tier"] != "local":
            suggestions.append(
                f"'{row['task_class']}' succeeds {rate:.0%} of the time on '{row['tier']}' "
                f"over {total} attempts; a cheaper rung is likely sufficient."
            )
        elif rate <= 0.45:
            suggestions.append(
                f"'{row['task_class']}' succeeds only {rate:.0%} of the time on '{row['tier']}' "
                f"over {total} attempts; start this class one rung higher to avoid wasted attempts."
            )
    return suggestions


def gate_promotions(ledger: Ledger) -> list[str]:
    """Suggest gate configuration changes from measured behaviour."""
    suggestions: list[str] = []
    rows = ledger.conn.execute(
        """SELECT gate,
                  SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) AS passes,
                  SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) AS failures,
                  AVG(duration) AS mean_duration,
                  COUNT(*) AS runs
           FROM gate_results GROUP BY gate"""
    ).fetchall()
    for row in rows:
        runs = int(row["runs"])
        if runs < 5:
            continue
        if int(row["failures"]) == 0 and float(row["mean_duration"] or 0) > 120:
            suggestions.append(
                f"Gate '{row['gate']}' has never failed in {runs} runs but averages "
                f"{row['mean_duration']:.0f}s; consider running it once per milestone "
                f"rather than per node."
            )
        if int(row["passes"]) == 0 and runs >= 5:
            suggestions.append(
                f"Gate '{row['gate']}' has never passed in {runs} runs; it is probably "
                f"misconfigured for this project rather than detecting a real problem."
            )
    return suggestions
