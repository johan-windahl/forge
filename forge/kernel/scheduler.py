"""Scheduling policy: what runs next, what happens when it fails.

Separated from the orchestrator (which *executes*) because these are the
decisions worth reasoning about and testing in isolation: which node to pick,
how long to wait before retrying, when to escalate, and when to stop trying.

The failure policy is the part that determines whether a multi-day run degrades
gracefully or thrashes. Its shape:

* **Transient failures are free.** A network blip or a rate limit does not
  consume an attempt. Otherwise an hour of provider instability would exhaust
  every node's retries and permanently fail a project that had nothing wrong
  with it.
* **Retries escalate rather than repeat.** The second attempt at a task uses a
  stronger model than the first. Repeating an identical call is the least
  informative thing a system can do with its budget.
* **Backoff is exponential with jitter.** Jitter matters with several workers:
  synchronised retries against one local model server produce queueing that
  looks like the model being slow.
* **Give up loudly, keep working.** A node that exhausts its attempts is parked
  as blocked with a specific question. The rest of the graph carries on -- the
  system does not stop because one leaf is stuck.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..config import SchedulerConfig
from ..errors import ForgeError
from ..obs.log import get_logger
from ..util.clock import Clock, default_clock
from .graph import Node, NodeStatus, TaskGraph

log = get_logger("kernel.scheduler")


class Disposition(StrEnum):
    """What to do with a node after an attempt failed."""

    RETRY = "retry"  # same tier, after backoff
    ESCALATE = "escalate"  # stronger model, after backoff
    BLOCK = "block"  # out of options; park for a human
    FAIL = "fail"  # terminal; cascade to dependents


@dataclass(slots=True)
class FailurePlan:
    disposition: Disposition
    delay: float = 0.0
    reason: str = ""
    escalate_to: str | None = None
    #: Give the attempt back. `graph.start` charges one on every claim, so
    #: without this a failure the node did not cause still spends its budget.
    refund_attempt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": str(self.disposition),
            "delay": round(self.delay, 2),
            "reason": self.reason,
            "escalate_to": self.escalate_to,
        }


class Scheduler:
    """Node selection and failure handling."""

    def __init__(
        self,
        graph: TaskGraph,
        config: SchedulerConfig,
        *,
        clock: Clock | None = None,
        ladder: list[str] | None = None,
        seed: int | None = None,
        per_node_cost: float = 0.0,
    ) -> None:
        self.graph = graph
        self.config = config
        self._clock = clock or default_clock()
        self.ladder = ladder or ["local", "local_deep", "frontier"]
        self._rng = random.Random(seed)
        #: Needed only to explain a block. A node over this ceiling cannot reach
        #: any cloud rung, so "try a stronger model" is not advice an operator can
        #: act on and the question has to say why.
        self.per_node_cost = per_node_cost
        #: Consecutive transient failures per node, in memory only. It cannot
        #: live on the node: attempts are refunded precisely so an outage leaves
        #: no trace there, and a run that restarts mid-outage should give the
        #: environment a fresh chance rather than inherit a grudge.
        self._transient: dict[str, int] = {}

    # -- selection -------------------------------------------------------

    def next_node(self, *, exclude: set[str] | None = None) -> Node | None:
        """The best node to start now, or ``None`` if nothing is runnable."""
        exclude = exclude or set()
        for node in self.graph.runnable(limit=32):
            if node.id in exclude:
                continue
            return node
        return None

    def reap(self) -> list[str]:
        """Reclaim nodes whose worker died. Safe to call on every poll."""
        return self.graph.reap_expired_leases()

    def sweep(self) -> dict[str, int]:
        """Periodic maintenance: promote pending nodes, reclaim dead leases.

        Both operations are idempotent and cheap. They exist as a safety net for
        the window between a crash and a restart, where an event was written but
        its follow-up action was not.
        """
        reaped = self.reap()
        promoted = self.graph.promote_ready()
        return {"reaped": len(reaped), "promoted": len(promoted)}

    # -- failure policy --------------------------------------------------

    def plan_failure(self, node: Node, error: Exception | None, *, escalatable: bool = False) -> FailurePlan:
        """Decide what happens after a failed attempt."""
        transient = isinstance(error, ForgeError) and error.transient
        retryable = not isinstance(error, ForgeError) or error.retryable
        wants_escalation = escalatable or (isinstance(error, ForgeError) and error.escalatable)

        # Transient faults are the environment's problem, not the node's, so they
        # do not consume an attempt. That exemption used to be unbounded in both
        # directions: the backoff was capped at `min(attempts, 3)` (about twenty
        # seconds) and the check ran before `max_attempts`, so a subscription rate
        # limit that lasted ninety minutes produced ~270 identical retries, each
        # logging a failure, and never once told the operator that the run had
        # stopped making progress for a reason no retry could fix.
        #
        # Now: back off properly toward `backoff_max` as the outage persists, and
        # after `max_transient_attempts` park the node so it is visible. Still
        # free of the escalation ladder -- a stronger model cannot fix a quota.
        #
        # And count them separately. "Does not consume an attempt" was only
        # ever true of this function: `graph.start` charges one on every claim,
        # so an outage still drained the budget checked below. Ten minutes of
        # provider 529s left a node at `attempts == max_attempts`, and the first
        # genuine failure after the provider recovered blocked it instantly --
        # never retried, never escalated. Exactly what the exemption exists to
        # prevent, arrived at from the other direction.
        if transient:
            name = type(error).__name__ if error else "unknown"
            streak = self._transient.get(node.id, 0) + 1
            self._transient[node.id] = streak
            if (
                self.config.max_transient_attempts > 0
                and streak >= self.config.max_transient_attempts
            ):
                return FailurePlan(
                    disposition=Disposition.BLOCK,
                    reason=(
                        f"{name} has not cleared after {streak} consecutive attempts. "
                        "This is an environment or quota problem, not a code problem: "
                        "no retry and no stronger model will resolve it."
                    ),
                )
            return FailurePlan(
                disposition=Disposition.RETRY,
                delay=self.backoff(streak),
                reason=f"transient failure: {name}",
                refund_attempt=True,
            )

        # The node itself failed, so the outage (if any) is over.
        self._transient.pop(node.id, None)

        if not retryable:
            return FailurePlan(
                disposition=Disposition.BLOCK,
                reason=f"unrecoverable: {error}" if error else "unrecoverable failure",
            )

        if node.attempts >= self.config.max_attempts:
            # Name the budget, not just the count. "exhausted 14 attempts" against
            # a `max_attempts` of 4 reads as a contradiction, and the explanation
            # -- that the counter is cumulative across operator unblocks -- was
            # nowhere in the message.
            reason = f"exhausted the {self.config.max_attempts}-attempt budget"
            if node.attempts > self.config.max_attempts:
                reason += f" ({node.attempts} attempts on this node in total)"
            return FailurePlan(disposition=Disposition.BLOCK, reason=reason)

        same_failure = int(node.spec.get("_same_failure_count", 0))
        strategies = set(node.spec.get("_strategies_tried", []))
        at_top = node.tier == self.ladder[-1] if self.ladder else True
        decomposed_or_not_code = bool(node.spec.get("decomposed")) or node.kind not in {
            "implement", "debug", "refactor", "scaffold", "test_author", "document"
        }
        if (
            same_failure >= self.config.max_no_progress_attempts
            and at_top
            and decomposed_or_not_code
            and "coach" in strategies
        ):
            return FailurePlan(
                disposition=Disposition.BLOCK,
                reason=(
                    "practically unsolvable: the same deterministic failure remained "
                    f"after {same_failure} attempts and local, decomposition, coaching, "
                    "and direct strong-model strategies were exhausted"
                ),
            )

        # An exception that is not a ForgeError escaped Forge's own code: an
        # internal bug, not a model that could not do the work. A stronger model
        # cannot fix our bug, and the pinball run measured the cost of pretending
        # otherwise -- attempts consumed by four platform defects pinned one node
        # to the most expensive rung for the rest of the project, against a
        # cloud-spend target of 18%. Retry on the same rung; `max_attempts` above
        # still stops a permanently broken platform from looping.
        if error is not None and not isinstance(error, ForgeError):
            return FailurePlan(
                disposition=Disposition.RETRY,
                delay=self.backoff(node.attempts),
                reason=f"platform fault, not a model failure: {type(error).__name__}",
            )

        next_tier = self._next_tier(node.tier)
        if (wants_escalation or node.attempts >= self.config.escalate_after_attempts) and next_tier:
            return FailurePlan(
                disposition=Disposition.ESCALATE,
                delay=self.backoff(node.attempts),
                reason=f"attempt {node.attempts} failed; trying a stronger model",
                escalate_to=next_tier,
            )

        return FailurePlan(
            disposition=Disposition.RETRY,
            delay=self.backoff(node.attempts),
            reason=f"attempt {node.attempts} failed; retrying",
        )

    def apply(self, node: Node, plan: FailurePlan, error_message: str) -> None:
        """Write the failure decision into the graph."""
        match plan.disposition:
            case Disposition.RETRY:
                self.graph.fail(
                    node.id,
                    error_message,
                    terminal=False,
                    retry_at=self._clock.now() + plan.delay,
                    detail={"disposition": "retry", "reason": plan.reason},
                )
                if plan.refund_attempt:
                    # The claim that starts the next attempt will charge it
                    # again, so this nets to zero across an outage.
                    self.graph.update(
                        node.id, attempts=max(0, node.attempts - 1), actor="scheduler"
                    )
            case Disposition.ESCALATE:
                if plan.escalate_to:
                    self.graph.escalate(node.id, plan.escalate_to, plan.reason)
                self.graph.fail(
                    node.id,
                    error_message,
                    terminal=False,
                    retry_at=self._clock.now() + plan.delay,
                    detail={"disposition": "escalate", "tier": plan.escalate_to, "reason": plan.reason},
                )
            case Disposition.BLOCK:
                self.graph.block(
                    node.id,
                    plan.reason,
                    question=_question_for(node, error_message, per_node_cost=self.per_node_cost),
                )
            case Disposition.FAIL:
                self.graph.fail(node.id, error_message, terminal=True, detail={"reason": plan.reason})

    def backoff(self, attempt: int) -> float:
        """Exponential backoff with proportional jitter."""
        base = min(self.config.backoff_base * (2 ** max(0, attempt - 1)), self.config.backoff_max)
        jitter = base * self.config.backoff_jitter
        return max(0.0, base + self._rng.uniform(-jitter, jitter))

    def _next_tier(self, current: str) -> str | None:
        try:
            index = self.ladder.index(current)
        except ValueError:
            return self.ladder[-1] if self.ladder else None
        return self.ladder[index + 1] if index + 1 < len(self.ladder) else None

    # -- run-level state -------------------------------------------------

    def status(self) -> dict[str, Any]:
        counts = self.graph.counts()
        return {
            "counts": counts,
            "progress": round(self.graph.progress(), 4),
            "quiescent": self.graph.is_quiescent(),
            "runnable": len(self.graph.runnable(limit=64)),
        }

    def stalled(self) -> bool:
        """True when nothing can progress but work remains unfinished.

        Distinguished from "finished": a stalled run has blocked nodes that a
        human could unblock, and the operator should be told the difference.
        """
        counts = self.graph.counts()
        if not self.graph.is_quiescent():
            return False
        return counts[NodeStatus.BLOCKED] > 0 or counts[NodeStatus.FAILED] > 0


def _question_for(node: Node, error_message: str, *, per_node_cost: float = 0.0) -> str:
    """Compose the question a human would need to answer to unblock this.

    Being specific here is the difference between an operator who can help in
    thirty seconds and one who has to read a day of logs. The node's own
    acceptance criteria are included because they are usually where the
    ambiguity lives.
    """
    parts = [
        f"Node '{node.title}' ({node.kind}) could not be completed after {node.attempts} attempt(s).",
        f"Last error: {error_message[:400]}",
    ]
    if node.acceptance:
        parts.append("Its acceptance criteria were:")
        parts += [f"  - {item}" for item in node.acceptance]
    # The most useful sentence in the whole question, when it applies. A node
    # over its per-node ceiling cannot reach any cloud rung: every "escalation"
    # after that point re-runs the same local model, and the route reason still
    # says "climbing one rung". One node spent 16 attempts that way on an error a
    # frontier model would have fixed immediately, and nothing said the stronger
    # rungs had been priced out.
    if per_node_cost > 0 and node.cost > per_node_cost:
        parts.append(
            f"NOTE: this node has spent {node.cost:.2f} against a per-node ceiling of "
            f"{per_node_cost:.2f}, so every cloud rung was unaffordable and each retry "
            f"used the strongest *local* model rather than a stronger one. Raising "
            f"`budget.per_node_cost` is what would let a frontier model attempt this."
        )
    parts.append(
        "Resolve by either clarifying the requirement, fixing the environment, or "
        "cancelling this node with `forge cancel`."
    )
    return "\n".join(parts)
