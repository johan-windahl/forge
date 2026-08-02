"""The event vocabulary.

Forge is event-sourced: the ledger of events is the only authoritative state,
and everything else -- the node table, the budget, the dashboard, the
retrospective -- is a projection that can be discarded and rebuilt. This buys
three things that matter for unattended operation:

* **Crash recovery is free.** There is no "was the write half-applied?" question;
  either the event is in the log or it is not.
* **Post-hoc analysis is exact.** The self-improvement loop reads the same log
  the scheduler wrote, so its conclusions describe what actually happened rather
  than what a summary said happened.
* **Time travel.** ``forge rollback`` replays to any prior sequence number.

Event types are strings, not an enum's numeric values, so a log written by one
version of Forge stays readable by the next. Unknown types are preserved and
skipped by projections rather than rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..util.hashing import canonical_json
from ..util.ids import new_id


class EventType:
    """Canonical event names. Grouped by the subsystem that emits them."""

    # -- project lifecycle
    PROJECT_CREATED = "project.created"
    PROJECT_GOAL_SET = "project.goal_set"
    RUN_STARTED = "run.started"
    RUN_STOPPED = "run.stopped"
    RUN_HEARTBEAT = "run.heartbeat"
    #: An environmental problem the run cannot fix and did not cause: an
    #: unreachable model endpoint, a vanished toolchain. Recorded so `forge
    #: status` can explain a stall instead of only reporting one.
    RUN_WARNING = "run.warning"
    USAGE_REPORT = "usage.report"

    # -- planning
    PLAN_PROPOSED = "plan.proposed"
    PLAN_ACCEPTED = "plan.accepted"
    PLAN_REVISED = "plan.revised"
    ASSUMPTION_RECORDED = "assumption.recorded"
    ASSUMPTION_REVISED = "assumption.revised"
    DECISION_RECORDED = "decision.recorded"

    # -- task graph
    NODE_CREATED = "node.created"
    NODE_UPDATED = "node.updated"
    NODE_READY = "node.ready"
    NODE_LEASED = "node.leased"
    NODE_LEASE_RENEWED = "node.lease_renewed"
    NODE_LEASE_EXPIRED = "node.lease_expired"
    NODE_STARTED = "node.started"
    NODE_SUCCEEDED = "node.succeeded"
    NODE_FAILED = "node.failed"
    NODE_BLOCKED = "node.blocked"
    NODE_CANCELLED = "node.cancelled"
    NODE_ESCALATED = "node.escalated"
    NODE_DEFERRED = "node.deferred"

    # -- model layer
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"
    MODEL_ERROR = "model.error"
    MODEL_CACHE_HIT = "model.cache_hit"
    ROUTE_DECIDED = "route.decided"
    BUDGET_SPENT = "budget.spent"
    BUDGET_WARNING = "budget.warning"
    BUDGET_EXHAUSTED = "budget.exhausted"

    # -- work products
    PATCH_APPLIED = "patch.applied"
    TOOLCHAIN_INSTALLED = "toolchain.installed"
    PATCH_REJECTED = "patch.rejected"
    COMMAND_RUN = "command.run"
    ARTIFACT_STORED = "artifact.stored"
    #: A node branch could not absorb main and was rebuilt on the integrated
    #: head, discarding unmergeable provisional work.
    WORKTREE_REBASED = "worktree.rebased"

    # -- validation
    GATE_STARTED = "gate.started"
    GATE_PASSED = "gate.passed"
    GATE_FAILED = "gate.failed"
    GATE_ERRORED = "gate.errored"
    GATE_SKIPPED = "gate.skipped"
    REVIEW_RECORDED = "review.recorded"
    FINDING_RECORDED = "finding.recorded"

    # -- checkpoints
    CHECKPOINT_CREATED = "checkpoint.created"
    ROLLBACK_PERFORMED = "rollback.performed"

    # -- deployment
    DEPLOY_STARTED = "deploy.started"
    DEPLOY_SUCCEEDED = "deploy.succeeded"
    DEPLOY_FAILED = "deploy.failed"
    DEPLOY_ROLLED_BACK = "deploy.rolled_back"

    # -- memory and self-improvement
    MEMORY_WRITTEN = "memory.written"
    MEMORY_SUPERSEDED = "memory.superseded"
    MILESTONE_REACHED = "milestone.reached"
    RETROSPECTIVE_RECORDED = "retrospective.recorded"
    LESSON_LEARNED = "lesson.learned"
    POLICY_UPDATED = "policy.updated"
    PROMOTION_PROPOSED = "promotion.proposed"

    # -- telemetry
    METRIC = "metric"
    LOG = "log"


#: Event types that mutate the node projection. Used by the rebuild path.
NODE_EVENTS = frozenset(
    v
    for k, v in vars(EventType).items()
    if isinstance(v, str) and v.startswith("node.")
)


@dataclass(slots=True)
class Event:
    """One immutable fact about something that happened.

    ``causation_id`` points at the event that directly caused this one and
    ``correlation_id`` groups everything belonging to one logical operation (a
    node attempt, say). Together they let the retrospective reconstruct causal
    chains without heuristics -- "this 90k-token frontier call happened because
    that browser gate failed" is a graph traversal, not an inference.
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("evt"))
    seq: int = 0  # assigned by the ledger on append
    ts: float = 0.0  # assigned by the ledger on append
    project_id: str = ""
    node_id: str | None = None
    actor: str = "system"
    causation_id: str | None = None
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "id": self.id,
            "ts": self.ts,
            "type": self.type,
            "project_id": self.project_id,
            "node_id": self.node_id,
            "actor": self.actor,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
        }

    @classmethod
    def from_row(cls, row: Any) -> Event:
        import json

        return cls(
            seq=row["seq"],
            id=row["id"],
            ts=row["ts"],
            type=row["type"],
            project_id=row["project_id"] or "",
            node_id=row["node_id"],
            actor=row["actor"],
            causation_id=row["causation_id"],
            correlation_id=row["correlation_id"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
        )

    def fingerprint(self) -> str:
        """Content hash ignoring seq/ts -- used to detect duplicate appends."""
        from ..util.hashing import content_hash

        return content_hash(self.type, self.project_id, self.node_id, self.payload)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        body = canonical_json(self.payload)
        if len(body) > 120:
            body = body[:117] + "..."
        return f"<Event #{self.seq} {self.type} node={self.node_id} {body}>"
