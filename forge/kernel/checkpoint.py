"""Checkpoints and rollback.

A checkpoint is a triple: a git commit, a ledger sequence number, and the
metadata tying them to a node or milestone. Because code lives in git and every
other piece of state lives in the ledger, those two numbers fully describe the
system at a moment in time.

Rollback therefore has an exact meaning: reset the working tree to the commit,
and append a compensating event recording that everything after sequence N is
superseded. The ledger is never truncated -- the failed attempt stays in the
record, which is what lets the retrospective learn from it and what makes
rollback itself auditable.

Checkpoints are cheap (a git tag and a row) so they are taken liberally: before
every node attempt, after every success, and at every milestone. The expensive
thing in an unattended system is not storing state, it is discovering that the
state you needed was never stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import GitError, InvariantError
from ..obs.log import get_logger
from ..util.clock import Clock, default_clock, iso
from ..util.ids import new_id
from ..workspace.git import Repo
from .events import Event, EventType
from .ledger import Ledger

log = get_logger("kernel.checkpoint")

TAG_PREFIX = "forge/ckpt"


@dataclass(slots=True)
class Checkpoint:
    id: str
    label: str
    commit: str
    seq: int
    created_at: float
    kind: str = "node"  # node | milestone | manual | pre_attempt
    node_id: str | None = None
    milestone: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def tag(self) -> str:
        return f"{TAG_PREFIX}/{self.id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "commit": self.commit,
            "seq": self.seq,
            "created_at": self.created_at,
            "created_at_iso": iso(self.created_at),
            "kind": self.kind,
            "node_id": self.node_id,
            "milestone": self.milestone,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(
            id=data["id"],
            label=data.get("label", ""),
            commit=data["commit"],
            seq=int(data.get("seq", 0)),
            created_at=float(data.get("created_at", 0.0)),
            kind=data.get("kind", "node"),
            node_id=data.get("node_id"),
            milestone=data.get("milestone"),
            metadata=data.get("metadata") or {},
        )


class CheckpointManager:
    def __init__(
        self,
        ledger: Ledger,
        repo: Repo,
        clock: Clock | None = None,
        graph: Any = None,
    ) -> None:
        self.ledger = ledger
        self.repo = repo
        self._clock = clock or default_clock()
        #: Optional, because a checkpoint is meaningful without one. When present
        #: a rollback also reopens the nodes whose work it just discarded.
        self.graph = graph

    # -- creating --------------------------------------------------------

    def create(
        self,
        label: str,
        *,
        kind: str = "node",
        node_id: str | None = None,
        milestone: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Record a restorable point.

        The WAL is folded into the database file first, so an operator copying
        ``.forge/ledger.db`` at this instant gets a complete snapshot rather than
        a file whose recent history is in a sidecar.
        """
        if not self.repo.has_commits():
            self.repo.commit("chore: initial checkpoint", allow_empty=True)
        commit = self.repo.head()
        self.ledger.checkpoint_wal()

        checkpoint = Checkpoint(
            id=new_id("ckpt").split("_", 1)[1][:16].lower(),
            label=label,
            commit=commit,
            seq=self.ledger.head_seq(),
            created_at=self._clock.now(),
            kind=kind,
            node_id=node_id,
            milestone=milestone,
            metadata=metadata or {},
        )
        try:
            self.repo.tag(checkpoint.tag, f"{label} @ {iso(checkpoint.created_at)}")
        except GitError as exc:  # pragma: no cover - tagging is best effort
            log.warn("could not tag checkpoint", error=str(exc))

        self.ledger.append(
            Event(
                type=EventType.CHECKPOINT_CREATED,
                node_id=node_id,
                payload=checkpoint.to_dict(),
            )
        )
        log.debug("checkpoint created", id=checkpoint.id, label=label, commit=commit[:8])
        return checkpoint

    # -- listing ---------------------------------------------------------

    def list(self, *, limit: int = 50, kind: str | None = None) -> list[Checkpoint]:
        events = self.ledger.read(types=[EventType.CHECKPOINT_CREATED])
        checkpoints = [Checkpoint.from_dict(event.payload) for event in events]
        if kind:
            checkpoints = [c for c in checkpoints if c.kind == kind]
        return checkpoints[-limit:][::-1]

    def get(self, checkpoint_id: str) -> Checkpoint:
        for checkpoint in self.list(limit=10_000):
            if checkpoint.id == checkpoint_id or checkpoint.id.startswith(checkpoint_id):
                return checkpoint
        raise InvariantError("no such checkpoint", checkpoint_id=checkpoint_id)

    def latest(self, *, kind: str | None = None) -> Checkpoint | None:
        found = self.list(limit=1, kind=kind)
        return found[0] if found else None

    # -- restoring -------------------------------------------------------

    def rollback(self, checkpoint_id: str, *, reason: str = "") -> Checkpoint:
        """Restore the working tree to a checkpoint, and reopen what it undid.

        The event history is deliberately *not* rewound: every attempt stays in
        the ledger, because a rollback that erased the failed attempts would also
        erase the reason the rollback happened, and the system would try the same
        thing again.

        Node *status* is a different matter. A node still marked succeeded after
        its commit has been thrown away is simply wrong, and the lie is load
        bearing: `runnable()` selects on status, and dependencies are resolved by
        promotion at succeed-time rather than re-checked, so the dependents of a
        rolled-back node stay runnable and build against code that no longer
        exists. Observed after rolling back a node that had "succeeded" by
        committing a one-line placeholder.

        So: work discarded, node reopened, dependents demoted back to pending.
        Promotion is idempotent and will re-run them when the work really lands.
        """
        checkpoint = self.get(checkpoint_id)
        head_before = self.repo.head() if self.repo.has_commits() else ""

        self.repo.reset_hard(checkpoint.commit)
        reopened = self._reopen_discarded(checkpoint)

        self.ledger.append(
            Event(
                type=EventType.ROLLBACK_PERFORMED,
                node_id=checkpoint.node_id,
                payload={
                    "checkpoint_id": checkpoint.id,
                    "to_commit": checkpoint.commit,
                    "from_commit": head_before,
                    "to_seq": checkpoint.seq,
                    "reason": reason,
                    "reopened": reopened,
                },
            )
        )
        log.warn(
            "rolled back",
            checkpoint=checkpoint.id,
            commit=checkpoint.commit[:8],
            reason=reason,
            reopened=len(reopened),
        )
        return checkpoint

    def _reopen_discarded(self, checkpoint: Checkpoint) -> list[str]:
        """Nodes whose completed work this rollback just threw away."""
        if self.graph is None:
            return []
        from .graph import NodeStatus

        events = self.ledger.read(
            after_seq=checkpoint.seq, types=[EventType.NODE_SUCCEEDED]
        )
        reopened: list[str] = []
        for node_id in dict.fromkeys(e.node_id for e in events if e.node_id):
            node = self.graph.try_get(node_id)
            if node is None or node.status != NodeStatus.SUCCEEDED:
                continue
            self.graph.update(node_id, status=str(NodeStatus.READY), actor="rollback")
            reopened.append(node_id)

        # Demote afterwards, not inside the loop above. Events arrive
        # parent-first, so when A is reopened its dependent B -- which also
        # succeeded after the checkpoint -- is still SUCCEEDED and was skipped
        # by a check that only demoted READY nodes. B's own turn then set it
        # READY, and both became runnable at once: B could start against a tree
        # that no longer contains A's work. `promote_ready` puts B back when A
        # succeeds again.
        for node_id in reopened:
            for dependent in self.graph.dependents(node_id):
                child = self.graph.try_get(dependent)
                if child is not None and child.status == NodeStatus.READY:
                    self.graph.update(dependent, status=str(NodeStatus.PENDING), actor="rollback")
        return reopened

    def restore_for_attempt(self, node_id: str) -> str | None:
        """Reset the tree to the state a node should start from.

        Called before every attempt. This is what makes at-least-once execution
        of nodes safe: attempt two never inherits attempt one's half-written
        files, so a retry is a genuine retry rather than a compounding mess.
        Returns the commit restored to, or ``None`` when the tree was already
        clean.
        """
        if not self.repo.is_dirty():
            return None
        head = self.repo.head() if self.repo.has_commits() else "HEAD"
        self.repo.reset_hard(head)
        log.info("workspace reset before attempt", node=node_id, commit=head[:8])
        return head

    def prune(self, *, keep: int = 200) -> int:
        """Delete old checkpoint tags, keeping milestone ones forever.

        Tags are cheap but not free, and a year-long project would accumulate
        tens of thousands. Milestone checkpoints are the ones a human would ever
        name, so they are exempt.
        """
        checkpoints = self.list(limit=10_000)
        removable = [c for c in checkpoints if c.kind in ("node", "pre_attempt")]
        stale = removable[keep:]
        removed = 0
        for checkpoint in stale:
            try:
                self.repo._git("tag", "-d", checkpoint.tag, check=False)
                removed += 1
            except GitError:  # pragma: no cover
                continue
        return removed
