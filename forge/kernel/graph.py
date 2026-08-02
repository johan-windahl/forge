"""The task graph.

Forge does not run a fixed pipeline of "plan, then build, then test". A pipeline
cannot express "the renderer work can proceed while the audio work is blocked on
a decision", and it has no place to put work that the system discovers halfway
through. Instead the unit of work is a **node** in a dependency DAG, and the
graph grows as the project is understood: planning nodes emit implementation
nodes, implementation nodes emit validation and review nodes, a failed review
emits a repair node.

A node is the atom of durability. Everything a worker needs to resume after a
crash is in the node's row: its spec, its attempt count, the tier it escalated
to, the checkpoint it started from. Workers hold *leases* rather than locks, so
a worker that dies silently simply lets its lease lapse and another picks the
node up.

Idempotency is structural, not aspirational. Before every attempt the workspace
is reset to the node's base checkpoint, so a partially-applied previous attempt
cannot corrupt the next one. That is what makes at-least-once execution safe.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..errors import ConcurrencyError, InvariantError, LeaseLost
from ..obs.log import get_logger
from ..util.clock import Clock, default_clock
from ..util.ids import new_id
from .events import Event, EventType
from .ledger import Ledger

log = get_logger("kernel.graph")


class NodeKind(StrEnum):
    """What sort of work a node represents.

    The kind selects which agent handles it and which gates apply. Adding a kind
    is the primary extension point for new specialist agents -- see
    ``docs/architecture.md#extending``.
    """

    GOAL = "goal"  # the root; one per project
    PLAN = "plan"  # decompose a goal or epic into children
    ARCHITECT = "architect"  # produce/revise architecture and interfaces
    SCAFFOLD = "scaffold"  # create project skeleton, tooling, CI
    IMPLEMENT = "implement"  # write or change code
    TEST_AUTHOR = "test_author"  # write tests for existing behaviour
    DEBUG = "debug"  # diagnose and fix a specific failure
    REFACTOR = "refactor"
    REVIEW = "review"  # model-judged code review
    VALIDATE = "validate"  # run deterministic gates
    BROWSER_QA = "browser_qa"  # drive the app, capture screenshots/video
    VISUAL_REVIEW = "visual_review"  # judge captured media
    PERF = "perf"
    SECURITY = "security"
    DOCUMENT = "document"
    DEPLOY = "deploy"
    RETROSPECT = "retrospect"  # milestone self-critique
    IMPROVE = "improve"  # act on a retrospective finding


class NodeStatus(StrEnum):
    PENDING = "pending"  # dependencies not yet satisfied
    READY = "ready"  # runnable now
    RUNNING = "running"  # leased by a worker
    SUCCEEDED = "succeeded"
    FAILED = "failed"  # attempts exhausted; blocks dependents
    BLOCKED = "blocked"  # needs a human decision; does not block siblings
    CANCELLED = "cancelled"
    DEFERRED = "deferred"  # waiting on a backoff timer


#: Statuses from which no further work will be attempted.
TERMINAL = frozenset({NodeStatus.SUCCEEDED, NodeStatus.FAILED, NodeStatus.CANCELLED, NodeStatus.BLOCKED})
#: Statuses a dependent may not start from.
UNSATISFIED = frozenset(
    {NodeStatus.PENDING, NodeStatus.READY, NodeStatus.RUNNING, NodeStatus.DEFERRED, NodeStatus.FAILED,
     NodeStatus.BLOCKED, NodeStatus.CANCELLED}
)


@dataclass(slots=True)
class Node:
    """One unit of autonomous work."""

    id: str
    project_id: str
    kind: str
    title: str
    status: str = NodeStatus.PENDING
    spec: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    parent_id: str | None = None
    deps: list[str] = field(default_factory=list)
    priority: int = 100
    attempts: int = 0
    tier: str = "local"
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    not_before: float = 0.0
    version: int = 0
    milestone: str | None = None
    cost: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def acceptance(self) -> list[str]:
        """Acceptance criteria, in the node spec, as plain sentences.

        Every node carries these. They are what the reviewer judges against and
        what the retrospective checks were actually met -- a node that succeeds
        without satisfying its own acceptance criteria is a bug in the planner,
        and Forge can detect that mechanically because the criteria are data.
        """
        return list(self.spec.get("acceptance", []))

    @property
    def gates(self) -> list[str]:
        return list(self.spec.get("gates", []))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "spec": self.spec,
            "result": self.result,
            "deps": self.deps,
            "priority": self.priority,
            "attempts": self.attempts,
            "tier": self.tier,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "not_before": self.not_before,
            "version": self.version,
            "milestone": self.milestone,
            "cost": self.cost,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row, deps: list[str] | None = None) -> Node:
        return cls(
            id=row["id"],
            project_id=row["project_id"],
            parent_id=row["parent_id"],
            kind=row["kind"],
            title=row["title"],
            status=row["status"],
            spec=json.loads(row["spec"]) if row["spec"] else {},
            result=json.loads(row["result"]) if row["result"] else None,
            deps=deps or [],
            priority=row["priority"],
            attempts=row["attempts"],
            tier=row["tier"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            not_before=row["not_before"],
            version=row["version"],
            milestone=row["milestone"],
            cost=row["cost"],
        )

    def summary(self) -> str:
        return f"[{self.kind}] {self.title}"


@dataclass(slots=True)
class Lease:
    """A time-bounded claim on a node.

    Leases rather than locks: a lock held by a process that segfaults is held
    forever, whereas a lease simply expires. ``token`` makes the claim
    verifiable -- a worker that stalls past its expiry and wakes up mid-write
    finds its token no longer matches and aborts instead of corrupting the work
    of whoever took over.
    """

    node_id: str
    worker_id: str
    token: str
    acquired_at: float
    expires_at: float


# --------------------------------------------------------------------------
# Event application (shared by live writes and projection rebuild)
# --------------------------------------------------------------------------


def apply_node_event(conn: sqlite3.Connection, event: Event) -> None:
    """Fold one ``node.*`` event into the node projection.

    Deliberately total and side-effect free beyond the two tables it touches, so
    that replaying the entire log reproduces the exact live state. Unknown event
    types are ignored rather than raising, which keeps old ledgers readable.
    """
    p = event.payload
    node_id = event.node_id or p.get("id")
    if not node_id:
        return
    ts = event.ts

    match event.type:
        case EventType.NODE_CREATED:
            conn.execute(
                """INSERT OR REPLACE INTO nodes
                   (id, project_id, parent_id, kind, title, status, spec, result, priority,
                    attempts, tier, created_at, updated_at, started_at, finished_at,
                    not_before, version, milestone, cost)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    node_id,
                    p.get("project_id", event.project_id),
                    p.get("parent_id"),
                    p.get("kind", NodeKind.IMPLEMENT),
                    p.get("title", ""),
                    p.get("status", NodeStatus.PENDING),
                    json.dumps(p.get("spec", {}), default=str),
                    None,
                    int(p.get("priority", 100)),
                    0,
                    p.get("tier", "local"),
                    ts,
                    ts,
                    None,
                    None,
                    float(p.get("not_before", 0.0)),
                    0,
                    p.get("milestone"),
                    0.0,
                ),
            )
            conn.execute("DELETE FROM node_deps WHERE node_id = ?", (node_id,))
            for dep in p.get("deps", []):
                conn.execute(
                    "INSERT OR IGNORE INTO node_deps(node_id, depends_on) VALUES (?, ?)", (node_id, dep)
                )

        case EventType.NODE_UPDATED:
            sets, params = [], []
            # `attempts` is settable so an operator answering a blocked node can
            # hand it a fresh retry budget. Without that, unblocking gave the
            # node exactly one more attempt -- `plan_failure` compares the
            # lifetime count against `max_attempts`, so a node at 14 attempts
            # re-blocked on the first failure and reported "exhausted 14
            # attempts" against a limit of 4, which reads as a contradiction.
            for column in (
                "title", "priority", "milestone", "tier", "status", "not_before", "attempts",
            ):
                if column in p:
                    sets.append(f"{column} = ?")
                    params.append(p[column])
            if "spec" in p:
                sets.append("spec = ?")
                params.append(json.dumps(p["spec"], default=str))
            if "result" in p:
                sets.append("result = ?")
                params.append(json.dumps(p["result"], default=str) if p["result"] is not None else None)
            sets.append("updated_at = ?")
            params.append(ts)
            sets.append("version = version + 1")
            params.append(node_id)
            conn.execute(f"UPDATE nodes SET {', '.join(sets)} WHERE id = ?", params)
            if "deps" in p:
                conn.execute("DELETE FROM node_deps WHERE node_id = ?", (node_id,))
                for dep in p["deps"]:
                    conn.execute(
                        "INSERT OR IGNORE INTO node_deps(node_id, depends_on) VALUES (?, ?)",
                        (node_id, dep),
                    )

        case EventType.NODE_READY:
            conn.execute(
                "UPDATE nodes SET status = ?, updated_at = ?, not_before = 0 WHERE id = ?",
                (NodeStatus.READY, ts, node_id),
            )

        case EventType.NODE_LEASED:
            conn.execute(
                """INSERT OR REPLACE INTO leases(node_id, worker_id, acquired_at, expires_at, token)
                   VALUES (?, ?, ?, ?, ?)""",
                (node_id, p.get("worker_id", "?"), ts, float(p.get("expires_at", ts)), p.get("token", "")),
            )
            conn.execute(
                "UPDATE nodes SET status = ?, updated_at = ? WHERE id = ?",
                (NodeStatus.RUNNING, ts, node_id),
            )

        case EventType.NODE_LEASE_RENEWED:
            conn.execute(
                "UPDATE leases SET expires_at = ? WHERE node_id = ? AND token = ?",
                (float(p.get("expires_at", ts)), node_id, p.get("token", "")),
            )

        case EventType.NODE_LEASE_EXPIRED:
            conn.execute("DELETE FROM leases WHERE node_id = ?", (node_id,))
            conn.execute(
                "UPDATE nodes SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (NodeStatus.READY, ts, node_id, NodeStatus.RUNNING),
            )

        case EventType.NODE_STARTED:
            conn.execute(
                """UPDATE nodes SET status = ?, attempts = attempts + 1, started_at = ?,
                       updated_at = ?, tier = ? WHERE id = ?""",
                (NodeStatus.RUNNING, ts, ts, p.get("tier", "local"), node_id),
            )

        case EventType.NODE_SUCCEEDED:
            conn.execute("DELETE FROM leases WHERE node_id = ?", (node_id,))
            conn.execute(
                """UPDATE nodes SET status = ?, result = ?, finished_at = ?, updated_at = ?
                   WHERE id = ?""",
                (
                    NodeStatus.SUCCEEDED,
                    json.dumps(p.get("result", {}), default=str),
                    ts,
                    ts,
                    node_id,
                ),
            )

        case EventType.NODE_FAILED:
            conn.execute("DELETE FROM leases WHERE node_id = ?", (node_id,))
            terminal = bool(p.get("terminal"))
            conn.execute(
                """UPDATE nodes SET status = ?, result = ?, updated_at = ?,
                       finished_at = ?, not_before = ? WHERE id = ?""",
                (
                    NodeStatus.FAILED if terminal else NodeStatus.DEFERRED,
                    json.dumps(p.get("result", {"error": p.get("error")}), default=str),
                    ts,
                    ts if terminal else None,
                    float(p.get("not_before", 0.0)),
                    node_id,
                ),
            )

        case EventType.NODE_DEFERRED:
            conn.execute("DELETE FROM leases WHERE node_id = ?", (node_id,))
            conn.execute(
                "UPDATE nodes SET status = ?, not_before = ?, updated_at = ? WHERE id = ?",
                (NodeStatus.DEFERRED, float(p.get("not_before", ts)), ts, node_id),
            )

        case EventType.NODE_BLOCKED:
            conn.execute("DELETE FROM leases WHERE node_id = ?", (node_id,))
            conn.execute(
                """UPDATE nodes SET status = ?, result = ?, updated_at = ?, finished_at = ?
                   WHERE id = ?""",
                (NodeStatus.BLOCKED, json.dumps(p, default=str), ts, ts, node_id),
            )

        case EventType.NODE_CANCELLED:
            conn.execute("DELETE FROM leases WHERE node_id = ?", (node_id,))
            conn.execute(
                "UPDATE nodes SET status = ?, updated_at = ?, finished_at = ? WHERE id = ?",
                (NodeStatus.CANCELLED, ts, ts, node_id),
            )

        case EventType.NODE_ESCALATED:
            conn.execute(
                "UPDATE nodes SET tier = ?, updated_at = ? WHERE id = ?",
                (p.get("to_tier", "frontier"), ts, node_id),
            )

        case _:
            return


# --------------------------------------------------------------------------
# TaskGraph
# --------------------------------------------------------------------------


class TaskGraph:
    """Transactional operations over the node DAG.

    Every mutation writes the event and updates the projection inside one
    transaction, so the projection can never disagree with the log.
    """

    def __init__(self, ledger: Ledger, project_id: str, clock: Clock | None = None) -> None:
        self.ledger = ledger
        self.project_id = project_id
        self._clock = clock or default_clock()

    # -- creation --------------------------------------------------------

    def add_node(
        self,
        kind: str,
        title: str,
        *,
        spec: dict[str, Any] | None = None,
        deps: Sequence[str] = (),
        parent_id: str | None = None,
        priority: int = 100,
        milestone: str | None = None,
        node_id: str | None = None,
        actor: str = "system",
        causation_id: str | None = None,
    ) -> Node:
        """Create a node. Status is derived from whether its deps are satisfied."""
        node_id = node_id or new_id("node")
        deps = list(deps)
        with self.ledger.transaction() as conn:
            self._assert_no_cycle(conn, node_id, deps)
            status = NodeStatus.READY if self._deps_satisfied(conn, deps) else NodeStatus.PENDING
            event = Event(
                type=EventType.NODE_CREATED,
                node_id=node_id,
                project_id=self.project_id,
                actor=actor,
                causation_id=causation_id,
                correlation_id=node_id,
                payload={
                    "id": node_id,
                    "project_id": self.project_id,
                    "parent_id": parent_id,
                    "kind": str(kind),
                    "title": title,
                    "status": str(status),
                    "spec": spec or {},
                    "deps": deps,
                    "priority": priority,
                    "milestone": milestone,
                },
            )
            self.ledger.append(event, conn=conn, apply_projection=False)
            apply_node_event(conn, event)
        log.debug("node created", node=node_id, kind=str(kind), title=title, deps=len(deps))
        return self.get(node_id)

    def add_many(self, specs: Iterable[dict[str, Any]]) -> list[Node]:
        """Create several nodes that may depend on each other by *index*.

        Planners emit batches where node 3 depends on node 1, but ids do not
        exist yet. Passing ``deps: [1]`` (an int) resolves to the id of the
        batch's element at that index; string deps are treated as real ids.
        """
        specs = list(specs)
        ids = [spec.get("node_id") or new_id("node") for spec in specs]
        created: list[Node] = []
        for i, spec in enumerate(specs):
            deps = [
                ids[d] if isinstance(d, int) and 0 <= d < len(ids) else d
                for d in spec.get("deps", [])
            ]
            # A planner occasionally emits a self-reference. Dropping it is the
            # right autonomous response: the intent is obvious and refusing the
            # whole batch over one bad edge would stall the project.
            if ids[i] in deps:
                log.warn("dropping self-dependency", node=ids[i], title=spec.get("title", "")[:60])
                deps = [d for d in deps if d != ids[i]]
            created.append(
                self.add_node(
                    spec["kind"],
                    spec["title"],
                    spec=spec.get("spec", {}),
                    deps=deps,
                    parent_id=spec.get("parent_id"),
                    priority=spec.get("priority", 100),
                    milestone=spec.get("milestone"),
                    node_id=ids[i],
                    actor=spec.get("actor", "planner"),
                )
            )
        return created

    # -- reads -----------------------------------------------------------

    def get(self, node_id: str) -> Node:
        row = self.ledger.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise InvariantError("node not found", node_id=node_id)
        return Node.from_row(row, self._deps_of(self.ledger.conn, node_id))

    def try_get(self, node_id: str) -> Node | None:
        row = self.ledger.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return Node.from_row(row, self._deps_of(self.ledger.conn, node_id)) if row else None

    def all_nodes(self, *, status: str | Sequence[str] | None = None) -> list[Node]:
        conn = self.ledger.conn
        sql = "SELECT * FROM nodes WHERE project_id = ?"
        params: list[Any] = [self.project_id]
        if status:
            statuses = [status] if isinstance(status, str) else list(status)
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            params.extend(str(s) for s in statuses)
        sql += " ORDER BY priority, created_at, id"
        return [Node.from_row(row, self._deps_of(conn, row["id"])) for row in conn.execute(sql, params)]

    def children(self, node_id: str) -> list[Node]:
        conn = self.ledger.conn
        rows = conn.execute(
            "SELECT * FROM nodes WHERE parent_id = ? ORDER BY priority, created_at", (node_id,)
        )
        return [Node.from_row(row, self._deps_of(conn, row["id"])) for row in rows]

    def dependents(self, node_id: str) -> list[str]:
        return [
            row["node_id"]
            for row in self.ledger.conn.execute(
                "SELECT node_id FROM node_deps WHERE depends_on = ?", (node_id,)
            )
        ]

    @staticmethod
    def _deps_of(conn: sqlite3.Connection, node_id: str) -> list[str]:
        return [
            row["depends_on"]
            for row in conn.execute(
                "SELECT depends_on FROM node_deps WHERE node_id = ? ORDER BY depends_on", (node_id,)
            )
        ]

    def _deps_satisfied(self, conn: sqlite3.Connection, deps: Sequence[str]) -> bool:
        if not deps:
            return True
        placeholders = ",".join("?" * len(deps))
        rows = conn.execute(
            f"SELECT id, status FROM nodes WHERE id IN ({placeholders})", list(deps)
        ).fetchall()
        by_id = {row["id"]: row["status"] for row in rows}
        # A dependency that does not exist yet is treated as unsatisfied rather
        # than as an error: planners legitimately create forward references
        # inside a batch.
        return all(by_id.get(dep) == NodeStatus.SUCCEEDED for dep in deps)

    def _assert_no_cycle(self, conn: sqlite3.Connection, node_id: str, deps: Sequence[str]) -> None:
        """Reject a dependency edge that would close a cycle.

        Checked at insert time rather than at schedule time so a bad plan fails
        immediately and visibly, instead of producing a graph that silently
        never becomes ready.
        """
        seen: set[str] = set()
        frontier = list(deps)
        while frontier:
            current = frontier.pop()
            if current == node_id:
                raise InvariantError("dependency cycle", node_id=node_id, via=sorted(seen))
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(
                row["depends_on"]
                for row in conn.execute("SELECT depends_on FROM node_deps WHERE node_id = ?", (current,))
            )

    # -- state transitions ----------------------------------------------

    def _emit(self, type: str, node_id: str, actor: str = "system", **payload: Any) -> Event:
        with self.ledger.transaction() as conn:
            event = Event(
                type=type,
                node_id=node_id,
                project_id=self.project_id,
                actor=actor,
                correlation_id=node_id,
                payload=payload,
            )
            self.ledger.append(event, conn=conn, apply_projection=False)
            apply_node_event(conn, event)
        return event

    def update(self, node_id: str, actor: str = "system", **changes: Any) -> Node:
        self._emit(EventType.NODE_UPDATED, node_id, actor, **changes)
        return self.get(node_id)

    def start(self, node_id: str, *, tier: str, worker_id: str) -> Node:
        self._emit(EventType.NODE_STARTED, node_id, worker_id, tier=tier)
        return self.get(node_id)

    def succeed(self, node_id: str, result: dict[str, Any] | None = None, actor: str = "system") -> Node:
        """Mark a node done and promote any dependents whose deps are now met."""
        self._emit(EventType.NODE_SUCCEEDED, node_id, actor, result=result or {})
        self.promote_ready(self.dependents(node_id))
        return self.get(node_id)

    def fail(
        self,
        node_id: str,
        error: str,
        *,
        terminal: bool,
        retry_at: float = 0.0,
        detail: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> Node:
        self._emit(
            EventType.NODE_FAILED,
            node_id,
            actor,
            error=error,
            terminal=terminal,
            not_before=retry_at,
            result={"error": error, **(detail or {})},
        )
        if terminal:
            # Dependents can never run; cascade so the scheduler is not left
            # holding nodes that will wait forever.
            self.cascade_block(node_id, reason=f"dependency {node_id} failed")
        return self.get(node_id)

    def block(self, node_id: str, reason: str, *, question: str = "", actor: str = "system") -> Node:
        """Park a node for human input without stalling the rest of the graph."""
        self._emit(EventType.NODE_BLOCKED, node_id, actor, reason=reason, question=question)
        log.warn("node blocked", node=node_id, reason=reason, question=question)
        return self.get(node_id)

    def defer(self, node_id: str, until: float, reason: str = "", actor: str = "system") -> Node:
        self._emit(EventType.NODE_DEFERRED, node_id, actor, not_before=until, reason=reason)
        return self.get(node_id)

    def cancel(self, node_id: str, reason: str = "", actor: str = "system") -> Node:
        self._emit(EventType.NODE_CANCELLED, node_id, actor, reason=reason)
        return self.get(node_id)

    def escalate(self, node_id: str, to_tier: str, reason: str, actor: str = "system") -> Node:
        self._emit(EventType.NODE_ESCALATED, node_id, actor, to_tier=to_tier, reason=reason)
        log.info("node escalated", node=node_id, tier=to_tier, reason=reason)
        return self.get(node_id)

    def cascade_block(self, node_id: str, reason: str) -> list[str]:
        """Block the transitive dependents of a terminally failed node."""
        blocked: list[str] = []
        frontier = list(self.dependents(node_id))
        while frontier:
            current = frontier.pop()
            node = self.try_get(current)
            if node is None or node.is_terminal:
                continue
            self._emit(EventType.NODE_BLOCKED, current, "system", reason=reason, cascaded_from=node_id)
            blocked.append(current)
            frontier.extend(self.dependents(current))
        return blocked

    def promote_ready(self, candidates: Sequence[str] | None = None) -> list[str]:
        """Move nodes whose dependencies are now satisfied into ``ready``.

        Called after every success, and also periodically by the scheduler as a
        safety net -- a crash between "succeed" and "promote" would otherwise
        leave a dependent stuck in ``pending`` forever.
        """
        promoted: list[str] = []
        conn = self.ledger.conn
        if candidates is None:
            rows = conn.execute(
                "SELECT id FROM nodes WHERE project_id = ? AND status = ?",
                (self.project_id, NodeStatus.PENDING),
            ).fetchall()
            candidates = [row["id"] for row in rows]
        for node_id in candidates:
            node = self.try_get(node_id)
            if node is None or node.status != NodeStatus.PENDING:
                continue
            if self._deps_satisfied(conn, node.deps):
                self._emit(EventType.NODE_READY, node_id)
                promoted.append(node_id)
        return promoted

    # -- leases ----------------------------------------------------------

    def claim(self, node_id: str, worker_id: str, lease_seconds: float) -> Lease:
        """Atomically take a lease on a ready node.

        The ``INSERT`` into ``leases`` is the mutual-exclusion primitive: the
        primary key means exactly one worker wins, decided by SQLite rather than
        by application logic.
        """
        token = new_id("lease")
        now = self._clock.now()
        expires = now + lease_seconds
        with self.ledger.transaction() as conn:
            row = conn.execute("SELECT status FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if row is None:
                raise InvariantError("cannot claim unknown node", node_id=node_id)
            if row["status"] not in (NodeStatus.READY, NodeStatus.DEFERRED):
                raise ConcurrencyError("node not claimable", node_id=node_id, status=row["status"])
            existing = conn.execute(
                "SELECT expires_at FROM leases WHERE node_id = ?", (node_id,)
            ).fetchone()
            if existing and existing["expires_at"] > now:
                raise ConcurrencyError("node already leased", node_id=node_id)
            event = Event(
                type=EventType.NODE_LEASED,
                node_id=node_id,
                project_id=self.project_id,
                actor=worker_id,
                correlation_id=node_id,
                payload={"worker_id": worker_id, "token": token, "expires_at": expires},
            )
            self.ledger.append(event, conn=conn, apply_projection=False)
            apply_node_event(conn, event)
        return Lease(node_id=node_id, worker_id=worker_id, token=token, acquired_at=now, expires_at=expires)

    def renew(self, lease: Lease, lease_seconds: float) -> Lease:
        """Extend a lease. Raises :class:`LeaseLost` if it was stolen."""
        expires = self._clock.now() + lease_seconds
        with self.ledger.transaction() as conn:
            row = conn.execute("SELECT token FROM leases WHERE node_id = ?", (lease.node_id,)).fetchone()
            if row is None or row["token"] != lease.token:
                raise LeaseLost("lease no longer held", node_id=lease.node_id, worker=lease.worker_id)
            event = Event(
                type=EventType.NODE_LEASE_RENEWED,
                node_id=lease.node_id,
                project_id=self.project_id,
                actor=lease.worker_id,
                correlation_id=lease.node_id,
                payload={"token": lease.token, "expires_at": expires},
            )
            self.ledger.append(event, conn=conn, apply_projection=False)
            apply_node_event(conn, event)
        lease.expires_at = expires
        return lease

    def verify_lease(self, lease: Lease) -> None:
        row = self.ledger.conn.execute(
            "SELECT token, expires_at FROM leases WHERE node_id = ?", (lease.node_id,)
        ).fetchone()
        if row is None or row["token"] != lease.token:
            raise LeaseLost("lease lost", node_id=lease.node_id)
        if row["expires_at"] <= self._clock.now():
            raise LeaseLost("lease expired", node_id=lease.node_id)

    def release(self, lease: Lease) -> None:
        with self.ledger.transaction() as conn:
            conn.execute(
                "DELETE FROM leases WHERE node_id = ? AND token = ?", (lease.node_id, lease.token)
            )

    def release_all_leases(self) -> list[str]:
        """Reclaim every lease, regardless of expiry. Startup only.

        A process that has just started holds no leases, so any lease in the
        table belongs to a previous, now-dead process. Waiting for those to
        expire naturally would idle the whole project for up to a full lease
        period after every crash -- which is exactly the moment when resuming
        promptly matters most.
        """
        rows = self.ledger.conn.execute(
            "SELECT node_id, worker_id FROM leases", ()
        ).fetchall()
        released: list[str] = []
        for row in rows:
            self._emit(
                EventType.NODE_LEASE_EXPIRED,
                row["node_id"],
                "recovery",
                worker_id=row["worker_id"],
                reason="process restart",
            )
            released.append(row["node_id"])
        if released:
            log.info("released leases from a previous process", count=len(released))
        return released

    def next_wakeup(self) -> float | None:
        """When the earliest deferred node becomes runnable, if any."""
        row = self.ledger.conn.execute(
            """SELECT MIN(not_before) AS t FROM nodes
               WHERE project_id = ? AND status IN (?, ?) AND not_before > ?""",
            (self.project_id, NodeStatus.READY, NodeStatus.DEFERRED, self._clock.now()),
        ).fetchone()
        return float(row["t"]) if row and row["t"] else None

    def reap_expired_leases(self) -> list[str]:
        """Return nodes whose worker died, restoring them to ``ready``.

        This is the mechanism that makes crash recovery automatic: on restart,
        every node the dead process held becomes runnable again after its lease
        window, with its attempt count preserved so retries stay bounded.
        """
        now = self._clock.now()
        reaped: list[str] = []
        rows = self.ledger.conn.execute(
            "SELECT node_id, worker_id FROM leases WHERE expires_at <= ?", (now,)
        ).fetchall()
        for row in rows:
            self._emit(
                EventType.NODE_LEASE_EXPIRED, row["node_id"], "reaper", worker_id=row["worker_id"]
            )
            reaped.append(row["node_id"])
        if reaped:
            log.warn("reclaimed expired leases", count=len(reaped), nodes=reaped[:5])
        return reaped

    # -- scheduling queries ---------------------------------------------

    def has_active_work(self, *, excluding_barriers: bool = True) -> bool:
        """Whether any non-barrier node can still make progress."""
        rows = self.ledger.conn.execute(
            """SELECT spec FROM nodes
               WHERE project_id = ? AND status IN (?, ?, ?, ?)""",
            (
                self.project_id,
                NodeStatus.PENDING,
                NodeStatus.READY,
                NodeStatus.RUNNING,
                NodeStatus.DEFERRED,
            ),
        ).fetchall()
        for row in rows:
            spec = json.loads(row["spec"]) if row["spec"] else {}
            if excluding_barriers and spec.get("barrier"):
                continue
            return True
        return False

    def runnable(self, limit: int = 16) -> list[Node]:
        """Nodes that may be started right now, best first.

        Ordering is priority, then creation time. Priority is set by the planner
        so that work on the critical path -- and work that unblocks the most
        dependents -- is picked up first.

        Nodes whose spec sets ``barrier`` are held back until nothing else can
        make progress. That is how a node can depend on "everything", which
        static edges cannot express in a graph that grows while it executes --
        the project's own goal node is the canonical example.
        """
        now = self._clock.now()
        barriers_held = self.has_active_work()
        rows = self.ledger.conn.execute(
            """SELECT n.* FROM nodes n
               LEFT JOIN leases l ON l.node_id = n.id AND l.expires_at > ?
               WHERE n.project_id = ?
                 AND n.status IN (?, ?)
                 AND n.not_before <= ?
                 AND l.node_id IS NULL
               ORDER BY n.priority, n.created_at, n.id
               LIMIT ?""",
            (now, self.project_id, NodeStatus.READY, NodeStatus.DEFERRED, now, limit),
        ).fetchall()
        nodes = [Node.from_row(row, self._deps_of(self.ledger.conn, row["id"])) for row in rows]
        if barriers_held:
            nodes = [n for n in nodes if not n.spec.get("barrier")]
        return nodes

    def counts(self) -> dict[str, int]:
        rows = self.ledger.conn.execute(
            "SELECT status, COUNT(*) AS c FROM nodes WHERE project_id = ? GROUP BY status",
            (self.project_id,),
        )
        counts = {str(s): 0 for s in NodeStatus}
        counts.update({row["status"]: row["c"] for row in rows})
        return counts

    def is_quiescent(self) -> bool:
        """True when no node can make progress without human input."""
        counts = self.counts()
        active = (
            counts[NodeStatus.PENDING]
            + counts[NodeStatus.READY]
            + counts[NodeStatus.RUNNING]
            + counts[NodeStatus.DEFERRED]
        )
        return active == 0

    def progress(self) -> float:
        counts = self.counts()
        total = sum(counts.values()) - counts[NodeStatus.CANCELLED]
        if total <= 0:
            return 0.0
        done = counts[NodeStatus.SUCCEEDED]
        return done / total

    def critical_path(self) -> list[Node]:
        """Longest chain of unfinished nodes -- what actually gates completion.

        Reported to the operator so a days-long run has an honest answer to "how
        far along is it?" that is not just a percentage of node counts.
        """
        nodes = {n.id: n for n in self.all_nodes()}
        memo: dict[str, tuple[int, list[str]]] = {}
        # Cancelled work is finished, just not by being done. Treating only
        # SUCCEEDED as terminal put cancelled nodes on the critical path, so a
        # completed project reported that it was gated on a proposal it had
        # already declined -- in one case "Set default model for
        # 'implementation' task class to haiku", which read as though routing
        # had been changed to start on a cloud rung. It had not.
        done = {NodeStatus.SUCCEEDED, NodeStatus.CANCELLED}

        def depth(node_id: str, seen: frozenset[str] = frozenset()) -> tuple[int, list[str]]:
            if node_id in memo:
                return memo[node_id]
            if node_id in seen:  # pragma: no cover - cycles rejected at insert
                return 0, []
            node = nodes.get(node_id)
            if node is None or node.status in done:
                return 0, []
            best_len, best_path = 0, []
            for dependent in self.dependents(node_id):
                length, path = depth(dependent, seen | {node_id})
                if length > best_len:
                    best_len, best_path = length, path
            result = (best_len + 1, [node_id, *best_path])
            memo[node_id] = result
            return result

        roots = [n.id for n in nodes.values() if not n.deps and n.status not in done]
        best: list[str] = []
        for root in roots:
            _, path = depth(root)
            if len(path) > len(best):
                best = path
        return [nodes[i] for i in best if i in nodes]
