"""The append-only ledger: Forge's single source of truth.

SQLite in WAL mode is the storage engine. That choice deserves defending, since
"just use Postgres" is the reflex. Forge's write pattern is one process, a
handful of worker threads, a few hundred small appends per hour, and reads that
are almost always "everything since sequence N". SQLite handles that with an
fsync per commit, no daemon to supervise, no port to secure, and a database that
is one file the operator can copy, diff or email. For a system whose main job is
surviving crashes on a single box, removing a network dependency from the
durability path is the robust choice, not the lazy one.

Concurrency is handled by keeping one connection per thread (SQLite connections
are not thread-safe) and serialising writers through a process-level lock. WAL
mode means readers never block the writer.

The projection tables (``nodes``, ``leases``, ``gate_results``, ``budget``,
``kv``) live in the same database and are updated *in the same transaction* as
the event append. That gives atomicity between "the fact was recorded" and "the
derived state reflects it" without the read-side cost of replaying the log on
every query -- while :meth:`Ledger.rebuild_projections` can still throw all
derived state away and recompute it from events alone.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..errors import ConcurrencyError, LedgerError
from ..obs.log import get_logger
from ..util.clock import Clock, default_clock
from .events import Event

log = get_logger("kernel.ledger")

SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 15000;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- The log. Never updated, never deleted (see compact()).
CREATE TABLE IF NOT EXISTS events (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    id             TEXT NOT NULL UNIQUE,
    ts             REAL NOT NULL,
    type           TEXT NOT NULL,
    project_id     TEXT,
    node_id        TEXT,
    actor          TEXT NOT NULL DEFAULT 'system',
    causation_id   TEXT,
    correlation_id TEXT,
    payload        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, seq);
CREATE INDEX IF NOT EXISTS idx_events_node ON events(node_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_corr ON events(correlation_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(ts);

-- Projection: the task graph.
CREATE TABLE IF NOT EXISTS nodes (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    parent_id     TEXT,
    kind          TEXT NOT NULL,
    title         TEXT NOT NULL,
    status        TEXT NOT NULL,
    spec          TEXT NOT NULL DEFAULT '{}',
    result        TEXT,
    priority      INTEGER NOT NULL DEFAULT 100,
    attempts      INTEGER NOT NULL DEFAULT 0,
    tier          TEXT NOT NULL DEFAULT 'local',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    not_before    REAL NOT NULL DEFAULT 0,
    version       INTEGER NOT NULL DEFAULT 0,
    milestone     TEXT,
    cost          REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(project_id, status, priority, id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_milestone ON nodes(milestone);

CREATE TABLE IF NOT EXISTS node_deps (
    node_id    TEXT NOT NULL,
    depends_on TEXT NOT NULL,
    PRIMARY KEY (node_id, depends_on)
);
CREATE INDEX IF NOT EXISTS idx_deps_reverse ON node_deps(depends_on);

-- Projection: who is working on what, and until when.
CREATE TABLE IF NOT EXISTS leases (
    node_id    TEXT PRIMARY KEY,
    worker_id  TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    token      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_leases_expiry ON leases(expires_at);

-- Projection: deterministic validation outcomes, keyed by content.
CREATE TABLE IF NOT EXISTS gate_results (
    cache_key  TEXT PRIMARY KEY,
    gate       TEXT NOT NULL,
    node_id    TEXT,
    passed     INTEGER NOT NULL,
    score      REAL,
    duration   REAL NOT NULL DEFAULT 0,
    summary    TEXT,
    detail     TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gate_node ON gate_results(node_id, gate);

-- Projection: spend, one row per model call.
CREATE TABLE IF NOT EXISTS spend (
    id            TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    day           TEXT NOT NULL,
    node_id       TEXT,
    model         TEXT NOT NULL,
    tier          TEXT NOT NULL,
    hosted        TEXT NOT NULL DEFAULT 'cloud',
    task_class    TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    cost          REAL NOT NULL DEFAULT 0,
    escalation    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_spend_day ON spend(day);
CREATE INDEX IF NOT EXISTS idx_spend_node ON spend(node_id);
-- Rolling per-model quota lookups run on every routing decision, and this
-- projection only ever grows. Without this the hot path is a table scan.
CREATE INDEX IF NOT EXISTS idx_spend_model_ts ON spend(model, ts);

-- Projection: routing outcomes feeding the adaptive policy.
CREATE TABLE IF NOT EXISTS routing_stats (
    task_class TEXT NOT NULL,
    tier       TEXT NOT NULL,
    successes  INTEGER NOT NULL DEFAULT 0,
    failures   INTEGER NOT NULL DEFAULT 0,
    cost       REAL NOT NULL DEFAULT 0,
    latency    REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (task_class, tier)
);

-- Version 2 contains exactly one validated node outcome per attempt (plus
-- terminal format failures).  The original projection mixed schema success,
-- gate verdicts and stale responses from earlier attempts, so its counts are
-- retained only for ledger compatibility and historical inspection.
CREATE TABLE IF NOT EXISTS routing_stats_v2 (
    task_class TEXT NOT NULL,
    tier       TEXT NOT NULL,
    successes  INTEGER NOT NULL DEFAULT 0,
    failures   INTEGER NOT NULL DEFAULT 0,
    cost       REAL NOT NULL DEFAULT 0,
    latency    REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (task_class, tier)
);

-- Projection: metric samples (rolled up; raw samples stay in events).
CREATE TABLE IF NOT EXISTS metrics (
    name       TEXT NOT NULL,
    labels     TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL,
    count      INTEGER NOT NULL DEFAULT 0,
    total      REAL NOT NULL DEFAULT 0,
    last       REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (name, labels)
);

-- Generic durable key/value for small singletons (cursors, config snapshots).
CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    version    INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
"""


class Ledger:
    """Durable event log plus its transactional projections.

    Thread-safety model: one SQLite connection per thread (kept in thread-local
    storage), one process-wide reentrant lock around write transactions. Reads
    proceed concurrently under WAL.
    """

    def __init__(self, path: Path | str, *, clock: Clock | None = None, project_id: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or default_clock()
        self.project_id = project_id
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._subscribers: list[Callable[[Event], None]] = []
        self._closed = False
        self._init_schema()

    # -- connection management -------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            try:
                conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            except sqlite3.Error as exc:  # pragma: no cover
                raise LedgerError(f"cannot open ledger at {self.path}: {exc}") from exc
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 15000")
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._write_lock:
            conn = self.conn
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),)
                )
            elif int(row["value"]) > SCHEMA_VERSION:
                raise LedgerError(
                    "ledger was written by a newer Forge",
                    found=int(row["value"]),
                    supported=SCHEMA_VERSION,
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """An IMMEDIATE transaction. All writes must go through this."""
        with self._write_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def close(self) -> None:
        self._closed = True
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- append ----------------------------------------------------------

    def append(
        self,
        event: Event,
        *,
        conn: sqlite3.Connection | None = None,
        apply_projection: bool = True,
    ) -> Event:
        """Append one event, atomically with its projection update.

        Pass ``conn`` to enlist in a caller-managed transaction, which is how
        the graph writes an event and the node row it implies as one unit.
        """
        if not event.project_id:
            event.project_id = self.project_id
        event.ts = event.ts or self._clock.now()

        def _write(c: sqlite3.Connection) -> None:
            cursor = c.execute(
                """INSERT INTO events (id, ts, type, project_id, node_id, actor,
                                       causation_id, correlation_id, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.id,
                    event.ts,
                    event.type,
                    event.project_id,
                    event.node_id,
                    event.actor,
                    event.causation_id,
                    event.correlation_id,
                    json.dumps(event.payload, default=str),
                ),
            )
            event.seq = int(cursor.lastrowid or 0)
            if apply_projection:
                self._project(c, event)

        try:
            if conn is not None:
                _write(conn)
            else:
                with self.transaction() as c:
                    _write(c)
        except sqlite3.IntegrityError as exc:
            raise ConcurrencyError(f"duplicate event id {event.id}", event_type=event.type) from exc
        except sqlite3.Error as exc:
            raise LedgerError(f"append failed: {exc}", event_type=event.type) from exc

        self._notify(event)
        return event

    def append_many(self, events: Sequence[Event]) -> list[Event]:
        """Append a batch atomically. Either all land or none do."""
        written: list[Event] = []
        with self.transaction() as conn:
            for event in events:
                written.append(self.append(event, conn=conn))
        return written

    def emit(self, type: str, /, node_id: str | None = None, actor: str = "system", **payload: Any) -> Event:
        """Convenience constructor-and-append for the common case."""
        return self.append(Event(type=type, node_id=node_id, actor=actor, payload=payload))

    # -- subscriptions ---------------------------------------------------

    def subscribe(self, handler: Callable[[Event], None]) -> Callable[[], None]:
        """Register an in-process listener. Returns an unsubscribe callable.

        Subscribers are for live concerns only -- progress display, dashboard
        refresh. Durable reactions belong in projections, because a subscriber
        that was not running when the event was appended never sees it.
        """
        self._subscribers.append(handler)

        def _unsubscribe() -> None:
            with suppress_value_error():
                self._subscribers.remove(handler)

        return _unsubscribe

    def _notify(self, event: Event) -> None:
        for handler in list(self._subscribers):
            try:
                handler(event)
            except Exception as exc:  # pragma: no cover - listeners must not break writes
                log.warn("event subscriber raised", error=str(exc), event_type=event.type)

    # -- read ------------------------------------------------------------

    def read(
        self,
        *,
        after_seq: int = 0,
        types: Sequence[str] | None = None,
        node_id: str | None = None,
        correlation_id: str | None = None,
        since_ts: float | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        clauses = ["seq > ?"]
        params: list[Any] = [after_seq]
        if types:
            clauses.append(f"type IN ({','.join('?' * len(types))})")
            params.extend(types)
        if node_id:
            clauses.append("node_id = ?")
            params.append(node_id)
        if correlation_id:
            clauses.append("correlation_id = ?")
            params.append(correlation_id)
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(since_ts)
        sql = f"SELECT * FROM events WHERE {' AND '.join(clauses)} ORDER BY seq"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [Event.from_row(row) for row in self.conn.execute(sql, params)]

    def tail(self, count: int = 50, types: Sequence[str] | None = None) -> list[Event]:
        """The most recent ``count`` events, in chronological order."""
        params: list[Any] = []
        where = ""
        if types:
            where = f"WHERE type IN ({','.join('?' * len(types))})"
            params.extend(types)
        rows = self.conn.execute(
            f"SELECT * FROM events {where} ORDER BY seq DESC LIMIT ?", (*params, count)
        ).fetchall()
        return [Event.from_row(row) for row in reversed(rows)]

    def head_seq(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq), 0) AS s FROM events").fetchone()
        return int(row["s"])

    def count(self, type: str | None = None) -> int:
        if type:
            row = self.conn.execute("SELECT COUNT(*) AS c FROM events WHERE type = ?", (type,)).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()
        return int(row["c"])

    # -- key/value -------------------------------------------------------

    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def kv_set(self, key: str, value: Any, *, expected_version: int | None = None) -> int:
        """Write a value; with ``expected_version`` this is a compare-and-swap."""
        with self.transaction() as conn:
            row = conn.execute("SELECT version FROM kv WHERE key = ?", (key,)).fetchone()
            current = int(row["version"]) if row else 0
            if expected_version is not None and current != expected_version:
                raise ConcurrencyError(
                    "kv version mismatch", key=key, expected=expected_version, actual=current
                )
            version = current + 1
            conn.execute(
                """INSERT INTO kv(key, value, version, updated_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                        version=excluded.version, updated_at=excluded.updated_at""",
                (key, json.dumps(value, default=str), version, self._clock.now()),
            )
            return version

    def kv_delete(self, key: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM kv WHERE key = ?", (key,))

    # -- metrics sink ----------------------------------------------------

    def record_metric(self, name: str, value: float, kind: str, labels: dict[str, str]) -> None:
        """Implements :class:`forge.obs.metrics.MetricSink`.

        Metric writes are best-effort and deliberately do *not* append an event
        per sample -- at hundreds of samples per node that would drown the log.
        Aggregates live in their own table; the events that matter (spend,
        gate outcomes) are recorded explicitly by their subsystems.
        """
        rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        try:
            with self.transaction() as conn:
                conn.execute(
                    """INSERT INTO metrics(name, labels, kind, count, total, last, updated_at)
                       VALUES (?, ?, ?, 1, ?, ?, ?)
                       ON CONFLICT(name, labels) DO UPDATE SET
                           count = count + 1,
                           total = total + excluded.total,
                           last = excluded.last,
                           updated_at = excluded.updated_at""",
                    (name, rendered, kind, value, value, self._clock.now()),
                )
        except (LedgerError, sqlite3.Error):  # pragma: no cover
            pass

    def metrics_snapshot(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM metrics ORDER BY name, labels")]

    # -- projections -----------------------------------------------------

    def _project(self, conn: sqlite3.Connection, event: Event) -> None:
        """Apply an event's effect to derived tables.

        Only a few event types carry projection logic here; node mutations are
        applied by :mod:`forge.kernel.graph`, which owns that table and writes
        event and row in one transaction. This hook covers the cross-cutting
        projections (spend, gate results, routing stats) so any subsystem can
        emit the event and get the projection for free.
        """
        from .events import EventType as E

        payload = event.payload
        now = event.ts

        if event.type == E.BUDGET_SPENT:
            import datetime

            day = datetime.datetime.fromtimestamp(now, tz=datetime.UTC).strftime("%Y-%m-%d")
            conn.execute(
                """INSERT OR REPLACE INTO spend
                   (id, ts, day, node_id, model, tier, hosted, task_class,
                    input_tokens, output_tokens, cached_tokens, cost, escalation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.id,
                    now,
                    day,
                    event.node_id,
                    payload.get("model", "?"),
                    payload.get("tier", "?"),
                    payload.get("hosted", "cloud"),
                    payload.get("task_class"),
                    int(payload.get("input_tokens", 0)),
                    int(payload.get("output_tokens", 0)),
                    int(payload.get("cached_tokens", 0)),
                    float(payload.get("cost", 0.0)),
                    1 if payload.get("escalation") else 0,
                ),
            )
            if event.node_id:
                conn.execute(
                    "UPDATE nodes SET cost = cost + ? WHERE id = ?",
                    (float(payload.get("cost", 0.0)), event.node_id),
                )

        elif event.type in (E.GATE_PASSED, E.GATE_FAILED, E.GATE_ERRORED):
            key = payload.get("cache_key")
            if key:
                conn.execute(
                    """INSERT OR REPLACE INTO gate_results
                       (cache_key, gate, node_id, passed, score, duration, summary, detail, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        key,
                        payload.get("gate", "?"),
                        event.node_id,
                        1 if event.type == E.GATE_PASSED else 0,
                        payload.get("score"),
                        float(payload.get("duration", 0.0)),
                        payload.get("summary"),
                        json.dumps(payload.get("detail"), default=str) if payload.get("detail") else None,
                        now,
                    ),
                )

        elif event.type == E.ROUTE_DECIDED and payload.get("outcome") in ("success", "failure"):
            success = 1 if payload["outcome"] == "success" else 0
            conn.execute(
                """INSERT INTO routing_stats(task_class, tier, successes, failures, cost, latency, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(task_class, tier) DO UPDATE SET
                       successes = successes + excluded.successes,
                       failures  = failures + excluded.failures,
                       cost      = cost + excluded.cost,
                       latency   = latency + excluded.latency,
                       updated_at = excluded.updated_at""",
                (
                    payload.get("task_class", "unknown"),
                    payload.get("tier", "unknown"),
                    success,
                    1 - success,
                    float(payload.get("cost", 0.0)),
                    float(payload.get("latency", 0.0)),
                    now,
                ),
            )
            if int(payload.get("feedback_version", 1)) >= 2:
                conn.execute(
                    """INSERT INTO routing_stats_v2(task_class, tier, successes, failures, cost, latency, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(task_class, tier) DO UPDATE SET
                           successes = successes + excluded.successes,
                           failures  = failures + excluded.failures,
                           cost      = cost + excluded.cost,
                           latency   = latency + excluded.latency,
                           updated_at = excluded.updated_at""",
                    (
                        payload.get("task_class", "unknown"),
                        payload.get("tier", "unknown"),
                        success,
                        1 - success,
                        float(payload.get("cost", 0.0)),
                        float(payload.get("latency", 0.0)),
                        now,
                    ),
                )

    def rebuild_projections(self) -> dict[str, int]:
        """Discard all derived state and recompute it from the event log.

        This is the escape hatch that makes event sourcing worth the trouble: if
        a projection is ever corrupted -- by a bug, a partial disk write, a
        schema change -- the fix is to throw it away, not to reason about how it
        got wrong. ``forge repair`` calls this.
        """
        from .graph import apply_node_event

        counts = {"events": 0, "nodes": 0}
        with self.transaction() as conn:
            for table in (
                "nodes",
                "node_deps",
                "leases",
                "gate_results",
                "spend",
                "routing_stats",
                "routing_stats_v2",
            ):
                conn.execute(f"DELETE FROM {table}")
            for row in conn.execute("SELECT * FROM events ORDER BY seq"):
                event = Event.from_row(row)
                counts["events"] += 1
                if event.type.startswith("node."):
                    apply_node_event(conn, event)
                else:
                    self._project(conn, event)
            counts["nodes"] = int(conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"])
        log.info("projections rebuilt", **counts)
        return counts

    # -- maintenance -----------------------------------------------------

    def vacuum(self) -> None:
        self.conn.execute("VACUUM")

    def checkpoint_wal(self) -> None:
        """Fold the WAL back into the main database file.

        Called before creating a checkpoint so that copying the ``.db`` file
        captures a complete, self-consistent snapshot.
        """
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def stats(self) -> dict[str, Any]:
        conn = self.conn
        by_type = {
            row["type"]: row["c"]
            for row in conn.execute("SELECT type, COUNT(*) AS c FROM events GROUP BY type ORDER BY c DESC")
        }
        return {
            "path": str(self.path),
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "events": self.count(),
            "head_seq": self.head_seq(),
            "by_type": by_type,
        }


class suppress_value_error:
    """Tiny context manager; avoids importing contextlib at call sites."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> bool:
        return exc_type is ValueError
