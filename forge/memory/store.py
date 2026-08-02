"""Durable project memory.

Records are written as ledger events and projected into a queryable table, the
same pattern the task graph uses. So memory inherits crash safety, replayability
and a full audit trail for free -- "when did the system start believing this?"
is answerable down to the event.

Retrieval combines a BM25 index over record text with kind weighting and path
affinity. The index is rebuilt from the table on demand rather than maintained
incrementally, because at Forge's corpus sizes a rebuild is milliseconds and a
stale index is a bug that would take days to notice.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from ..kernel.events import Event, EventType
from ..kernel.ledger import Ledger
from ..obs.log import get_logger
from ..util.bm25 import Document, Index
from ..util.clock import Clock, default_clock
from .records import KIND_WEIGHTS, MemoryKind, MemoryRecord, MemoryStatus

log = get_logger("memory.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL DEFAULT '',
    kind          TEXT NOT NULL,
    title         TEXT NOT NULL,
    body          TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '[]',
    paths         TEXT NOT NULL DEFAULT '[]',
    confidence    REAL NOT NULL DEFAULT 0.7,
    status        TEXT NOT NULL DEFAULT 'active',
    source        TEXT NOT NULL DEFAULT 'system',
    data          TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    superseded_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory(kind, status);
CREATE INDEX IF NOT EXISTS idx_memory_status ON memory(status, updated_at);
"""


class MemoryStore:
    def __init__(self, ledger: Ledger, project_id: str = "", clock: Clock | None = None) -> None:
        self.ledger = ledger
        self.project_id = project_id or ledger.project_id
        self._clock = clock or default_clock()
        self._lock = threading.Lock()
        self._index: Index | None = None
        self._index_version = -1
        # DDL runs outside a transaction: sqlite3's executescript issues an
        # implicit COMMIT, which would tear down an enclosing transaction.
        for statement in filter(None, (s.strip() for s in _SCHEMA.split(";"))):
            ledger.conn.execute(statement)

    # -- writing ---------------------------------------------------------

    def write(self, record: MemoryRecord, *, node_id: str | None = None) -> MemoryRecord:
        """Persist a record. Supersedes any active record with the same title.

        Title-based supersession is what stops memory growing without bound: an
        assumption refined three times leaves one active record and a readable
        chain of three, rather than three contradictory statements that a future
        prompt has to disambiguate.
        """
        now = self._clock.now()
        record.created_at = record.created_at or now
        record.updated_at = now

        with self.ledger.transaction() as conn:
            previous = conn.execute(
                """SELECT id FROM memory
                   WHERE kind = ? AND title = ? AND status = ? AND id != ?""",
                (record.kind, record.title, MemoryStatus.ACTIVE, record.id),
            ).fetchone()
            if previous:
                conn.execute(
                    "UPDATE memory SET status = ?, superseded_by = ?, updated_at = ? WHERE id = ?",
                    (MemoryStatus.SUPERSEDED, record.id, now, previous["id"]),
                )
                self.ledger.append(
                    Event(
                        type=EventType.MEMORY_SUPERSEDED,
                        node_id=node_id,
                        payload={"id": previous["id"], "by": record.id, "title": record.title},
                    ),
                    conn=conn,
                    apply_projection=False,
                )
            self._upsert(conn, record)
            self.ledger.append(
                Event(
                    type=EventType.MEMORY_WRITTEN,
                    node_id=node_id,
                    payload=record.to_dict(),
                ),
                conn=conn,
                apply_projection=False,
            )
        self._bump()
        log.debug("memory written", kind=record.kind, title=record.title[:60])
        return record

    def write_many(self, records: list[MemoryRecord], *, node_id: str | None = None) -> list[MemoryRecord]:
        return [self.write(r, node_id=node_id) for r in records]

    def _upsert(self, conn: sqlite3.Connection, record: MemoryRecord) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO memory
               (id, project_id, kind, title, body, tags, paths, confidence, status,
                source, data, created_at, updated_at, superseded_by)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.id,
                self.project_id,
                record.kind,
                record.title,
                record.body,
                json.dumps(record.tags),
                json.dumps(record.paths),
                record.confidence,
                record.status,
                record.source,
                json.dumps(record.data, default=str),
                record.created_at,
                record.updated_at,
                record.superseded_by,
            ),
        )

    def set_status(self, record_id: str, status: str, *, node_id: str | None = None) -> None:
        with self.ledger.transaction() as conn:
            conn.execute(
                "UPDATE memory SET status = ?, updated_at = ? WHERE id = ?",
                (status, self._clock.now(), record_id),
            )
            self.ledger.append(
                Event(
                    type=EventType.MEMORY_SUPERSEDED,
                    node_id=node_id,
                    payload={"id": record_id, "status": status},
                ),
                conn=conn,
                apply_projection=False,
            )
        self._bump()

    def resolve_finding(self, record_id: str, *, node_id: str | None = None) -> None:
        self.set_status(record_id, MemoryStatus.RESOLVED, node_id=node_id)

    # -- reading ---------------------------------------------------------

    def get(self, record_id: str) -> MemoryRecord | None:
        row = self.ledger.conn.execute("SELECT * FROM memory WHERE id = ?", (record_id,)).fetchone()
        return _from_row(row) if row else None

    def by_kind(self, kind: str, *, status: str = MemoryStatus.ACTIVE, limit: int = 200) -> list[MemoryRecord]:
        rows = self.ledger.conn.execute(
            "SELECT * FROM memory WHERE kind = ? AND status = ? ORDER BY updated_at DESC LIMIT ?",
            (kind, status, limit),
        )
        return [_from_row(row) for row in rows]

    def active(self, limit: int = 1000) -> list[MemoryRecord]:
        rows = self.ledger.conn.execute(
            "SELECT * FROM memory WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
            (MemoryStatus.ACTIVE, limit),
        )
        return [_from_row(row) for row in rows]

    def open_findings(self) -> list[MemoryRecord]:
        return self.by_kind(MemoryKind.FINDING)

    def counts(self) -> dict[str, int]:
        return {
            row["kind"]: row["c"]
            for row in self.ledger.conn.execute(
                "SELECT kind, COUNT(*) AS c FROM memory WHERE status = ? GROUP BY kind",
                (MemoryStatus.ACTIVE,),
            )
        }

    # -- retrieval -------------------------------------------------------

    def _bump(self) -> None:
        # Writes are durable ledger events.  Readers in this process no longer
        # need a separate in-memory generation counter: ``version`` observes
        # writes made by both this store and other Forge processes.
        with self._lock:
            self._index = None

    @property
    def version(self) -> int:
        """Return the durable memory revision visible across processes.

        Commands such as ``forge tell`` open their own MemoryStore while the
        daemon keeps running.  An in-memory counter therefore leaves the
        daemon's prompt digest and BM25 index stale.  The event sequence is the
        authoritative, monotonic revision and is cheap to query through the
        existing type/sequence index.
        """
        row = self.ledger.conn.execute(
            """SELECT COALESCE(MAX(seq), 0) AS revision
               FROM events WHERE type IN (?, ?)""",
            (EventType.MEMORY_WRITTEN, EventType.MEMORY_SUPERSEDED),
        ).fetchone()
        return int(row["revision"] if row is not None else 0)

    def _ensure_index(self) -> Index:
        version = self.version
        with self._lock:
            if self._index is None or self._index_version != version:
                index = Index()
                index.add_all(
                    [
                        Document(id=r.id, text=r.searchable(), weight=r.weight, meta={"kind": r.kind})
                        for r in self.active()
                    ]
                )
                self._index = index
                self._index_version = version
            return self._index

    def search(
        self,
        query: str,
        *,
        limit: int = 12,
        kinds: list[str] | None = None,
        paths: list[str] | None = None,
    ) -> list[MemoryRecord]:
        """Rank active records against a query.

        ``paths`` is the important refinement: when a node is about to edit
        ``src/render.ts``, records tagged with that path are boosted hard.
        Lexical similarity alone would rank a general convention above the
        interface contract for the exact file being changed.
        """
        index = self._ensure_index()
        hits = index.search(query, limit=limit * 3)
        by_id = {r.id: r for r in self.active()}

        scored: list[tuple[float, MemoryRecord]] = []
        path_set = {p for p in (paths or [])}
        for hit in hits:
            record = by_id.get(hit.doc.id)
            if record is None:
                continue
            if kinds and record.kind not in kinds:
                continue
            score = hit.score
            if path_set and path_set & set(record.paths):
                # A record explicitly about the file being edited is almost
                # always more relevant than a lexically similar one that is not,
                # so the boost has to be large enough to beat keyword density.
                score *= 3.5
            scored.append((score, record))

        # Always surface high-value kinds even when the query does not match
        # them lexically; an agent that never sees the project's conventions
        # will violate them, and no phrasing of the task would have retrieved
        # them.
        if not kinds:
            for kind in (MemoryKind.REQUIREMENT, MemoryKind.CONVENTION):
                for record in self.by_kind(kind, limit=5):
                    if all(record.id != r.id for _, r in scored):
                        scored.append((KIND_WEIGHTS.get(kind, 1.0) * 0.5, record))

        scored.sort(key=lambda pair: -pair[0])
        return [record for _, record in scored[:limit]]

    # -- maintenance -----------------------------------------------------

    def rebuild(self) -> int:
        """Recompute the memory projection from the event log."""
        count = 0
        with self.ledger.transaction() as conn:
            conn.execute("DELETE FROM memory")
            for row in conn.execute(
                "SELECT * FROM events WHERE type IN (?, ?) ORDER BY seq",
                (EventType.MEMORY_WRITTEN, EventType.MEMORY_SUPERSEDED),
            ):
                event = Event.from_row(row)
                if event.type == EventType.MEMORY_WRITTEN:
                    self._upsert(conn, MemoryRecord.from_dict(event.payload))
                    count += 1
                else:
                    payload = event.payload
                    conn.execute(
                        "UPDATE memory SET status = ?, superseded_by = ?, updated_at = ? WHERE id = ?",
                        (
                            payload.get("status", MemoryStatus.SUPERSEDED),
                            payload.get("by"),
                            event.ts,
                            payload.get("id"),
                        ),
                    )
        self._bump()
        return count

    def compact(self, *, keep_per_kind: int = 40) -> int:
        """Replace old low-value records with a digest.

        Runs when memory grows past what retrieval can rank usefully. Facts and
        findings age out first; decisions, requirements and interfaces never do,
        because they describe the system rather than a moment in its history.
        """
        compacted = 0
        for kind in (MemoryKind.FACT, MemoryKind.FINDING):
            records = self.by_kind(kind, limit=1000)
            if len(records) <= keep_per_kind:
                continue
            stale = records[keep_per_kind:]
            digest = MemoryRecord(
                kind=MemoryKind.DIGEST,
                title=f"Digest of {len(stale)} older {kind} records",
                body="\n".join(f"- {r.title}" for r in stale[:80]),
                confidence=0.6,
                source="compaction",
                tags=[kind, "digest"],
            )
            self.write(digest)
            for record in stale:
                self.set_status(record.id, MemoryStatus.SUPERSEDED)
                compacted += 1
        if compacted:
            log.info("memory compacted", records=compacted)
        return compacted

    def export_markdown(self) -> str:
        """Render memory as project documentation.

        Written to ``docs/`` in the project itself, so the artefact a human
        reads is generated from the same records the agents reason over. There
        is no second copy to drift.
        """
        sections = [
            (MemoryKind.REQUIREMENT, "Requirements"),
            (MemoryKind.ASSUMPTION, "Assumptions"),
            (MemoryKind.DECISION, "Architectural decisions"),
            (MemoryKind.INTERFACE, "Interfaces"),
            (MemoryKind.CONVENTION, "Conventions"),
            (MemoryKind.FACT, "Observed facts"),
            (MemoryKind.FINDING, "Open findings"),
        ]
        parts = ["# Project memory", "", "_Generated by Forge. Do not edit by hand._", ""]
        for kind, heading in sections:
            records = self.by_kind(kind, limit=200)
            if not records:
                continue
            parts += [f"## {heading}", ""]
            parts += [record.as_markdown() + "\n" for record in records]
        return "\n".join(parts)


def _from_row(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        kind=row["kind"],
        title=row["title"],
        body=row["body"],
        tags=json.loads(row["tags"]),
        paths=json.loads(row["paths"]),
        confidence=row["confidence"],
        status=row["status"],
        source=row["source"],
        data=json.loads(row["data"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        superseded_by=row["superseded_by"],
    )


# -- convenience constructors, used all over the agent layer ---------------


def assumption(title: str, body: str, *, confidence: float = 0.6, source: str = "planner", **kw: Any) -> MemoryRecord:
    return MemoryRecord(kind=MemoryKind.ASSUMPTION, title=title, body=body, confidence=confidence, source=source, **kw)


def decision(title: str, body: str, *, alternatives: list[str] | None = None, source: str = "architect", **kw: Any) -> MemoryRecord:
    data = kw.pop("data", {})
    if alternatives:
        data["alternatives"] = alternatives
    return MemoryRecord(kind=MemoryKind.DECISION, title=title, body=body, confidence=0.9, source=source, data=data, **kw)


def interface(title: str, body: str, *, paths: list[str] | None = None, **kw: Any) -> MemoryRecord:
    return MemoryRecord(kind=MemoryKind.INTERFACE, title=title, body=body, confidence=0.9, paths=paths or [], **kw)


def convention(title: str, body: str, **kw: Any) -> MemoryRecord:
    return MemoryRecord(kind=MemoryKind.CONVENTION, title=title, body=body, confidence=0.9, **kw)


def fact(title: str, body: str, *, source: str = "observation", **kw: Any) -> MemoryRecord:
    return MemoryRecord(kind=MemoryKind.FACT, title=title, body=body, confidence=1.0, source=source, **kw)


def finding(title: str, body: str, *, severity: str = "medium", paths: list[str] | None = None, **kw: Any) -> MemoryRecord:
    data = kw.pop("data", {})
    data["severity"] = severity
    return MemoryRecord(kind=MemoryKind.FINDING, title=title, body=body, data=data, paths=paths or [], **kw)


def requirement(title: str, body: str, *, source: str = "human", **kw: Any) -> MemoryRecord:
    return MemoryRecord(kind=MemoryKind.REQUIREMENT, title=title, body=body, confidence=1.0, source=source, **kw)
