"""Content-addressed response cache.

Autonomous runs repeat themselves more than one would like: a node retries after
an unrelated gate fails, a rollback replays work, two sibling nodes ask the same
question about the same file. Caching completions by the exact content of the
request turns those repeats into free instant answers.

The cache is *correct by construction* because the key covers everything that
could change the answer -- messages, tools, schema, model, temperature, output
limit. If any of those differ the key differs. Requests marked ``no_cache``
(deliberately sampled diversity) bypass it entirely.

Storage is one file per entry under a two-level fan-out directory, which keeps
directory sizes sane over years of use and lets an operator delete a subtree
without a database migration.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..obs.log import get_logger
from ..util.clock import Clock, default_clock
from ..util.hashing import content_hash
from .types import Completion, Request, ToolCall, Usage

log = get_logger("models.cache")

CACHE_VERSION = 1


def cache_key(request: Request, model: str) -> str:
    """Hash everything that can influence the response."""
    return content_hash(
        CACHE_VERSION,
        model,
        [
            {
                "role": m.role,
                "content": m.content,
                "tool_calls": [c.to_dict() for c in m.tool_calls],
                "tool_call_id": m.tool_call_id,
                "images": [i.data_b64[:64] + str(len(i.data_b64)) for i in m.images],
            }
            for m in request.messages
        ],
        [t.to_dict() for t in request.tools],
        request.schema,
        request.max_output_tokens,
        request.temperature,
        request.stop,
    )


class ResponseCache:
    def __init__(self, root: Path, *, ttl: float = 14 * 24 * 3600, clock: Clock | None = None, enabled: bool = True) -> None:
        self.root = Path(root)
        self.ttl = ttl
        self.enabled = enabled
        self._clock = clock or default_clock()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        if self.ttl and self._clock.now() - raw.get("stored_at", 0) > self.ttl:
            with self._lock:
                path.unlink(missing_ok=True)
            self.misses += 1
            return None
        self.hits += 1
        return _decode(raw["completion"])

    def put(self, key: str, completion: Completion) -> None:
        if not self.enabled or completion.cached:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stored_at": self._clock.now(), "completion": _encode(completion)}
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
            tmp.replace(path)  # atomic: a crash mid-write never yields a torn entry
        except OSError as exc:  # pragma: no cover - disk full etc.
            log.warn("cache write failed", error=str(exc))
            tmp.unlink(missing_ok=True)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "enabled": self.enabled,
        }

    def purge(self, older_than: float | None = None) -> int:
        """Delete expired entries. Returns the number removed."""
        if not self.enabled or not self.root.exists():
            return 0
        cutoff = self._clock.now() - (older_than if older_than is not None else self.ttl)
        removed = 0
        for path in self.root.rglob("*.json"):
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("stored_at", 0) < cutoff:
                    path.unlink()
                    removed += 1
            except (OSError, json.JSONDecodeError):  # pragma: no cover
                path.unlink(missing_ok=True)
                removed += 1
        return removed


def _encode(completion: Completion) -> dict[str, Any]:
    data = {
        "text": completion.text,
        "model": completion.model,
        "tier": completion.tier,
        "usage": asdict(completion.usage),
        "tool_calls": [c.to_dict() for c in completion.tool_calls],
        "finish_reason": completion.finish_reason,
        "latency": completion.latency,
        "cost": completion.cost,
    }
    return data


def _decode(data: dict[str, Any]) -> Completion:
    return Completion(
        text=data["text"],
        model=data["model"],
        tier=data["tier"],
        usage=Usage(**data.get("usage", {})),
        tool_calls=[ToolCall(**c) for c in data.get("tool_calls", [])],
        finish_reason=data.get("finish_reason", "stop"),
        latency=data.get("latency", 0.0),
        # A cache hit costs nothing. Reporting the original cost here would
        # double-count spend across retries and corrupt the budget.
        cost=0.0,
        cached=True,
    )
