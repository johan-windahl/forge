"""Content addressing.

Forge caches aggressively -- model responses, gate verdicts, retrieval results.
Every cache key is a hash of *canonical* content, so equality is structural
rather than incidental. ``canonical_json`` guarantees that two logically equal
payloads hash identically regardless of key order or float formatting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace, UTF-8 safe."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_fallback)


def _fallback(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(*parts: Any) -> str:
    """Hash an ordered sequence of arbitrary values into one hex digest.

    Parts are length-prefixed before hashing so that ``("ab", "c")`` and
    ``("a", "bc")`` produce different digests.
    """
    h = hashlib.sha256()
    for part in parts:
        blob = part if isinstance(part, bytes) else canonical_json(part).encode("utf-8")
        h.update(len(blob).to_bytes(8, "big"))
        h.update(blob)
    return h.hexdigest()


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def tree_hash(root: Path, include: Iterable[str] | None = None, exclude: Iterable[str] | None = None) -> str:
    """Hash a directory tree by (relative path, content).

    Used to key gate-result caches: if the tree hash has not changed, a gate
    that passed before will pass again, and we skip the run entirely. This is
    the single largest saving in a long autonomous run, where the same test
    suite would otherwise be re-executed after every unrelated edit.
    """
    exclude_set = set(exclude or (".git", "node_modules", "__pycache__", ".venv", "dist", "build", ".forge"))
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in exclude_set for part in rel.parts):
            continue
        if include is not None and not any(rel.match(pattern) for pattern in include):
            continue
        entries.append((rel.as_posix(), file_hash(path)))
    return content_hash(entries)


def stable_key(*parts: Any, length: int = 32) -> str:
    """A truncated content hash suitable for filenames and cache keys."""
    return content_hash(*parts)[:length]
