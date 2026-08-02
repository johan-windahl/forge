"""Time-sortable identifiers.

Forge uses ULID-style ids everywhere: 48 bits of millisecond timestamp followed
by 80 bits of randomness, Crockford base32 encoded. Two properties matter:

* **Lexicographic order equals chronological order.** Ledger keys, artifact
  filenames and node ids all sort into causal order without a separate index.
* **Generated without coordination.** Parallel workers -- and future distributed
  workers -- mint ids without touching the database.

A process-local monotonic guard ensures ids minted inside the same millisecond
still sort correctly, so a burst of events never reorders itself.
"""

from __future__ import annotations

import os
import threading
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_LOCK = threading.Lock()
_last_ms = 0
_last_rand = 0

_TIME_BITS = 48
_RAND_BITS = 80
_RAND_MAX = (1 << _RAND_BITS) - 1


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def ulid(ts_ms: int | None = None) -> str:
    """Return a new 26-character ULID.

    ``ts_ms`` is exposed only for tests and replay tooling; production callers
    should let it default to the wall clock.
    """
    global _last_ms, _last_rand
    with _LOCK:
        now = ts_ms if ts_ms is not None else int(time.time() * 1000)
        if now == _last_ms:
            # Same millisecond: increment the random component instead of
            # drawing a fresh one, preserving intra-millisecond ordering.
            _last_rand = (_last_rand + 1) & _RAND_MAX
            if _last_rand == 0:  # pragma: no cover - astronomically unlikely
                now = _last_ms + 1
                _last_ms = now
                _last_rand = int.from_bytes(os.urandom(10), "big")
        else:
            if now < _last_ms:
                # Clock moved backwards (NTP step). Never emit a smaller id.
                now = _last_ms
                _last_rand = (_last_rand + 1) & _RAND_MAX
            else:
                _last_ms = now
                _last_rand = int.from_bytes(os.urandom(10), "big")
        return _encode(now, 10) + _encode(_last_rand, 16)


def timestamp_of(uid: str) -> float:
    """Extract the embedded creation time (epoch seconds) from a ULID."""
    value = 0
    for ch in uid[:10]:
        idx = _CROCKFORD.find(ch.upper())
        if idx < 0:
            raise ValueError(f"not a ULID: {uid!r}")
        value = (value << 5) | idx
    return value / 1000.0


def short(uid: str, length: int = 8) -> str:
    """A human-readable abbreviation, stable for the lifetime of the id."""
    return uid[-length:]


def new_id(prefix: str) -> str:
    """A namespaced id such as ``node_01J2X...``. Prefixes aid log grepping."""
    return f"{prefix}_{ulid()}"
