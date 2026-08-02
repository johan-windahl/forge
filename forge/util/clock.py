"""Time access, funnelled through one seam.

Every timestamp and every sleep in Forge goes through this module. That single
seam is what makes the scheduler testable: a test can install a
:class:`ManualClock`, advance it by an hour, and observe lease expiry and
backoff without waiting an hour.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime


class Clock:
    """Wall-clock time and real sleeps."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class ManualClock(Clock):
    """Deterministic clock for tests. ``sleep`` advances time instantly."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._t = start
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def monotonic(self) -> float:
        with self._lock:
            return self._t

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._t += max(0.0, seconds)


_default = Clock()


def default_clock() -> Clock:
    return _default


def iso(ts: float | None = None) -> str:
    """RFC 3339 / ISO 8601 UTC string, second precision, ``Z`` suffix."""
    dt = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_ms(ts: float | None = None) -> str:
    """ISO 8601 UTC with millisecond precision, for log lines."""
    dt = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def human_duration(seconds: float) -> str:
    """``4212`` -> ``1h 10m``. Used in progress reports."""
    seconds = max(0.0, seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"
