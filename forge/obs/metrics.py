"""Metrics collection backed by the ledger.

Metrics are not a separate telemetry system. Every counter and timing is written
as an event to the same append-only ledger that holds the project's history,
because the retrospective analyser needs to correlate "this milestone used
420k cloud tokens" with "these three nodes failed their gates twice". Two stores
would mean two clocks and two truths.

In-process aggregates exist only as a write-behind buffer so a hot loop is not
one SQLite transaction per increment.
"""

from __future__ import annotations

import contextlib
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..util.clock import Clock, default_clock


class MetricSink(Protocol):
    """Anything that can durably accept a metric sample."""

    def record_metric(self, name: str, value: float, kind: str, labels: dict[str, str]) -> None: ...


@dataclass(slots=True)
class Aggregate:
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float, keep_samples: int = 256) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.samples.append(value)
        if len(self.samples) > keep_samples * 2:
            # Reservoir-free decimation: keep the most recent window. Forge
            # cares about recent behaviour (is the model getting slower now?),
            # not lifetime distribution.
            del self.samples[:-keep_samples]

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def quantile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[idx]

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "total": round(self.total, 4),
            "mean": round(self.mean, 4),
            "min": round(self.minimum, 4) if self.count else 0.0,
            "max": round(self.maximum, 4) if self.count else 0.0,
            "p50": round(self.quantile(0.5), 4),
            "p95": round(self.quantile(0.95), 4),
        }


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{rendered}}}"


class Metrics:
    """Thread-safe metric registry with an optional durable sink."""

    def __init__(self, sink: MetricSink | None = None, clock: Clock | None = None) -> None:
        self._sink = sink
        self._clock = clock or default_clock()
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, Aggregate] = defaultdict(Aggregate)

    def attach(self, sink: MetricSink) -> None:
        """Bind a durable sink after construction (the ledger opens later)."""
        self._sink = sink

    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[_key(name, labels)] += value
        self._forward(name, value, "counter", labels)

    def gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges[_key(name, labels)] = value
        self._forward(name, value, "gauge", labels)

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._histograms[_key(name, labels)].observe(value)
        self._forward(name, value, "histogram", labels)

    @contextmanager
    def timer(self, name: str, **labels: str) -> Iterator[None]:
        """Time a block; records seconds into a histogram even if it raises."""
        start = self._clock.monotonic()
        try:
            yield
        finally:
            self.observe(name, self._clock.monotonic() - start, **labels)

    def _forward(self, name: str, value: float, kind: str, labels: dict[str, str]) -> None:
        if self._sink is None:
            return
        # Metrics must never break work.
        with contextlib.suppress(Exception):  # pragma: no cover
            self._sink.record_metric(name, value, kind, labels)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: v.to_dict() for k, v in self._histograms.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
