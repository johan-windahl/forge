"""Deterministic analysis of a run.

Every number here comes from the event ledger. None of it is estimated,
summarised or inferred by a model, which means the retrospective's premises are
facts and only its conclusions are judgement.

The metrics were chosen to answer questions the platform can act on:

* *Are we spending cloud tokens well?* -- escalation rate and cost by task class.
  A class that escalates constantly should start higher; one that never does
  should start lower.
* *Are we doing work twice?* -- rework ratio: nodes that ran more than once, and
  commits later reverted or rewritten. High rework means planning or context is
  wrong, not that the implementer is bad.
* *Where does the wall clock actually go?* -- almost always gates, and almost
  always one gate. Knowing which one is worth more than any prompt tuning.
* *Which gates are lying?* -- a gate that fails then passes with no intervening
  change is flaky, and a flaky gate poisons the router's success statistics.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..kernel.events import EventType
from ..kernel.graph import NodeStatus, TaskGraph
from ..kernel.ledger import Ledger
from ..util.clock import human_duration


@dataclass(slots=True)
class MilestoneMetrics:
    """Everything measurable about a slice of the run."""

    milestone: str = ""
    nodes_total: int = 0
    nodes_succeeded: int = 0
    nodes_failed: int = 0
    nodes_blocked: int = 0
    #: Nodes that took more than one attempt.
    nodes_retried: int = 0
    total_attempts: int = 0
    wall_clock: float = 0.0
    cost_total: float = 0.0
    cost_by_tier: dict[str, float] = field(default_factory=dict)
    tokens_local: int = 0
    tokens_cloud: int = 0
    escalations: int = 0
    model_calls: int = 0
    cache_hits: int = 0
    schema_violations: int = 0
    gate_time: dict[str, float] = field(default_factory=dict)
    gate_failures: dict[str, int] = field(default_factory=dict)
    gate_cache_hits: int = 0
    flaky_gates: list[str] = field(default_factory=list)
    slowest_nodes: list[dict[str, Any]] = field(default_factory=list)
    costliest_nodes: list[dict[str, Any]] = field(default_factory=list)
    by_task_class: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def cloud_fraction(self) -> float:
        total = self.tokens_cloud + self.tokens_local
        return self.tokens_cloud / total if total else 0.0

    @property
    def rework_ratio(self) -> float:
        """Attempts per completed node, minus one. Zero means no rework."""
        if not self.nodes_succeeded:
            return 0.0
        return max(0.0, self.total_attempts / self.nodes_succeeded - 1.0)

    @property
    def success_rate(self) -> float:
        closed = self.nodes_succeeded + self.nodes_failed + self.nodes_blocked
        return self.nodes_succeeded / closed if closed else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "milestone": self.milestone,
            "nodes": {
                "total": self.nodes_total,
                "succeeded": self.nodes_succeeded,
                "failed": self.nodes_failed,
                "blocked": self.nodes_blocked,
                "retried": self.nodes_retried,
                "success_rate": round(self.success_rate, 3),
                "rework_ratio": round(self.rework_ratio, 3),
            },
            "wall_clock": round(self.wall_clock, 1),
            "wall_clock_human": human_duration(self.wall_clock),
            "cost": {
                "total": round(self.cost_total, 4),
                "by_tier": {k: round(v, 4) for k, v in self.cost_by_tier.items()},
            },
            "tokens": {
                "local": self.tokens_local,
                "cloud": self.tokens_cloud,
                "cloud_fraction": round(self.cloud_fraction, 4),
            },
            "model": {
                "calls": self.model_calls,
                "cache_hits": self.cache_hits,
                "escalations": self.escalations,
                "schema_violations": self.schema_violations,
            },
            "gates": {
                "time": {k: round(v, 1) for k, v in sorted(self.gate_time.items(), key=lambda p: -p[1])},
                "failures": self.gate_failures,
                "cache_hits": self.gate_cache_hits,
                "flaky": self.flaky_gates,
            },
            "slowest_nodes": self.slowest_nodes,
            "costliest_nodes": self.costliest_nodes,
            "by_task_class": self.by_task_class,
        }

    def render(self) -> str:
        """A compact rendering for the retrospective prompt.

        Deliberately dense: this is what a model reads to form conclusions, and
        every token spent on formatting is one not spent on reasoning.
        """
        lines = [
            f"Milestone: {self.milestone or 'whole run'}",
            f"Nodes: {self.nodes_succeeded} succeeded, {self.nodes_failed} failed, "
            f"{self.nodes_blocked} blocked, {self.nodes_retried} needed more than one attempt",
            f"Rework ratio: {self.rework_ratio:.2f} extra attempts per completed node",
            f"Wall clock: {human_duration(self.wall_clock)}",
            f"Cost: {self.cost_total:.4f} total ({', '.join(f'{k}={v:.4f}' for k, v in self.cost_by_tier.items())})",
            f"Tokens: {self.tokens_local:,} local, {self.tokens_cloud:,} cloud "
            f"({self.cloud_fraction:.1%} cloud)",
            f"Model calls: {self.model_calls} ({self.cache_hits} served from cache, "
            f"{self.escalations} escalations, {self.schema_violations} schema violations)",
        ]
        if self.gate_time:
            ranked = sorted(self.gate_time.items(), key=lambda pair: -pair[1])[:6]
            lines.append("Gate time: " + ", ".join(f"{name} {seconds:.0f}s" for name, seconds in ranked))
        if self.gate_failures:
            lines.append(
                "Gate failures: "
                + ", ".join(f"{name} x{count}" for name, count in sorted(self.gate_failures.items(), key=lambda p: -p[1]))
            )
        if self.flaky_gates:
            lines.append(f"Gates that failed then passed without a fix: {', '.join(self.flaky_gates)}")
        if self.by_task_class:
            lines.append("By task class:")
            for name, data in sorted(self.by_task_class.items(), key=lambda p: -p[1].get("cost", 0)):
                lines.append(
                    f"  {name}: {data['calls']} call(s), cost {data['cost']:.4f}, "
                    f"{data['escalations']} escalation(s)"
                )
        if self.slowest_nodes:
            lines.append("Slowest nodes:")
            lines += [f"  {n['title'][:70]} -- {human_duration(n['duration'])}" for n in self.slowest_nodes[:5]]
        if self.costliest_nodes:
            lines.append("Costliest nodes:")
            lines += [f"  {n['title'][:70]} -- {n['cost']:.4f}" for n in self.costliest_nodes[:5]]
        return "\n".join(lines)


def compute_metrics(
    ledger: Ledger,
    graph: TaskGraph,
    *,
    milestone: str | None = None,
    since_seq: int = 0,
) -> MilestoneMetrics:
    """Derive metrics from the ledger for a milestone, or for the whole run."""
    metrics = MilestoneMetrics(milestone=milestone or "")

    nodes = [n for n in graph.all_nodes() if milestone is None or n.milestone == milestone]
    node_ids = {n.id for n in nodes}
    metrics.nodes_total = len(nodes)
    for node in nodes:
        match node.status:
            case NodeStatus.SUCCEEDED:
                metrics.nodes_succeeded += 1
            case NodeStatus.FAILED:
                metrics.nodes_failed += 1
            case NodeStatus.BLOCKED:
                metrics.nodes_blocked += 1
        metrics.total_attempts += node.attempts
        if node.attempts > 1:
            metrics.nodes_retried += 1
        if node.started_at and node.finished_at:
            metrics.wall_clock += node.finished_at - node.started_at

    durations = [
        {"id": n.id, "title": n.title, "duration": (n.finished_at or 0) - (n.started_at or 0), "kind": n.kind}
        for n in nodes
        if n.started_at and n.finished_at
    ]
    durations.sort(key=lambda item: -item["duration"])
    metrics.slowest_nodes = durations[:8]

    costs = [{"id": n.id, "title": n.title, "cost": n.cost, "kind": n.kind} for n in nodes if n.cost > 0]
    costs.sort(key=lambda item: -item["cost"])
    metrics.costliest_nodes = costs[:8]

    # -- spend -----------------------------------------------------------
    by_class: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "cost": 0.0, "escalations": 0, "tokens": 0})
    rows = ledger.conn.execute("SELECT * FROM spend").fetchall()
    for row in rows:
        if node_ids and row["node_id"] not in node_ids and milestone is not None:
            continue
        tier = row["tier"]
        cost = float(row["cost"])
        tokens = int(row["input_tokens"]) + int(row["output_tokens"])
        metrics.cost_total += cost
        metrics.cost_by_tier[tier] = metrics.cost_by_tier.get(tier, 0.0) + cost
        if row["hosted"] == "local":
            metrics.tokens_local += tokens
        else:
            metrics.tokens_cloud += tokens
        metrics.model_calls += 1
        if row["escalation"]:
            metrics.escalations += 1
        task_class = row["task_class"] or "unknown"
        entry = by_class[task_class]
        entry["calls"] += 1
        entry["cost"] += cost
        entry["tokens"] += tokens
        if row["escalation"]:
            entry["escalations"] += 1
    metrics.by_task_class = {
        name: {**data, "cost": round(data["cost"], 5)} for name, data in by_class.items()
    }

    # -- events ----------------------------------------------------------
    gate_history: dict[str, list[str]] = defaultdict(list)
    gate_failure_counts: Counter[str] = Counter()
    for event in ledger.read(after_seq=since_seq):
        if milestone is not None and event.node_id and event.node_id not in node_ids:
            continue
        match event.type:
            case EventType.MODEL_CACHE_HIT:
                metrics.cache_hits += 1
            case EventType.GATE_PASSED:
                gate = event.payload.get("gate", "?")
                metrics.gate_time[gate] = metrics.gate_time.get(gate, 0.0) + float(event.payload.get("duration", 0))
                gate_history[gate].append("pass")
                if event.payload.get("cached"):
                    metrics.gate_cache_hits += 1
            case EventType.GATE_FAILED | EventType.GATE_ERRORED:
                gate = event.payload.get("gate", "?")
                metrics.gate_time[gate] = metrics.gate_time.get(gate, 0.0) + float(event.payload.get("duration", 0))
                gate_failure_counts[gate] += 1
                gate_history[gate].append("fail")
    metrics.gate_failures = dict(gate_failure_counts)

    # A gate that alternates fail/pass repeatedly is flaky. Flaky gates are
    # actively harmful: they train the router to think a capable model is
    # failing, and they trigger debug nodes for bugs that do not exist.
    for gate, history in gate_history.items():
        flips = sum(1 for a, b in itertools.pairwise(history) if a != b)
        if len(history) >= 4 and flips >= len(history) * 0.5:
            metrics.flaky_gates.append(gate)

    schema_row = ledger.conn.execute(
        "SELECT COALESCE(SUM(count), 0) AS c FROM metrics WHERE name = 'model.schema_violation'"
    ).fetchone()
    metrics.schema_violations = int(schema_row["c"])

    return metrics


def compare(previous: MilestoneMetrics, current: MilestoneMetrics) -> dict[str, Any]:
    """Milestone-over-milestone deltas.

    Direction matters more than magnitude: the useful question is whether the
    platform is getting better or worse at building this project, and a single
    milestone's absolute numbers cannot answer it.
    """

    def delta(a: float, b: float) -> dict[str, Any]:
        change = b - a
        pct = (change / a * 100) if a else None
        return {"from": round(a, 4), "to": round(b, 4), "change": round(change, 4),
                "percent": round(pct, 1) if pct is not None else None}

    return {
        "cost": delta(previous.cost_total, current.cost_total),
        "cloud_fraction": delta(previous.cloud_fraction, current.cloud_fraction),
        "rework_ratio": delta(previous.rework_ratio, current.rework_ratio),
        "success_rate": delta(previous.success_rate, current.success_rate),
        "wall_clock": delta(previous.wall_clock, current.wall_clock),
    }
