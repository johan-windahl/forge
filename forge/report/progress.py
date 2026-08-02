"""Terminal rendering of run state.

The operator of a multi-day autonomous run asks three questions, in this order:
is it still working, is it going well, and does it need me. Everything here is
arranged to answer those in the first few lines, before any detail.

Blocked nodes are printed with their questions in full. A blocked node is the
only thing in the system that requires a human, so it gets the most space.
"""

from __future__ import annotations

from typing import Any

from ..kernel.graph import NodeStatus
from ..util.clock import human_duration, iso

_BAR_FILLED = "█"
_BAR_EMPTY = "░"

#: Fallback for how long a run may make no observable progress before the status
#: block says so. Only used when the run does not report a ``quiet_threshold`` of
#: its own -- that one is derived from the slowest rung's timeout, because how
#: long silence may legitimately last depends entirely on how slow the model is.
_QUIET_WARNING = 900.0

_STATUS_ORDER = [
    NodeStatus.RUNNING,
    NodeStatus.READY,
    NodeStatus.PENDING,
    NodeStatus.DEFERRED,
    NodeStatus.SUCCEEDED,
    NodeStatus.FAILED,
    NodeStatus.BLOCKED,
    NodeStatus.CANCELLED,
]


def bar(fraction: float, width: int = 28) -> str:
    filled = max(0, min(width, round(fraction * width)))
    return _BAR_FILLED * filled + _BAR_EMPTY * (width - filled)


def render_status(status: dict[str, Any], *, verbose: bool = False) -> str:
    project = status.get("project") or {}
    counts: dict[str, int] = status.get("counts", {})
    budget = status.get("budget", {})
    lines: list[str] = []

    lines.append(f"Project   {project.get('name', '(none)')}")
    if project.get("goal"):
        lines.append(f"Goal      {_wrap(project['goal'], 68, indent=10)}")

    progress = float(status.get("progress", 0.0))
    headline = "quiescent" if status.get("quiescent") else "working"
    if status.get("stalled"):
        headline = "STALLED -- needs attention"
    lines.append("")
    lines.append(f"Progress  {bar(progress)} {progress:.0%}   [{headline}]")
    if status.get("stub_runs"):
        lines.append(
            "          ^ includes work from "
            f"{status['stub_runs']} dry run(s) on the echo stub -- not real output."
        )

    run = status.get("run")
    if run:
        lines.append(f"Run       {_run_line(run)}")

    quiet = float(status.get("quiet_for", 0.0))
    # The run knows its own slowest rung; fall back only when it did not say.
    threshold = float(status.get("quiet_threshold") or _QUIET_WARNING)
    if quiet >= threshold:
        # A wedged worker keeps the heartbeat going and the counts unchanged, so
        # without this the display is identical to healthy work.
        lines.append(
            f"          ^ nothing has happened for {human_duration(quiet)} -- "
            "a node may be stuck (forge node <id>, forge stop)"
        )

    active = ", ".join(
        f"{counts.get(str(state), 0)} {state}"
        for state in _STATUS_ORDER
        if counts.get(str(state), 0)
    )
    lines.append(f"Nodes     {active or 'none'}")

    spent = budget.get("total", 0.0)
    limit = (budget.get("limits") or {}).get("total", 0.0)
    cloud = budget.get("cloud_fraction", 0.0)
    target = (budget.get("limits") or {}).get("cloud_fraction_target", 0.0)
    lines.append(
        f"Budget    {spent:.4f} of {limit:.2f} spent"
        + (f" ({spent / limit:.0%})" if limit else "")
        + f" | cloud tokens {cloud:.1%} (target {target:.0%})"
    )
    cache = status.get("cache", {})
    if cache.get("hits") or cache.get("misses"):
        lines.append(
            f"Cache     {cache.get('hits', 0)} hit / {cache.get('misses', 0)} miss "
            f"({cache.get('hit_rate', 0):.0%})"
        )

    blocked = status.get("blocked") or []
    if blocked:
        lines.append("")
        lines.append(f"NEEDS ATTENTION -- {len(blocked)} blocked node(s):")
        for item in blocked:
            # The id, because the guidance below says to run `forge unblock
            # <node>` and this is the only place the operator is shown one.
            lines.append(f"  * [{_short_id(item.get('id', ''))}] {item['title']}")
            question = (item.get("question") or "").strip()
            for line in question.splitlines():
                lines.append(f"      {line}")
            lines.append("")

    critical = status.get("critical_path") or []
    if critical:
        lines.append("Critical path:")
        for item in critical:
            marker = {"succeeded": "+", "running": ">", "blocked": "!", "failed": "x"}.get(item["status"], "-")
            lines.append(f"  {marker} {item['title'][:76]}")

    if verbose:
        memory = status.get("memory", {})
        if memory:
            lines.append("")
            lines.append("Memory:   " + ", ".join(f"{v} {k}" for k, v in sorted(memory.items())))
        lessons = status.get("lessons", {})
        if lessons.get("count"):
            lines.append(
                f"Lessons:  {lessons['count']} stored, {lessons.get('established', 0)} established"
            )
        toolchain = status.get("toolchain", {})
        if toolchain.get("languages"):
            lines.append(f"Stack:    {', '.join(toolchain['languages'])}")
        by_model = budget.get("by_model") or []
        if by_model:
            lines.append("")
            lines.append("Spend by model:")
            for row in by_model:
                lines.append(
                    f"  {row['model']:<16} {row['calls']:>4} calls  "
                    f"{row['input_tokens']:>9,} in  {row['output_tokens']:>8,} out  "
                    f"{row['cost']:>9.4f}"
                )

    stats = status.get("stats", {})
    if stats.get("started_at"):
        lines.append("")
        lines.append(
            f"This run: {stats.get('nodes_completed', 0)} completed, "
            f"{stats.get('attempts', 0)} attempt(s), "
            f"{stats.get('nodes_blocked', 0)} blocked"
        )
    return "\n".join(lines)


def _run_line(run: dict[str, Any]) -> str:
    """Whether a process is actually building right now.

    Distinct from the ``[working]`` marker above it, which only says the graph has
    work left. Since runs detach, the two came apart -- a run that was killed
    leaves a graph full of pending nodes and looks, from the counts alone,
    perfectly healthy. This line is the one that answers "is it still working".
    """
    state = run.get("state", "stopped")
    if state == "live":
        return (
            f"active, pid {run.get('pid', '?')}, up {human_duration(run.get('uptime', 0.0))}"
            "  (forge watch / forge stop)"
        )
    if state == "starting":
        return f"starting, pid {run.get('pid', '?')}"
    if state == "crashed":
        return "NOT RUNNING -- the last run died without stopping cleanly. `forge run` resumes it."
    return "not running. `forge run` to start."


def render_timeline(events: list[Any], *, limit: int = 40) -> str:
    """A readable trace of recent activity.

    Filtered to the events a person would care about. The full log is always in
    ``.forge/logs`` and in the ledger; this is the view that fits on a screen.
    """
    interesting = {
        "node.started": ">",
        "node.succeeded": "+",
        "node.failed": "x",
        "node.blocked": "!",
        "node.escalated": "^",
        "model.request": ">",
        "patch.applied": "+",
        "gate.failed": "x",
        "milestone.reached": "*",
        "checkpoint.created": ".",
        "rollback.performed": "<",
        "lesson.learned": "~",
        "budget.warning": "$",
        "usage.report": "$",
        "deploy.succeeded": "^",
        "deploy.failed": "x",
    }
    lines: list[str] = []
    for event in events[-limit:]:
        marker = interesting.get(event.type)
        if marker is None:
            continue
        detail = _event_detail(event)
        lines.append(f"{iso(event.ts)}  {marker}  {event.type:<20} {detail}")
        lines.extend(_failure_evidence(event))
    return "\n".join(lines) or "(no notable events yet)"


#: Lines of gate output to show under a failure in the timeline. Enough to name
#: the cause -- a compiler's first few errors -- without turning a watch stream
#: into a build log.
_EVIDENCE_LINES = 6
_EVIDENCE_INDENT = " " * 28


def _failure_evidence(event: Any) -> list[str]:
    """Why a gate failed, under the line saying that it did.

    "`npm run test` failed with exit 1" tells an operator nothing they could act
    on, and the answer was already in the payload. Finding out that a source file
    was zero bytes meant leaving the tool and running tsc by hand.
    """
    if event.type not in ("gate.failed", "deploy.failed", "node.failed"):
        return []
    evidence = str((event.payload or {}).get("evidence") or "").strip()
    if not evidence:
        return []
    shown = [line.rstrip() for line in evidence.splitlines() if line.strip()][:_EVIDENCE_LINES]
    out = [f"{_EVIDENCE_INDENT}{line[:120]}" for line in shown]
    hidden = len([x for x in evidence.splitlines() if x.strip()]) - len(shown)
    if hidden > 0:
        out.append(f"{_EVIDENCE_INDENT}... {hidden} more line(s), `forge node <id>` for the rest")
    return out


def _event_detail(event: Any) -> str:
    payload = event.payload or {}
    if event.type == "usage.report":
        return _usage_detail(payload)
    for key in ("title", "summary", "gate", "milestone", "label", "reason", "error"):
        if payload.get(key):
            return str(payload[key])[:90]
    if event.node_id:
        return f"node {event.node_id[-8:]}"
    return ""


def _usage_detail(payload: dict) -> str:
    """One line of per-model token usage for the watch stream.

    Shows the *window*, not the running total: `forge status` already answers
    "what has this cost", and what an operator watching a live run wants to know
    is whether the last few minutes went local or cloud.
    """
    active = [m for m in payload.get("models", []) if m.get("calls")]
    if not active:
        return (
            "no completed model calls in the last "
            f"{int(payload.get('window_seconds', 0)) // 60}m"
        )
    parts = [
        f"{m['model']} {_thousands(m['input_tokens'])}in/{_thousands(m['output_tokens'])}out"
        f"x{m['calls']}"
        for m in active
    ]
    window = int(payload.get("window_seconds", 0)) // 60
    cloud = payload.get("cloud_fraction", 0.0)
    return f"last {window}m: " + "  ".join(parts) + f"  | cloud {cloud:.0%} cumulative"


def _thousands(value: int) -> str:
    value = int(value or 0)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


def _short_id(node_id: str) -> str:
    """The last 8 characters, matching the timeline and accepted by the CLI.

    ULIDs sort by prefix, so the tail is the part that differs between sibling
    nodes created in the same millisecond -- and it is what `_resolve_node`
    matches on, so anything printed here can be pasted straight back.
    """
    return (node_id or "")[-8:]


def render_nodes(nodes: list[Any], *, show_all: bool = False) -> str:
    """A table of the task graph."""
    if not nodes:
        return "(no nodes)"
    rows = [n for n in nodes if show_all or n.status != NodeStatus.SUCCEEDED]
    if not rows:
        return f"All {len(nodes)} nodes succeeded."
    # The short id is shown because every command that acts on a node takes one,
    # and `forge status` tells the operator to run `forge unblock <node>` --
    # advice that was impossible to follow from any listing.
    lines = [f"{'ID':<10} {'STATUS':<10} {'KIND':<14} {'TRY':<4} {'COST':>8}  TITLE"]
    for node in rows:
        lines.append(
            f"{_short_id(node.id):<10} {node.status:<10} {node.kind:<14} "
            f"{node.attempts:<4} {node.cost:>8.4f}  {node.title[:52]}"
        )
    hidden = len(nodes) - len(rows)
    if hidden:
        lines.append(f"({hidden} completed node(s) hidden; use --all to show)")
    return "\n".join(lines)


def render_metrics(metrics: Any) -> str:
    return metrics.render()


def _wrap(text: str, width: int, indent: int = 0) -> str:
    import textwrap

    wrapped = textwrap.wrap(text, width=width)
    if not wrapped:
        return ""
    pad = " " * indent
    return ("\n" + pad).join(wrapped)


def format_duration(seconds: float) -> str:
    return human_duration(seconds)
