"""A self-contained HTML dashboard.

Written to ``.forge/reports/index.html`` after every run and on demand. It is a
single file with inlined CSS and no external requests, so it can be opened from
a file path, copied to another machine, or served by any static host -- which
matters when the box running a three-day build is a headless server reachable
only over SSH.

Screenshots are embedded as data URIs. A build's visual output is the part a
human most wants to see and the part hardest to get at over a terminal, so it is
worth the file size.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from ..util.clock import human_duration, iso

_CSS = """
:root {
  --bg: #0f1115; --panel: #171a21; --line: #262b36; --text: #e6e9ef;
  --muted: #9aa4b8; --ok: #3fb950; --warn: #d29922; --bad: #f85149;
  --accent: #58a6ff;
}
@media (prefers-color-scheme: light) {
  :root { --bg:#f6f7f9; --panel:#fff; --line:#e2e5ea; --text:#1c2128; --muted:#5a6474; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:32px 20px 64px; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:15px; text-transform:uppercase; letter-spacing:.07em;
  color:var(--muted); margin:32px 0 12px; font-weight:600; }
.goal { color:var(--muted); margin:0 0 24px; max-width:70ch; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.card .label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
.card .value { font-size:22px; font-weight:600; margin-top:4px; }
.card .sub { color:var(--muted); font-size:12px; margin-top:2px; }
.progress { height:10px; background:var(--line); border-radius:5px; overflow:hidden; margin:6px 0 2px; }
.progress > div { height:100%; background:var(--accent); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; color:var(--muted); font-weight:600; font-size:11px;
  text-transform:uppercase; letter-spacing:.05em; padding:8px 10px; border-bottom:1px solid var(--line); }
td { padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
tr:last-child td { border-bottom:none; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.pill { display:inline-block; padding:1px 8px; border-radius:99px; font-size:11px; font-weight:600; }
.s-succeeded { background:rgba(63,185,80,.15); color:var(--ok); }
.s-running { background:rgba(88,166,255,.15); color:var(--accent); }
.s-ready,.s-pending,.s-deferred { background:rgba(154,164,184,.15); color:var(--muted); }
.s-failed,.s-cancelled { background:rgba(248,81,73,.15); color:var(--bad); }
.s-blocked { background:rgba(210,153,34,.18); color:var(--warn); }
.blocked { background:rgba(210,153,34,.08); border:1px solid var(--warn);
  border-radius:10px; padding:14px 16px; margin-bottom:12px; }
.blocked pre { white-space:pre-wrap; margin:8px 0 0; color:var(--muted); font-size:12.5px; }
.shots { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }
.shot img { width:100%; border-radius:8px; border:1px solid var(--line); display:block; }
.shot span { color:var(--muted); font-size:11.5px; display:block; margin-top:5px; word-break:break-all; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
.muted { color:var(--muted); }
footer { margin-top:48px; color:var(--muted); font-size:12px; }
"""


def write_dashboard(
    path: Path,
    *,
    status: dict[str, Any],
    nodes: list[Any],
    events: list[Any],
    metrics: Any = None,
    screenshots: list[Path] | None = None,
) -> Path:
    """Render the dashboard to ``path`` and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    project = status.get("project") or {}
    budget = status.get("budget", {})
    counts = status.get("counts", {})
    progress = float(status.get("progress", 0.0))

    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Forge -- {html.escape(str(project.get('name', 'project')))}</title>",
        f"<style>{_CSS}</style></head><body><div class='wrap'>",
        f"<h1>{html.escape(str(project.get('name', 'Forge project')))}</h1>",
        f"<p class='goal'>{html.escape(str(project.get('goal', '')))}</p>",
    ]

    state = "Stalled" if status.get("stalled") else ("Idle" if status.get("quiescent") else "Working")
    stats = status.get("stats", {})
    parts.append("<div class='grid'>")
    parts.append(
        _card(
            "Progress",
            f"{progress:.0%}",
            f"<div class='progress'><div style='width:{progress * 100:.1f}%'></div></div>{state}",
            raw_sub=True,
        )
    )
    parts.append(
        _card(
            "Nodes",
            str(sum(counts.values())),
            f"{counts.get('succeeded', 0)} done &middot; {counts.get('blocked', 0)} blocked",
            raw_sub=True,
        )
    )
    limit = (budget.get("limits") or {}).get("total", 0)
    parts.append(
        _card(
            "Spend",
            f"{budget.get('total', 0):.3f}",
            f"of {limit:g} budget" if limit else "no limit",
        )
    )
    parts.append(
        _card(
            "Cloud share",
            f"{budget.get('cloud_fraction', 0):.1%}",
            f"target {(budget.get('limits') or {}).get('cloud_fraction_target', 0):.0%}",
        )
    )
    if stats.get("started_at"):
        parts.append(
            _card("This run", str(stats.get("nodes_completed", 0)) + " nodes", f"{stats.get('attempts', 0)} attempts")
        )
    parts.append("</div>")

    blocked = status.get("blocked") or []
    if blocked:
        parts.append("<h2>Needs attention</h2>")
        for item in blocked:
            parts.append(
                f"<div class='blocked'><strong>{html.escape(item['title'])}</strong>"
                f"<pre>{html.escape(str(item.get('question', '')))}</pre></div>"
            )

    parts.append("<h2>Task graph</h2><div class='panel'><table>")
    parts.append("<tr><th>Status</th><th>Kind</th><th>Title</th><th>Try</th><th>Cost</th><th>Milestone</th></tr>")
    for node in nodes[:250]:
        parts.append(
            "<tr>"
            f"<td><span class='pill s-{html.escape(node.status)}'>{html.escape(node.status)}</span></td>"
            f"<td class='mono'>{html.escape(node.kind)}</td>"
            f"<td>{html.escape(node.title[:110])}</td>"
            f"<td class='muted'>{node.attempts}</td>"
            f"<td class='mono'>{node.cost:.4f}</td>"
            f"<td class='muted'>{html.escape(node.milestone or '')}</td>"
            "</tr>"
        )
    parts.append("</table></div>")

    by_model = budget.get("by_model") or []
    if by_model:
        parts.append("<h2>Model usage</h2><div class='panel'><table>")
        parts.append("<tr><th>Model</th><th>Tier</th><th>Calls</th><th>Input</th><th>Output</th><th>Cached</th><th>Cost</th></tr>")
        for row in by_model:
            parts.append(
                "<tr>"
                f"<td class='mono'>{html.escape(str(row['model']))}</td>"
                f"<td class='muted'>{html.escape(str(row['tier']))}</td>"
                f"<td>{row['calls']}</td>"
                f"<td class='mono'>{row['input_tokens']:,}</td>"
                f"<td class='mono'>{row['output_tokens']:,}</td>"
                f"<td class='mono muted'>{row['cached_tokens']:,}</td>"
                f"<td class='mono'>{row['cost']:.4f}</td>"
                "</tr>"
            )
        parts.append("</table></div>")

    if metrics is not None:
        parts.append("<h2>Measured outcomes</h2><div class='panel' style='padding:14px 16px'>")
        parts.append(f"<pre class='mono muted' style='white-space:pre-wrap;margin:0'>{html.escape(metrics.render())}</pre>")
        parts.append("</div>")

    shots = [p for p in (screenshots or []) if p.is_file()][:12]
    if shots:
        parts.append("<h2>Latest screenshots</h2><div class='shots'>")
        for shot in shots:
            try:
                encoded = base64.b64encode(shot.read_bytes()).decode("ascii")
                captured_at = iso(shot.stat().st_mtime)
            except OSError:  # pragma: no cover
                continue
            parts.append(
                f"<div class='shot'><img src='data:image/png;base64,{encoded}' alt='{html.escape(shot.name)}'>"
                f"<span>{html.escape(shot.name)} &middot; captured {captured_at}</span></div>"
            )
        parts.append("</div>")

    parts.append("<h2>Recent activity</h2><div class='panel'><table>")
    for event in reversed(events[-60:]):
        parts.append(
            "<tr>"
            f"<td class='muted mono' style='white-space:nowrap'>{iso(event.ts)}</td>"
            f"<td class='mono'>{html.escape(event.type)}</td>"
            f"<td class='muted'>{html.escape(_summarise(event))}</td>"
            "</tr>"
        )
    parts.append("</table></div>")

    duration = ""
    if stats.get("started_at"):
        duration = f" &middot; run duration {human_duration(max(0.0, events[-1].ts - stats['started_at']))}" if events else ""
    parts.append(
        f"<footer>Generated by Forge at {iso()}{duration}. "
        "This file is self-contained and safe to copy or share.</footer>"
    )
    parts.append("</div></body></html>")

    path.write_text("".join(parts), encoding="utf-8")
    return path


def _card(label: str, value: str, sub: str = "", *, raw_sub: bool = False) -> str:
    sub_html = sub if raw_sub else html.escape(sub)
    return (
        f"<div class='card'><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div>"
        f"<div class='sub'>{sub_html}</div></div>"
    )


def _summarise(event: Any) -> str:
    payload = event.payload or {}
    for key in ("title", "summary", "gate", "milestone", "reason", "error", "label"):
        if payload.get(key):
            return str(payload[key])[:120]
    if payload:
        rendered = json.dumps(payload, default=str)
        return rendered[:110] + ("…" if len(rendered) > 110 else "")
    return ""


def collect_screenshots(artifacts_dir: Path, limit: int = 12) -> list[Path]:
    """Most recent screenshots across all nodes."""
    if not artifacts_dir.is_dir():
        return []
    shots = [p for p in artifacts_dir.rglob("*.png") if not p.name.startswith("diff_")]
    shots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return shots[:limit]
