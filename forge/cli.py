"""The ``forge`` command line.

Designed around what an operator actually does over a long project:

    forge init "Build a browser-based Quake-inspired FPS with one polished level."
    forge run                 # start (detached), or resume after a crash
    forge status              # is it working, is it going well, does it need me
    forge watch               # follow along
    forge stop                # wind down the active run
    forge report              # open the dashboard

``run`` detaches because a build lasts days and a terminal does not. The run owns
a pidfile in the state directory, which is what makes "is it running" answerable
and what stops two orchestrators from racing on one ledger.

Everything else -- rollback, policy inspection, memory queries, repair -- exists
for the days when something has gone wrong, and is written so that the recovery
path is a single obvious command rather than a research project.

Every command is safe to run against a live project: reads never take the write
lock for longer than a query, and no command mutates the workspace without
saying so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    Config,
    attempts_needed_for_ladder,
    load_config,
    write_default_config,
)
from .errors import ForgeError
from .obs.log import get_logger, setup_logging

log = get_logger("cli")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _config(args: argparse.Namespace, *, create: bool = True) -> Config:
    """Load configuration for a command.

    ``create=False`` is for read-only commands. Materialising ``.forge`` --
    ledger, cache, logs -- merely because someone ran ``forge status`` in the
    wrong directory leaves a decoy project behind that later looks real.
    """
    overrides: dict[str, Any] = {}
    if getattr(args, "workers", None):
        overrides.setdefault("scheduler", {})["workers"] = args.workers
    if getattr(args, "budget", None) is not None:
        overrides.setdefault("budget", {})["total_cost"] = args.budget
    if getattr(args, "sandbox", None):
        overrides.setdefault("sandbox", {})["kind"] = args.sandbox
    config = load_config(args.dir, overrides=overrides or None)
    if not create and not config.forge_dir.exists():
        print(
            f"error: no Forge project in {config.project_dir}.\n"
            "       Run `forge init \"<description>\"` here, or `cd` to the project.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    config.ensure_dirs()
    setup_logging(
        path=config.log_path,
        console=not getattr(args, "quiet", False),
        console_level="debug" if getattr(args, "verbose", False) else config.log_level,
    )
    return config


def _orchestrator(config: Config, *, dry_run: bool = False):
    from .kernel.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)
    if dry_run:
        _install_echo(orchestrator)
    return orchestrator


def _install_echo(orchestrator: Any) -> None:
    """Replace every provider with the deterministic stub.

    ``--dry-run`` exercises the entire orchestration path -- graph, gates, git,
    checkpoints, recovery -- without a token of spend. It is the fastest way to
    verify a configuration change on a real project, and the reason the echo
    provider is a first-class part of the model layer rather than a test fixture.
    """
    from .config import ProviderConfig
    from .models.provider import EchoProvider

    for name in list(orchestrator.config.models.providers):
        provider = EchoProvider(name, ProviderConfig(kind="echo", base_url="", api_key_env=""))
        provider.stub = True
        orchestrator.models.registry.install(name, provider)
    orchestrator.dry_run = True
    log.warn("dry run: all model calls are served by the echo provider")


def _print(data: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data if isinstance(data, str) else json.dumps(data, indent=2, default=str))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    from .workspace.references import ReferenceError, ReferenceStore

    config = _config(args)
    interactive = _can_prompt(args)

    goal = " ".join(args.goal or []).strip()
    if not goal and interactive:
        goal = _ask_goal()
    if not goal:
        print("error: provide a project description", file=sys.stderr)
        return 2

    config_path = config.forge_dir / "config.toml"
    if not config_path.exists() and not args.no_config:
        write_default_config(config_path)

    store = ReferenceStore(config.forge_dir)
    pending = [_split_reference(raw) for raw in (args.reference or [])]
    if interactive and not pending:
        pending = _ask_references()

    added = []
    for source, description in pending:
        try:
            ref = store.add(source, description=description)
        except ReferenceError as exc:
            # One bad URL must not discard a goal the operator just typed.
            print(f"  skipped {source}: {exc.message}", file=sys.stderr)
            continue
        added.append(ref)

    with _orchestrator(config) as orchestrator:
        project = orchestrator.create_project(goal, name=args.name or "")
        print(f"Created project '{project.name}' in {config.project_dir}")
        print(f"  workspace: {config.workspace_dir}")
        print(f"  state:     {config.forge_dir}")
        if added:
            print(f"  references: {len(added)} in {store.root}")
            for ref in added:
                print(f"    [{ref.role}] {ref.label()}")
        print()
        print("Next: `forge run` to start building. It will keep going unattended.")
        if not added:
            print("Tip: `forge reference add <url|path>` supplies material to build against.")
    return 0


def _can_prompt(args: argparse.Namespace) -> bool:
    """Only prompt a human who is actually there.

    `forge init` runs in CI and in scripts as often as at a terminal. Blocking
    on input in either would hang a pipeline with no indication why.
    """
    if getattr(args, "no_input", False):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def _ask_goal() -> str:
    print("What should Forge build? One or two sentences is enough; it plans the rest.")
    try:
        return input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def _ask_references() -> list[tuple[str, str]]:
    """Collect reference material, one item at a time, until an empty line.

    Asking for the description separately rather than parsing it out of one line
    is deliberate: the description is the part operators skip, and a prompt that
    names it gets an answer where a suggestion in help text does not.
    """
    print()
    print("Reference material to build against? A URL or a local path:")
    print("images, video, audio, documents, example files. Blank line when done.")
    out: list[tuple[str, str]] = []
    while True:
        try:
            source = input("reference> ").strip()
            if not source:
                return out
            description = input("  what should Forge take from it? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return out
        out.append((source, description))


def _split_reference(raw: str) -> tuple[str, str]:
    """Parse ``source`` or ``source::description`` from a non-interactive flag."""
    source, sep, description = raw.partition("::")
    return source.strip(), description.strip() if sep else ""


def cmd_reference(args: argparse.Namespace) -> int:
    from .workspace.references import ReferenceError, ReferenceStore

    config = _config(args, create=False)
    store = ReferenceStore(config.forge_dir)

    if args.reference_command == "list":
        refs = store.load()
        if not refs:
            print(f"No references in {store.root}")
            return 0
        for ref in refs:
            marker = " (derived)" if ref.is_derived else ""
            print(f"[{ref.role}]{marker} {ref.label()}")
            if ref.source:
                print(f"    from {ref.source}")
        return 0

    added = []
    for raw in args.source:
        source, inline = _split_reference(raw)
        try:
            added.append(store.add(source, description=args.describe or inline, role=args.role or ""))
        except ReferenceError as exc:
            print(f"error: {exc.message}", file=sys.stderr)
            return 1
    for ref in added:
        print(f"Added [{ref.role}] {ref.label()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Start or resume the build.

    Detached by default. A run lasts hours; attaching it to a terminal means an
    SSH drop or a closed laptop kills work that is thirty nodes deep. The
    foreground form is still there for CI and for `--dry-run`, where the whole
    point is to watch it happen.
    """
    from .kernel import daemon

    config = _config(args)
    try:
        token = daemon.claim(
            config.forge_dir, token=os.environ.get(_CLAIM_ENV, ""), argv=list(sys.argv[1:])
        )
    except daemon.AlreadyRunning as exc:
        return _report_already_running(config, exc.handle)

    # A dry run exists to be watched: it takes seconds, spends nothing, and its
    # whole output is the trace. Detaching it would send that to a file and leave
    # a pidfile behind that blocks the real run.
    if args.foreground or args.dry_run:
        try:
            daemon.mark_running(config.forge_dir, token)
            return _run_here(config, args)
        finally:
            daemon.release(config.forge_dir, token)

    try:
        return _run_detached(config, args, token)
    except BaseException:
        daemon.release(config.forge_dir, token)
        raise


def _run_here(config: Config, args: argparse.Namespace) -> int:
    """Run the orchestrator in this process."""
    with _orchestrator(config, dry_run=args.dry_run) as orchestrator:
        if orchestrator.project is None:
            print("error: no project here. Run `forge init \"<description>\"` first.", file=sys.stderr)
            return 2
        summary = orchestrator.run(max_nodes=args.max_nodes, until_quiescent=not args.forever)
        _write_report(config, orchestrator)

        from .report.progress import render_status

        print()
        print(render_status(summary, verbose=args.verbose))
        if summary.get("stalled"):
            return 3
    return 0


#: Passed to the detached child so it adopts the claim made on its behalf rather
#: than deciding a run is already active and refusing to start.
_CLAIM_ENV = "FORGE_RUN_CLAIM"


def _run_detached(config: Config, args: argparse.Namespace, token: str) -> int:
    from .kernel import daemon

    if not config.ledger_path.exists():
        print("error: no project here. Run `forge init \"<description>\"` first.", file=sys.stderr)
        daemon.release(config.forge_dir, token)
        return 2

    argv = _child_argv(args)
    env = dict(os.environ)
    env[_CLAIM_ENV] = token
    handle = daemon.spawn(argv, forge_dir=config.forge_dir, cwd=Path.cwd(), token=token, env=env)

    # A misconfigured project dies in the first second. Without this wait the
    # operator is told "started, pid 1234" about a process that is already gone,
    # and finds out only on the next command.
    if not daemon.await_start(handle, timeout=2.5):
        print("error: the run exited immediately. Last lines of its log:", file=sys.stderr)
        for line in _log_tail(Path(handle.log), lines=15):
            print(f"  {line}", file=sys.stderr)
        print(f"\nfull log: {handle.log}", file=sys.stderr)
        daemon.release(config.forge_dir, token)
        return 1

    print(f"Run started in the background (pid {handle.pid}).")
    print(f"  log     {handle.log}")
    print("  follow  forge watch")
    print("  check   forge status")
    print("  stop    forge stop")
    if args.follow:
        print()
        return _follow(config, interval=3.0)
    return 0


def _report_already_running(config: Config, handle: Any) -> int:
    """A second `forge run` is a normal human action, not an error.

    Exit 0 because the postcondition the operator asked for -- a run is active --
    already holds. Two orchestrators on one ledger would race on node leases and
    the workspace, so starting another is the one thing that must not happen.
    """
    from .util.clock import human_duration

    what = "starting" if handle.starting else f"running for {human_duration(handle.age())}"
    print(f"A run is already active (pid {handle.pid}, {what}).")
    if handle.log:
        print(f"  log     {handle.log}")
    print("  follow  forge watch")
    print("  stop    forge stop")
    return 0


def _child_argv(args: argparse.Namespace) -> list[str]:
    """The command line for the detached child: ours again, in the foreground.

    Replaying the real argv rather than rebuilding it from the parsed namespace
    means a flag added to `forge run` next year is passed through without anyone
    remembering to update this function.
    """
    raw = list(getattr(args, "argv", None) or [])
    if not raw:  # programmatic invocation, no argv to replay
        raw = ["run"]
        for flag, value in (
            ("--workers", args.workers),
            ("--budget", args.budget),
            ("--max-nodes", args.max_nodes),
            ("--sandbox", args.sandbox),
        ):
            if value is not None:
                raw += [flag, str(value)]
        for flag, on in (("--forever", args.forever), ("--dry-run", args.dry_run)):
            if on:
                raw.append(flag)
        if args.dir not in (".", ""):
            raw = ["--dir", str(args.dir), *raw]
    return [*_self_argv(), *raw, "--foreground"]


def _self_argv() -> list[str]:
    """How to invoke this same Forge again.

    The console script when there is one, so `ps` shows something recognisable;
    otherwise `-m forge` on the interpreter currently running, which is correct
    inside a venv even when its bin directory is not on PATH.
    """
    exe = Path(sys.argv[0] or "")
    if exe.name in {"forge", "forge.exe"} and exe.exists():
        return [str(exe.resolve())]
    return [sys.executable, "-m", "forge"]


def _log_tail(path: Path, *, lines: int = 15) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ["(no log written)"]
    tail = [line for line in content.splitlines() if line.strip()][-lines:]
    return tail or ["(log is empty)"]


def cmd_stop(args: argparse.Namespace) -> int:
    """Wind down the active run.

    SIGTERM, not SIGKILL: the orchestrator catches it, lets in-flight nodes
    finish, writes a checkpoint and records `run.stopped`. That is what makes the
    next `forge run` resume cleanly instead of recovering.
    """
    from .kernel import daemon

    config = _config(args, create=False)
    handle = daemon.active_run(config.forge_dir)
    if handle is None:
        print("No run is active.")
        return 0

    print(f"Stopping pid {handle.pid} (in-flight nodes finish first)...")
    outcome = daemon.stop(handle, timeout=args.timeout, kill=args.kill)
    if outcome in ("stopped", "gone"):
        daemon.release(config.forge_dir, handle.token)
        print("Run stopped.")
        return 0
    if outcome == "killed":
        daemon.release(config.forge_dir, handle.token)
        print(
            "Run killed. It did not shut down cleanly, so the last node's work is\n"
            "unrecorded -- `forge run` will recover it from the ledger."
        )
        return 0
    print(
        f"Still winding down after {args.timeout:.0f}s. A node can hold a lease for\n"
        f"{config.scheduler.lease_seconds / 60:.0f} minutes, so this is normal.\n"
        "  watch it finish   forge watch\n"
        "  force it          forge stop --kill  (loses the current node's work)"
    )
    return 4


def cmd_status(args: argparse.Namespace) -> int:
    config = _config(args, create=False)
    with _orchestrator(config) as orchestrator:
        if orchestrator.project is None:
            print("No project in this directory.", file=sys.stderr)
            return 2
        status = orchestrator.status()
        # "working" in the status block means the graph has work left, not that
        # anything is doing it. Since runs detach, those two came apart: a dead
        # run with pending nodes reads as healthy. Say which it is.
        status["run"] = _run_summary(config, orchestrator.ledger)
        if args.json:
            _print(status, True)
            return 0
        from .report.progress import render_status

        print(render_status(status, verbose=args.verbose))
    return 0


def _run_summary(config: Config, ledger: Any) -> dict[str, Any]:
    """Process-level facts about the active run, for the status block."""
    from .kernel import daemon

    handle = daemon.active_run(config.forge_dir)
    if handle is None:
        return {"state": _liveness(config, ledger)}
    return {
        "state": "starting" if handle.starting else "live",
        "pid": handle.pid,
        "uptime": handle.age(),
        "log": handle.log,
    }


def cmd_watch(args: argparse.Namespace) -> int:
    """Follow a running build.

    Reads the ledger rather than attaching to the process, so it works against a
    run started in a different terminal, under systemd, or on another day.
    """
    config = _config(args, create=False)
    return _follow(
        config,
        interval=args.interval,
        show_status=args.status,
        show_code=not args.no_code,
    )


def _follow(
    config: Config,
    *,
    interval: float,
    show_status: bool = False,
    show_code: bool = True,
) -> int:
    """Tail the ledger until interrupted. Shared by `watch` and `run --follow`."""
    import time

    from .kernel.ledger import Ledger
    from .report.progress import render_status, render_timeline

    ledger = Ledger(config.ledger_path)
    cursor = max(0, ledger.head_seq() - 30)
    code_seen: dict[tuple[str, str], str] = {}
    state = _liveness(config, ledger)
    print(_liveness_line(state), flush=True)
    try:
        while True:
            events = ledger.read(after_seq=cursor)
            if events:
                cursor = events[-1].seq
                timeline = render_timeline(events, limit=len(events))
                if timeline.strip() and timeline != "(no notable events yet)":
                    print(timeline, flush=True)
            if show_code:
                code = _render_live_code(
                    config,
                    _running_node_ids(ledger),
                    code_seen,
                )
                if code:
                    print(code, flush=True)
            # Checked every tick, not just when events arrive: a run that dies
            # silently produces no event, and that is exactly the case where the
            # operator most needs to be told. Without this, `watch` on a dead run
            # is indistinguishable from a hang.
            now = _liveness(config, ledger)
            if now != state:
                state = now
                print(_liveness_line(state), flush=True)
            if show_status:
                with _orchestrator(config) as orchestrator:
                    print(render_status(orchestrator.status()), flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    finally:
        ledger.close()


_WATCH_CODE_MAX_BYTES = 200_000
_WATCH_CODE_IGNORES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
}


def _running_node_ids(ledger: Any) -> list[str]:
    """Node ids whose isolated worktrees may currently be changing."""
    rows = ledger.conn.execute(
        "SELECT id FROM nodes WHERE status = 'running' ORDER BY started_at, id"
    )
    return [str(row["id"]) for row in rows]


def _render_live_code(
    config: Config,
    node_ids: list[str],
    seen: dict[tuple[str, str], str],
) -> str:
    """Render newly written worktree files once per content version.

    The ledger only learns about a patch after validation and integration. A
    local model can spend an hour editing before that event, which made
    ``forge watch`` look idle and hid the most useful thing to inspect: the code
    being produced. Persistent node worktrees are already Forge's source of
    truth during an attempt, so watch them read-only and print changed text
    files as soon as their contents settle between polling ticks.
    """
    import hashlib

    from .workspace.git import Repo

    rendered: list[str] = []
    active: set[tuple[str, str]] = set()
    for node_id in node_ids:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "-" for c in node_id)
        root = config.worktrees_dir / safe_id
        if not (root / ".git").exists():
            continue
        try:
            entries = Repo(root).status()
        except ForgeError:
            continue
        expanded: list[tuple[str, str]] = []
        for status, raw_path in entries:
            path = raw_path.rsplit(" -> ", 1)[-1]
            candidate = root / path
            # Porcelain status collapses an entirely untracked directory to
            # ``?? src/``. Expand it or watch would print a fictitious deleted
            # directory and omit every new source file inside it.
            if status == "??" and candidate.is_dir():
                expanded.extend(
                    (status, str(child.relative_to(root)))
                    for child in sorted(candidate.rglob("*"))
                    if child.is_file()
                )
            else:
                expanded.append((status, path))
        for status, path in expanded:
            if (
                path in _WATCH_CODE_IGNORES
                or path == ".forge"
                or path.startswith((".forge/", "node_modules/"))
            ):
                continue
            key = (node_id, path)
            active.add(key)
            candidate = (root / path).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                continue
            if not candidate.is_file():
                fingerprint = "<deleted>"
                body = "[deleted]"
            else:
                try:
                    raw = candidate.read_bytes()
                except OSError:
                    continue
                if b"\0" in raw[:8192]:
                    continue
                fingerprint = hashlib.sha256(raw).hexdigest()
                clipped = raw[:_WATCH_CODE_MAX_BYTES]
                body = clipped.decode("utf-8", errors="replace")
                if len(raw) > len(clipped):
                    body += (
                        f"\n... [file truncated at {_WATCH_CODE_MAX_BYTES} bytes; "
                        f"{len(raw)} bytes total] ..."
                    )
            if seen.get(key) == fingerprint:
                continue
            seen[key] = fingerprint
            rendered.extend(
                [
                    f"=== {_short_node_id(node_id)} wrote {path} [{status or 'changed'}] ===",
                    body.rstrip(),
                    f"=== end {path} ===",
                ]
            )

    # A reset makes a path clean. Forget its old fingerprint so writing the
    # same bytes during a later attempt is still visible.
    for key in list(seen):
        if key not in active:
            seen.pop(key, None)
    return "\n".join(rendered)


def _short_node_id(node_id: str) -> str:
    return (node_id or "")[-8:]


def _liveness(config: Config, ledger: Any) -> str:
    """`live`, `crashed` or `stopped`.

    Two sources, because either alone lies. The pidfile knows whether a process
    exists; the ledger knows whether it shut down on purpose. A process that
    vanished without recording `run.stopped` is the case worth naming out loud --
    it is what a crash, an OOM kill or a `kill -9` looks like from outside.
    """
    from .kernel import daemon
    from .kernel.events import EventType

    if daemon.active_run(config.forge_dir) is not None:
        return "live"
    events = ledger.read(types=[EventType.RUN_STARTED, EventType.RUN_STOPPED])
    if events and events[-1].type == EventType.RUN_STARTED:
        return "crashed"
    return "stopped"


def _liveness_line(state: str) -> str:
    if state == "live":
        return "-- a run is active; following the ledger. Ctrl-C to detach (the run keeps going). --"
    if state == "crashed":
        return (
            "-- the last run's process is gone but it never recorded a clean stop: "
            "it crashed or was killed. `forge run` resumes from the ledger. --"
        )
    return (
        "-- no run is active. Nothing new will appear here until you start one "
        "(`forge run`). Ctrl-C to exit. --"
    )


def cmd_report(args: argparse.Namespace) -> int:
    config = _config(args)
    with _orchestrator(config) as orchestrator:
        path = _write_report(config, orchestrator)
        print(f"Dashboard written to {path}")
        if args.open:
            import webbrowser

            webbrowser.open(path.as_uri())
    return 0


def _write_report(config: Config, orchestrator: Any) -> Path:
    from .improve.metrics import compute_metrics
    from .report.dashboard import collect_screenshots, write_dashboard

    metrics = compute_metrics(orchestrator.ledger, orchestrator.graph)
    return write_dashboard(
        config.reports_dir / "index.html",
        status=orchestrator.status(),
        nodes=orchestrator.graph.all_nodes(),
        events=orchestrator.ledger.tail(200),
        metrics=metrics,
        screenshots=collect_screenshots(config.artifacts_dir),
    )


def cmd_nodes(args: argparse.Namespace) -> int:
    config = _config(args, create=False)
    with _orchestrator(config) as orchestrator:
        nodes = orchestrator.graph.all_nodes(status=args.status or None)
        if args.json:
            _print([n.to_dict() for n in nodes], True)
            return 0
        from .report.progress import render_nodes

        print(render_nodes(nodes, show_all=args.all))
    return 0


def cmd_node(args: argparse.Namespace) -> int:
    config = _config(args, create=False)
    with _orchestrator(config) as orchestrator:
        node = orchestrator.graph.try_get(args.node_id) or _resolve_node(orchestrator, args.node_id)
        if node is None:
            print(f"error: no node matching {args.node_id!r}", file=sys.stderr)
            return 2
        if args.json:
            _print(node.to_dict(), True)
            return 0
        print(f"{node.id}\n  {node.kind}: {node.title}")
        print(f"  status: {node.status}   attempts: {node.attempts}   tier: {node.tier}   cost: {node.cost:.4f}")
        if node.deps:
            print(f"  depends on: {', '.join(node.deps)}")
        if node.acceptance:
            print("  acceptance:")
            for item in node.acceptance:
                print(f"    - {item}")
        if node.result:
            print("  result:")
            print("    " + json.dumps(node.result, indent=2, default=str)[:2000].replace("\n", "\n    "))
        events = orchestrator.ledger.read(node_id=node.id)
        _print_gate_failures(events)

        # Routine bookkeeping crowds out everything worth reading: this node's
        # last 25 events were 25 lease renewals, which says only that a worker
        # was alive. Hide them and say how many were hidden.
        interesting = [e for e in events if e.type not in _ROUTINE_EVENTS]
        hidden = len(events) - len(interesting)
        print(f"\n  {len(events)} event(s)" + (f", {hidden} routine hidden:" if hidden else ":"))
        for event in interesting[-25:]:
            print(f"    {event.type:<24} {json.dumps(event.payload, default=str)[:110]}")
    return 0


#: Events that say a worker is alive rather than that anything happened. Plain
#: strings, not EventType: this module imports that lazily to keep `forge --help`
#: fast, and EventType is a StrEnum so the comparison holds either way.
_ROUTINE_EVENTS = frozenset({"node.lease_renewed", "run.heartbeat", "usage.report"})


def _print_gate_failures(events: list[Any], *, limit: int = 4) -> None:
    """The reason the node is stuck, in full, before the event list.

    A gate failure is almost always the answer to "why is this node blocked",
    and the event list truncates each payload to a line -- which is shorter than
    a single compiler error. Diagnosing a zero-byte source file meant leaving
    Forge and running tsc by hand; the evidence was there, just never shown.
    """
    failures = [e for e in events if e.type == "gate.failed"]
    if not failures:
        return
    last_run = failures[-limit:]
    print(f"\n  last gate failures ({len(failures)} in this node's history):")
    for event in last_run:
        payload = event.payload
        print(f"    {payload.get('gate', '?')}: {payload.get('summary', '')}")
        evidence = str(payload.get("evidence") or "").strip()
        for line in evidence.splitlines()[:12]:
            print(f"      {line[:160]}")


def _resolve_node(orchestrator: Any, prefix: str) -> Any:
    matches = [n for n in orchestrator.graph.all_nodes() if n.id.endswith(prefix) or n.id.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


def cmd_cancel(args: argparse.Namespace) -> int:
    config = _config(args)
    with _orchestrator(config) as orchestrator:
        node = orchestrator.graph.try_get(args.node_id) or _resolve_node(orchestrator, args.node_id)
        if node is None:
            print(f"error: no node matching {args.node_id!r}", file=sys.stderr)
            return 2
        orchestrator.graph.cancel(node.id, reason=args.reason or "cancelled by operator", actor="human")
        print(f"Cancelled {node.id}: {node.title}")
    return 0


def cmd_unblock(args: argparse.Namespace) -> int:
    """Answer a blocked node's question and return it to the queue.

    The answer is written into the node's spec and into project memory as a
    human-sourced requirement, so it informs every later prompt rather than only
    the retry.
    """
    config = _config(args)
    from .memory.store import requirement

    with _orchestrator(config) as orchestrator:
        node = orchestrator.graph.try_get(args.node_id) or _resolve_node(orchestrator, args.node_id)
        if node is None:
            print(f"error: no node matching {args.node_id!r}", file=sys.stderr)
            return 2
        answer = " ".join(args.answer).strip()
        if answer:
            orchestrator.memory.write(
                requirement(
                    f"Operator guidance for '{node.title[:60]}'",
                    answer,
                    source="human",
                    tags=["guidance"],
                ),
                node_id=node.id,
            )
            spec = dict(node.spec)
            spec["operator_guidance"] = answer
            orchestrator.graph.update(node.id, spec=spec, actor="human")
        # New operator information starts a new local-first strategy cycle.
        # Keeping the previous cloud tier meant restarting or unblocking a
        # Sonnet node immediately resumed cloud authorship, so the new
        # OpenCode/local worker never got to apply the guidance.  The
        # convergence markers describe the old information state and must be
        # cleared for the same reason; structural decomposition metadata stays.
        spec = dict(orchestrator.graph.get(node.id).spec)
        for key in ("_failure_signature", "_same_failure_count", "_strategies_tried"):
            spec.pop(key, None)

        # A fresh retry budget, not one last chance. The attempt counter is
        # compared against `max_attempts` for the life of the node, so leaving it
        # at 14 meant the guidance just written got a single attempt before the
        # node re-blocked. Answering a blocked node is new information; the
        # retries it buys should be the same as any other node's.
        orchestrator.graph.update(
            node.id,
            status="ready",
            attempts=0,
            tier=config.models.default,
            spec=spec,
            not_before=0.0,
            actor="human",
        )
        print(
            f"Unblocked {node.id} at {config.models.default!r} with a fresh retry budget "
            f"({config.scheduler.max_attempts} attempts). It will be retried on the next run."
        )
    return 0


def cmd_tell(args: argparse.Namespace) -> int:
    """Add guidance to a running project without waiting for it to get stuck.

    Writes a human-sourced requirement into project memory, which reaches every
    subsequent agent prompt through the normal retrieval path. This is the
    supported way to steer qualities no gate can measure -- how something should
    feel, which trade-off to prefer, a number to tune toward -- while the build
    is in flight.
    """
    config = _config(args)
    from .memory.store import convention, requirement

    text = " ".join(args.guidance).strip()
    if not text:
        print("error: provide the guidance to record", file=sys.stderr)
        return 2

    with _orchestrator(config) as orchestrator:
        if orchestrator.project is None:
            print("error: no project here. Run `forge init \"<description>\"` first.", file=sys.stderr)
            return 2
        title = args.title or f"Operator guidance: {text[:60]}"
        record = (convention if args.convention else requirement)(title, text, source="human")
        record.tags = [*record.tags, "guidance"]
        orchestrator.memory.write(record)
        kind = "convention" if args.convention else "requirement"
        print(f"Recorded as a {kind}. It will reach every agent prompt from now on.")
        print(f"  {title}")
    return 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    config = _config(args, create=False)
    with _orchestrator(config) as orchestrator:
        checkpoints = orchestrator.checkpoints.list(limit=args.limit, kind=args.kind)
        if args.json:
            _print([c.to_dict() for c in checkpoints], True)
            return 0
        if not checkpoints:
            print("(no checkpoints)")
            return 0
        print(f"{'ID':<18} {'KIND':<12} {'COMMIT':<10} WHEN                  LABEL")
        for checkpoint in checkpoints:
            from .util.clock import iso

            print(
                f"{checkpoint.id:<18} {checkpoint.kind:<12} {checkpoint.commit[:8]:<10} "
                f"{iso(checkpoint.created_at):<21} {checkpoint.label[:50]}"
            )
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    config = _config(args)
    with _orchestrator(config) as orchestrator:
        target = args.checkpoint
        if not target:
            latest = orchestrator.checkpoints.latest(kind="milestone")
            if latest is None:
                print("error: no milestone checkpoint to roll back to", file=sys.stderr)
                return 2
            target = latest.id
        checkpoint = orchestrator.checkpoints.get(target)
        if not args.yes:
            print(f"This will reset the workspace to: {checkpoint.label}")
            print(f"  commit {checkpoint.commit[:10]} from {checkpoint.created_at}")
            print("Uncommitted changes will be lost. Re-run with --yes to proceed.")
            return 1
        orchestrator.checkpoints.rollback(checkpoint.id, reason=args.reason or "operator rollback")
        print(f"Rolled back to {checkpoint.id} ({checkpoint.label})")
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    config = _config(args, create=False)
    with _orchestrator(config) as orchestrator:
        if args.export:
            text = orchestrator.memory.export_markdown()
            Path(args.export).write_text(text, encoding="utf-8")
            print(f"Exported project memory to {args.export}")
            return 0
        if args.query:
            records = orchestrator.memory.search(" ".join(args.query), limit=args.limit)
        elif args.kind:
            records = orchestrator.memory.by_kind(args.kind, limit=args.limit)
        else:
            records = orchestrator.memory.active(limit=args.limit)
        if args.json:
            _print([r.to_dict() for r in records], True)
            return 0
        for record in records:
            print(record.render(verbose=args.verbose))
            print()
        if not records:
            print("(no matching records)")
    return 0


def cmd_lessons(args: argparse.Namespace) -> int:
    config = _config(args, create=False)
    from .memory.lessons import LessonLibrary

    library = LessonLibrary(Path(config.memory.lessons_global_path).expanduser())

    if args.seed:
        from .models.host_notes import seed_library

        print("Seeding verified facts about this host's models:")
        count = seed_library(library)
        print(f"\n{count} lesson(s) installed or confirmed at {library.root}")
        return 0

    lessons = library.search(" ".join(args.query), limit=args.limit) if args.query else library.all()[: args.limit]
    if args.json:
        _print([lesson.to_dict() for lesson in lessons], True)
        return 0
    if not lessons:
        print("(no lessons recorded yet)")
        return 0
    for lesson in lessons:
        status = "established" if lesson.established else f"score {lesson.score:.2f}"
        print(f"[{status}, used {lesson.used}x] {lesson.title}")
        print(f"  {lesson.body.strip()[:400]}")
        print()
    print(json.dumps(library.stats(), indent=2))
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    """Mine the ledger for defects in Forge itself.

    Distinct from `lessons`, which records what was learned about building
    software. This records what the platform got wrong, and the output is meant
    to leave the machine: the fix is always a code change.
    """
    config = _config(args, create=False)
    from .improve.feedback import FeedbackStore, collect, render
    from .kernel.ledger import Ledger

    store = FeedbackStore(Path(config.memory.feedback_global_path).expanduser())

    if args.history:
        findings = store.all()
        if args.json:
            _print([f.to_dict() for f in findings], True)
            return 0
        print(render(findings, verbose=args.verbose))
        return 0

    ledger = Ledger(config.ledger_path)
    try:
        findings = collect(ledger, cloud_target=config.budget.cloud_fraction_target)
    finally:
        ledger.close()

    if args.json:
        _print([f.to_dict() for f in findings], True)
    else:
        print(render(findings, verbose=args.verbose))

    if findings and not args.no_record:
        new, updated = store.record(findings)
        print(f"\nRecorded to {store.root}: {new} new, {updated} updated.")
    return 3 if any(f.severity == "critical" for f in findings) else 0


def cmd_escalations(args: argparse.Namespace) -> int:
    """Inspect the cross-project corpus of weak-failed/strong-succeeded pairs.

    The raw material for teaching the local model, kept deliberately separate
    from `lessons`: a lesson is a claim, a pair is evidence.
    """
    config = _config(args, create=False)
    from .improve.escalation import EscalationCorpus, find_pairs, render
    from .kernel.ledger import Ledger

    corpus = EscalationCorpus(config.memory.escalations_global_path)

    if args.history:
        pairs = corpus.all()
    else:
        ledger = Ledger(config.ledger_path)
        try:
            with _orchestrator(config) as orchestrator:
                titles = {n.id: n.title for n in orchestrator.graph.all_nodes()}
                project = orchestrator.project.name if orchestrator.project else ""
            pairs = find_pairs(ledger, ladder=config.models.ladder, project=project, titles=titles)
        finally:
            ledger.close()
        if not args.no_record:
            written = corpus.record(pairs)
            print(f"Recorded {written} new pair(s) to {corpus.root}.\n")

    if args.json:
        _print([p.to_dict() for p in pairs], True)
        return 0
    print(render(pairs, verbose=args.verbose))
    if args.history:
        print(json.dumps(corpus.stats(), indent=2))
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    config = _config(args, create=False)
    with _orchestrator(config) as orchestrator:
        data = orchestrator.models.router.describe()
        if args.json:
            _print(data, True)
            return 0
        print("Ladder:", " -> ".join(data["ladder"]))
        if data["unusable"]:
            print("Unusable:", ", ".join(data["unusable"]), "(missing credentials or endpoint)")
        print()
        print(
            f"{'MODEL':<12} {'VIA':<14} {'HOSTED':<7} {'TIER':<9} {'CONTEXT':>9} {'MAXOUT':>7} "
            f"{'THINK':<6} {'QUOTA/H':>7}  READY"
        )
        for row in data["models"]:
            thinking = {True: "on", False: "off", None: "-"}[row.get("thinking")]
            quota = row["quota_per_hour"] or "-"
            print(
                f"{row['name']:<12} {row['provider_kind']:<14} {row['hosted']:<7} {row['tier']:<9} "
                f"{row['context']:>9,} {row['max_output']:>7,} {thinking:<6} {quota!s:>7}  "
                f"{'yes' if row['available'] else 'no'}"
            )
        subscription = [r for r in data["models"] if r["provider_kind"].endswith("_cli")]
        if subscription:
            print()
            print("Subscription rungs (no API key; they use your existing CLI login):")
            for row in subscription:
                used = data.get("quota_used", {}).get(row["name"], 0)
                limit = row["quota_per_hour"]
                print(
                    f"  {row['name']:<12} {used} call(s) in the last hour"
                    + (f" of {limit}" if limit else " (no quota configured)")
                )
        policy = data["policy"]
        if policy:
            print()
            print(f"{'TASK CLASS':<18} {'RUNG':<16} {'OK':>5} {'FAIL':>5} {'P(success)':>11} {'CONF':>6}")
            for row in policy:
                print(
                    f"{row['task_class']:<18} {row['rung']:<16} {row['successes']:>5.0f} "
                    f"{row['failures']:>5.0f} {row['posterior_mean']:>11.3f} {row['confidence']:>6.2f}"
                )
        for note in orchestrator.models.policy.recommendations():
            print(f"  * {note}")
        print()
        print(json.dumps(data["budget"], indent=2))
    return 0


def cmd_gates(args: argparse.Namespace) -> int:
    config = _config(args)
    from .validation import gates as _gates  # noqa: F401
    from .validation.gate import gate_registry

    if args.run:
        from .workspace.sandbox import detect_toolchain

        with _orchestrator(config) as orchestrator:
            gate_ctx = orchestrator.gates.build_context(
                root=orchestrator.repo.path,
                sandbox=orchestrator.sandbox,
                toolchain=detect_toolchain(orchestrator.sandbox),
                node_id="manual",
            )
            report = orchestrator.gates.run(args.run, gate_ctx, use_cache=not args.no_cache)
            print(report.render(failures_only=False))
            print()
            print(report.summary_line())
            return 0 if report.passed else 1

    rows = gate_registry.describe()
    if args.json:
        _print(rows, True)
        return 0
    print(f"{'GATE':<20} {'ORDER':>6} {'CACHE':>6} {'BLOCK':>6}  DESCRIPTION")
    for row in rows:
        print(
            f"{row['name']:<20} {row['order']:>6} {'yes' if row['cacheable'] else 'no':>6} "
            f"{'yes' if row['blocking'] else 'no':>6}  {row['description']}"
        )
    print()
    print("Enabled for this project:", ", ".join(config.validation.gates))
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    from .agents.registry import agent_registry, all_kinds

    all_kinds()
    rows = agent_registry.describe()
    if args.json:
        _print(rows, True)
        return 0
    print(f"{'KIND':<16} {'TASK CLASS':<18} {'DIFFICULTY':>10} {'STAKES':>7} {'COMMITS':>8}")
    for row in rows:
        print(
            f"{row['kind']:<16} {row['task_class']:<18} {row['difficulty']:>10.2f} "
            f"{row['stakes']:>7.2f} {'yes' if row['commits'] else 'no':>8}"
        )
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    config = _config(args, create=False)
    with _orchestrator(config) as orchestrator:
        from .improve.metrics import compute_metrics
        from .improve.promotion import detect_promotions, gate_promotions, routing_promotions

        metrics = compute_metrics(orchestrator.ledger, orchestrator.graph, milestone=args.milestone)
        if args.json:
            _print(metrics.to_dict(), True)
            return 0
        print(metrics.render())
        promotions = detect_promotions(orchestrator.ledger, orchestrator.memory)
        if promotions:
            print("\nRepeated problems that tooling could prevent:")
            for candidate in promotions:
                print(f"  ({candidate.occurrences}x from {candidate.origin}) {candidate.signature}")
        notes = routing_promotions(orchestrator.ledger) + gate_promotions(orchestrator.ledger)
        if notes:
            print("\nObservations:")
            for note in notes:
                print(f"  * {note}")
    return 0


def cmd_repair(args: argparse.Namespace) -> int:
    """Rebuild every projection from the event log.

    The fix for any suspected corruption of derived state. The event log is the
    truth; everything else is recomputable, and this recomputes it.
    """
    config = _config(args)
    from .kernel.ledger import Ledger
    from .memory.store import MemoryStore

    ledger = Ledger(config.ledger_path)
    try:
        counts = ledger.rebuild_projections()
        memory = MemoryStore(ledger)
        records = memory.rebuild()
        ledger.vacuum()
        print(f"Replayed {counts['events']} events -> {counts['nodes']} nodes, {records} memory records.")
    finally:
        ledger.close()
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    config = _config(args, create=False)
    from .kernel.ledger import Ledger

    ledger = Ledger(config.ledger_path)
    try:
        if args.stats:
            _print(ledger.stats(), args.json)
            return 0
        # The tail, not the head. `--limit 50` on a log means "the last 50";
        # reading from event #1 shows project creation forever, however long the
        # run goes on. `--after` switches to reading forward from a known point.
        if args.after:
            events = ledger.read(after_seq=args.after, types=args.type or None, limit=args.limit)
        else:
            events = ledger.tail(args.limit, types=args.type or None)
        if args.json:
            _print([e.to_dict() for e in events], True)
            return 0
        for event in events:
            from .util.clock import iso

            print(f"#{event.seq:<6} {iso(event.ts)} {event.type:<26} {json.dumps(event.payload, default=str)[:140]}")
    finally:
        ledger.close()
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    config = _config(args, create=False)
    if args.write:
        path = config.forge_dir / "config.toml"
        write_default_config(path)
        print(f"Wrote starter configuration to {path}")
        return 0
    _print(config.redacted(), True)
    if config.sources:
        print(f"\n# loaded from: {', '.join(config.sources)}", file=sys.stderr)
    return 0


def cmd_opencode_config(args: argparse.Namespace) -> int:
    """Render the same local-only config used by embedded OpenCode runs."""
    config = _config(args, create=False)
    from .errors import ConfigError
    from .execution.opencode import OpenCodeExecutor
    from .workspace.sandbox import LocalSandbox

    model_name = args.model or config.models.default
    spec = config.models.models.get(model_name)
    if spec is None:
        raise ConfigError(
            f"unknown OpenCode model {model_name!r}",
            available=sorted(config.models.models),
        )
    if spec.hosted != "local":
        raise ConfigError(
            "OpenCode execution config may only expose a local model",
            model=model_name,
        )
    # Configuration rendering does not execute anything. A LocalSandbox avoids
    # requiring Docker merely to print JSON even when the project runs in it.
    sandbox = LocalSandbox(config.sandbox, config.workspace_dir)
    executor = OpenCodeExecutor(
        config,
        sandbox,
        node_id="server",
        model_name=model_name,
    )
    rendered = json.dumps(executor.configuration(), indent=2) + "\n"
    if args.write is not None:
        path = (
            Path(args.write).expanduser()
            if args.write
            else config.forge_dir / "opencode" / "server.json"
        )
        if not path.is_absolute():
            path = config.project_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"Wrote local-only OpenCode configuration to {path}")
        return 0
    print(rendered, end="")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check that the host can actually run a build.

    Worth running before starting a multi-day job. Every check here corresponds
    to a failure that would otherwise appear hours in, disguised as something
    else.
    """
    config = _config(args, create=False)
    from .util.proc import which
    from .validation.gates.browser import playwright_available
    from .validation.gates.visual import imagemagick_available

    checks: list[tuple[str, bool, str]] = []
    checks.append(("git", which("git") is not None, "required for checkpoints and rollback"))
    checks.append(("python3", sys.version_info >= (3, 12), f"3.12+ required, running {sys.version.split()[0]}"))
    checks.append(("docker", which("docker") is not None, "optional: stronger sandbox isolation"))
    checks.append(("playwright", playwright_available(), "optional: browser and visual gates"))
    checks.append(("imagemagick", imagemagick_available(), "optional: visual regression comparison"))
    checks.append(("ffmpeg", which("ffmpeg") is not None, "optional: video capture"))
    if config.coding.backend != "native":
        opencode_present = which(config.coding.opencode_command) is not None
        optional = config.coding.backend == "auto" or config.coding.fallback_to_native
        checks.append(
            (
                "opencode",
                opencode_present,
                f"{config.coding.opencode_command} local coding executor"
                + (" [optional; native fallback enabled]" if optional else " [required]"),
            )
        )

    from .models.cli_provider import cli_login_state
    from .models.registry import Registry

    registry = Registry(config.models)
    usable: list[str] = []
    seen_commands: set[str] = set()
    for name in config.models.ladder:
        spec = registry.spec(name)
        provider = config.models.providers[spec.provider]
        available = registry.available(name)

        # CLI-backed rungs use the operator's existing subscription login, so
        # "installed" is not the same as "usable". Check the login too, once
        # per command rather than once per rung.
        if provider.kind in ("claude_cli", "codex_cli"):
            logged_in, detail = cli_login_state(provider.command)
            available = available and logged_in
            detail = f"{provider.command} ({detail})"
            if provider.command not in seen_commands:
                seen_commands.add(provider.command)
        elif provider.api_key_env:
            detail = f"{spec.provider} -> {spec.model or '(server default)'}"
            if not available:
                detail += f" (optional: set {provider.api_key_env})"
        else:
            detail = f"{spec.provider} -> {(spec.model or '(server default)')[-46:]}"

        if available:
            usable.append(name)
        checks.append((f"model:{name}", available, detail + ("" if available else " [optional]")))

        # A rung can be reachable and still incapable of ever answering: output
        # tokens take wall-clock, and a budget the model cannot emit before its
        # own timeout fails every single call, reporting a read timeout that
        # blames the network. Swapping to a model seven times slower did exactly
        # this to all the local rungs at once, and a live run spent two hours
        # discovering it. Checking the arithmetic takes no requests at all.
        deliverable = spec.deliverable_tokens()
        if deliverable and spec.max_output_tokens > deliverable:
            minutes = spec.max_output_tokens / spec.tokens_per_second / 60
            checks.append(
                (
                    f"budget:{name}",
                    False,
                    f"max_output_tokens={spec.max_output_tokens} needs ~{minutes:.0f} min "
                    f"at {spec.tokens_per_second:.0f} tok/s but timeout is "
                    f"{spec.timeout / 60:.0f} min: raise timeout or lower the budget",
                )
            )

    # A ladder taller than the attempt budget has rungs no node can ever reach.
    # The failure is silent and expensive: the node blocks with "exhausted the
    # N-attempt budget" having never once been served by the models that would
    # have finished it.
    #
    # One attempt per rung after the first escalation, not one *escalation
    # interval* per rung: escalation is a single step per failure once
    # `escalate_after_attempts` is passed, so the top rung is first served on
    # attempt `escalate_after_attempts + len(ladder) - 1`. The old formula
    # multiplied instead, which both missed real cases and failed configs that
    # were fine.
    rungs = len(config.models.ladder)
    needed = attempts_needed_for_ladder(rungs, config.scheduler.escalate_after_attempts)
    checks.append(
        (
            "ladder reachable",
            config.scheduler.max_attempts >= needed,
            f"{rungs} rungs escalating after {config.scheduler.escalate_after_attempts} "
            f"attempt(s) needs max_attempts >= {needed}, have "
            f"{config.scheduler.max_attempts}",
        )
    )

    # An individual rung being unreachable is survivable -- the ladder simply
    # gets shorter and the router adapts. Having *no* rung is not.
    checks.append(
        (
            "usable ladder",
            bool(usable),
            f"{len(usable)} of {len(config.models.ladder)} rungs reachable: {', '.join(usable) or 'none'}",
        )
    )

    reachable = _probe_local(config)
    checks.append(
        ("local endpoint", reachable[0], reachable[1] + ("" if reachable[0] else " -- optional if a cloud rung is reachable"))
    )

    writable = os.access(config.forge_dir, os.W_OK)
    checks.append(("state directory", writable, str(config.forge_dir)))

    ok = True
    for name, passed, detail in checks:
        optional = "optional" in detail
        mark = "ok  " if passed else ("warn" if optional else "FAIL")
        if not passed and not optional:
            ok = False
        print(f"[{mark}] {name:<20} {detail}")

    print()
    if ok:
        print("Ready to run.")
    else:
        print("Some required checks failed; fix them before starting a long run.", file=sys.stderr)
    return 0 if ok else 1


def _probe_local(config: Config) -> tuple[bool, str]:
    """Check the local model endpoint answers, without spending anything."""
    from .models.health import probe_local

    return probe_local(config)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Forge -- an autonomous software engineering platform.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical use:\n"
            '  forge init "Build a browser-based Quake-inspired FPS with one polished level."\n'
            "  forge run          # starts detached and returns; survives a closed terminal\n"
            "  forge watch        # follow it\n"
            "  forge status       # is it working, is it going well, does it need me\n"
            "  forge stop         # wind it down\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"forge {__version__}")
    parser.add_argument("--dir", "-C", default=".", help="project directory (default: cwd)")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug-level console output")
    parser.add_argument("--quiet", "-q", action="store_true", help="suppress console logging")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a project from a one-line description")
    # Optional: at a terminal with no goal, init asks for one and then for
    # reference material. Scripts pass both as arguments and never see a prompt.
    p.add_argument("goal", nargs="*", help="what to build (prompted for if omitted)")
    p.add_argument("--name", help="short project name (default: derived from the goal)")
    p.add_argument("--no-config", action="store_true", help="do not write a starter config file")
    p.add_argument(
        "--reference",
        action="append",
        metavar="SOURCE[::DESCRIPTION]",
        help="reference material: a URL or local path, repeatable. Skips the prompt.",
    )
    p.add_argument("--no-input", action="store_true", help="never prompt; for scripts and CI")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("reference", help="manage the material Forge builds against")
    ref_sub = p.add_subparsers(dest="reference_command", required=True)
    p_add = ref_sub.add_parser("add", help="fetch or copy reference material into the project")
    p_add.add_argument("source", nargs="+", metavar="SOURCE[::DESCRIPTION]", help="URL or path")
    p_add.add_argument("--describe", help="what Forge should take from it")
    p_add.add_argument(
        "--role",
        choices=["visual", "motion", "audio", "document", "example", "other"],
        help="override the role inferred from the file type",
    )
    ref_sub.add_parser("list", help="show the declared references")
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("run", help="start or resume the build (detached by default)")
    p.add_argument("--workers", type=int, help="parallel node execution")
    p.add_argument("--budget", type=float, help="override the total cost ceiling")
    p.add_argument("--max-nodes", type=int, help="stop after this many node attempts")
    p.add_argument("--forever", action="store_true", help="do not stop when the graph goes quiet")
    p.add_argument("--sandbox", choices=["local", "docker"], help="override the sandbox kind")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="use the echo provider; spend nothing (implies --foreground)",
    )
    p.add_argument(
        "--foreground",
        action="store_true",
        help="run in this terminal and die with it (for CI and --dry-run)",
    )
    p.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="start detached, then tail the log; Ctrl-C detaches without stopping",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("stop", help="wind down the active run")
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds to wait for in-flight nodes before reporting back",
    )
    p.add_argument(
        "--kill",
        action="store_true",
        help="SIGKILL after the timeout; loses the current node's work",
    )
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("status", help="show progress and anything needing attention")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("watch", help="follow a running build")
    p.add_argument("--interval", type=float, default=3.0)
    p.add_argument("--status", action="store_true", help="also print the status block each tick")
    p.add_argument(
        "--no-code",
        action="store_true",
        help="hide live file contents written in node worktrees",
    )
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("report", help="write the HTML dashboard")
    p.add_argument("--open", action="store_true", help="open it in a browser")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("nodes", help="list the task graph")
    p.add_argument("--status", help="filter by status")
    p.add_argument("--all", action="store_true", help="include completed nodes")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_nodes)

    p = sub.add_parser("node", help="inspect one node in detail")
    p.add_argument("node_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_node)

    p = sub.add_parser("cancel", help="cancel a node")
    p.add_argument("node_id")
    p.add_argument("--reason", default="")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("unblock", help="answer a blocked node and requeue it")
    p.add_argument("node_id")
    p.add_argument("answer", nargs="*", help="guidance to record and apply")
    p.set_defaults(func=cmd_unblock)

    p = sub.add_parser("tell", help="add guidance to a running project")
    p.add_argument("guidance", nargs="+", help="what the build should know or do differently")
    p.add_argument("--title", help="short title for the record")
    p.add_argument(
        "--convention",
        action="store_true",
        help="record as a coding convention rather than a requirement",
    )
    p.set_defaults(func=cmd_tell)

    p = sub.add_parser("checkpoints", help="list restorable points")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--kind", choices=["node", "milestone", "manual", "pre_attempt"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_checkpoints)

    p = sub.add_parser("rollback", help="reset the workspace to a checkpoint")
    p.add_argument("checkpoint", nargs="?", help="checkpoint id (default: last milestone)")
    p.add_argument("--reason", default="")
    p.add_argument("--yes", action="store_true", help="skip confirmation")
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("memory", help="query project memory")
    p.add_argument("query", nargs="*")
    p.add_argument("--kind", help="assumption, decision, interface, convention, fact, finding, lesson")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--export", help="write the whole memory to a markdown file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("lessons", help="inspect the cross-project lesson library")
    p.add_argument("query", nargs="*")
    p.add_argument(
        "--seed",
        action="store_true",
        help="install verified facts about this host's models into the library",
    )
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_lessons)

    p = sub.add_parser("feedback", help="find defects in Forge itself, from this project's ledger")
    p.add_argument("--history", action="store_true", help="show accumulated cross-project findings")
    p.add_argument("--no-record", action="store_true", help="report without adding to the store")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("escalations", help="weak-failed/strong-succeeded pairs, the corpus for teaching the local model")
    p.add_argument("--history", action="store_true", help="show the accumulated cross-project corpus")
    p.add_argument("--no-record", action="store_true", help="report without adding to the corpus")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_escalations)

    p = sub.add_parser("policy", help="show model routing state and spend")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_policy)

    p = sub.add_parser("gates", help="list or run validation gates")
    p.add_argument("--run", nargs="+", metavar="GATE", help="run these gates now")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_gates)

    p = sub.add_parser("agents", help="list registered specialist agents")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_agents)

    p = sub.add_parser("metrics", help="measured outcomes for the run or a milestone")
    p.add_argument("--milestone")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("ledger", help="read the event log")
    p.add_argument("--after", type=int, default=0)
    p.add_argument("--type", action="append", help="filter by event type (repeatable)")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--stats", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ledger)

    p = sub.add_parser("repair", help="rebuild derived state from the event log")
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser("config", help="show or write configuration")
    p.add_argument("--write", action="store_true", help="write a starter config file")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser(
        "opencode-config",
        help="render the local-only OpenCode server configuration",
    )
    p.add_argument("--model", help="local Forge model to expose (default: models.default)")
    p.add_argument(
        "--write",
        nargs="?",
        const="",
        metavar="PATH",
        help="write PATH (default: .forge/opencode/server.json)",
    )
    p.set_defaults(func=cmd_opencode_config)

    p = sub.add_parser("doctor", help="check this host can run a build")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw)
    # Kept so a detached run can replay this exact command line in the
    # foreground instead of reconstructing it from the namespace.
    args.argv = raw
    try:
        return int(args.func(args) or 0)
    except ForgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # pragma: no cover - piping into head
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
