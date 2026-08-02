

# --------------------------------------------------------------------------
# Node identity in the CLI output
# --------------------------------------------------------------------------


def test_node_listings_show_an_id_that_the_cli_accepts() -> None:
    """`forge status` says to run `forge unblock <node>`; something must print one.

    Observed live: an operator with a blocked node had no command that showed
    an id, so the instruction in the status output could not be followed.
    """
    from types import SimpleNamespace

    from forge.report.progress import render_nodes

    node = SimpleNamespace(
        id="node_01KYQ6HMAPMCJVRD3Q3KVHTP3Q", status="blocked", kind="scaffold",
        attempts=70, cost=7.434, title="Scaffold the Vite + TypeScript project",
    )
    rendered = render_nodes([node])

    assert "3KVHTP3Q" in rendered, "the short id must appear in the listing"
    assert "ID" in rendered.splitlines()[0], "and be labelled in the header"


def test_the_printed_id_resolves_back_to_the_node() -> None:
    """Printing an id that the CLI then rejects would be worse than none."""
    from types import SimpleNamespace

    from forge.cli import _resolve_node
    from forge.report.progress import _short_id

    node = SimpleNamespace(id="node_01KYQ6HMAPMCJVRD3Q3KVHTP3Q")
    orchestrator = SimpleNamespace(graph=SimpleNamespace(all_nodes=lambda: [node]))

    assert _resolve_node(orchestrator, _short_id(node.id)) is node


def test_blocked_nodes_in_status_carry_their_id() -> None:
    from forge.report.progress import render_status

    rendered = render_status({
        "project": {"name": "p", "goal": "g"},
        "counts": {"blocked": 1},
        "blocked": [{"id": "node_01KYQ6HMAPMCJVRD3Q3KVHTP3Q", "title": "Scaffold", "question": "stuck"}],
    })
    assert "3KVHTP3Q" in rendered


# --------------------------------------------------------------------------
# Periodic usage reporting
# --------------------------------------------------------------------------


def _usage_event(models, *, window=300, cloud=0.5):
    from types import SimpleNamespace

    return SimpleNamespace(
        type="usage.report", ts=0.0, node_id=None,
        payload={"models": models, "window_seconds": window, "cloud_fraction": cloud},
    )


def test_usage_reports_appear_in_the_watch_stream() -> None:
    from forge.report.progress import render_timeline

    rendered = render_timeline([_usage_event([
        {"model": "local", "calls": 8, "input_tokens": 12400, "output_tokens": 3200},
        {"model": "claude", "calls": 1, "input_tokens": 24000, "output_tokens": 6800},
    ])])

    assert "usage.report" in rendered
    assert "local" in rendered and "claude" in rendered
    assert "12.4kin/3.2kout" in rendered, "token counts must be legible at a glance"
    assert "x8" in rendered, "call counts distinguish one big call from many small ones"


def test_the_usage_line_reports_the_window_not_the_running_total() -> None:
    """`forge status` already answers "what has this cost".

    Watching a live run, the question is whether the last few minutes went
    local or cloud -- which a cumulative total cannot answer.
    """
    from forge.report.progress import _usage_detail

    detail = _usage_detail({
        "models": [{"model": "local", "calls": 3, "input_tokens": 900, "output_tokens": 400,
                    "total_output_tokens": 999_999}],
        "window_seconds": 300, "cloud_fraction": 0.12,
    })
    assert "last 5m" in detail
    assert "999" not in detail, "the cumulative figure must not be mistaken for the window"
    assert "cloud 12%" in detail


def test_a_quiet_window_says_so_rather_than_printing_nothing() -> None:
    """Silence during a long local node would read as the reporter being broken."""
    from forge.report.progress import _usage_detail

    detail = _usage_detail(
        {"models": [{"model": "local", "calls": 0, "input_tokens": 0, "output_tokens": 0}],
         "window_seconds": 300}
    )
    assert "no completed model calls" in detail
    assert "last 5m" in detail


def test_model_requests_are_visible_before_the_long_call_finishes() -> None:
    from types import SimpleNamespace

    from forge.report.progress import render_timeline

    event = SimpleNamespace(
        ts=1785417780.0,
        type="model.request",
        node_id="node_01KYQ6HMAPMCJVRD3Q3KVHTP3Y",
        payload={"label": "opencode:3KVHTP3Y"},
    )
    rendered = render_timeline([event])
    assert "model.request" in rendered
    assert "opencode:3KVHTP3Y" in rendered


def test_status_names_a_run_that_is_not_running() -> None:
    """`[working]` means the graph has work left, not that anything is doing it.

    Once runs detach those two came apart, and a killed run with pending nodes
    reads as perfectly healthy from the counts alone.
    """
    from forge.report.progress import render_status

    out = render_status({"counts": {"pending": 3}, "progress": 0.2, "run": {"state": "crashed"}})
    assert "NOT RUNNING" in out

    live = render_status(
        {"counts": {"running": 1}, "progress": 0.2, "run": {"state": "live", "pid": 42, "uptime": 90.0}}
    )
    assert "pid 42" in live


def test_status_reports_a_run_that_has_gone_quiet() -> None:
    """The wedged-worker case: full heartbeat, unchanged counts, no progress."""
    from forge.report.progress import render_status

    out = render_status({"counts": {"running": 1}, "progress": 0.3, "quiet_for": 2700.0})
    assert "nothing has happened" in out
    assert "45m" in out

    busy = render_status({"counts": {"running": 1}, "progress": 0.3, "quiet_for": 30.0})
    assert "nothing has happened" not in busy


def test_a_slow_model_does_not_make_every_healthy_call_look_stuck() -> None:
    """The threshold is relative to how long one call may legitimately take.

    A fixed 15 minutes was right at 85 tok/s and became a false alarm on every
    healthy call once the local rung dropped to 12 tok/s with an hour-long
    timeout. A warning that fires during normal operation trains the operator to
    ignore it, which costs more than having no warning.
    """
    from forge.report.progress import render_status

    slow_but_healthy = {
        "counts": {"running": 1},
        "progress": 0.3,
        "quiet_for": 1_800.0,
        "quiet_threshold": 5_400.0,
    }
    assert "may be stuck" not in render_status(slow_but_healthy)

    genuinely_wedged = {**slow_but_healthy, "quiet_for": 6_000.0}
    assert "may be stuck" in render_status(genuinely_wedged)


def test_a_failed_gate_shows_why_in_the_timeline() -> None:
    """"`npm run test` failed with exit 1" is not something an operator can act on.

    The reason was already in the payload; the timeline just never printed it,
    so diagnosing a failure meant leaving the tool and running the command by
    hand.
    """
    from types import SimpleNamespace

    from forge.report.progress import render_timeline

    event = SimpleNamespace(
        ts=1785417780.0,
        type="gate.failed",
        node_id="node_abc",
        payload={
            "gate": "types",
            "summary": "`npm run typecheck` failed with exit 2",
            "evidence": "src/engine/collide.ts(11,1): error TS2306: File is not a module.\n"
            "src/engine/solver.ts(134,21): error TS2345: Argument of type 'Manifold | undefined'",
        },
    )

    out = render_timeline([event])
    assert "failed with exit 2" in out
    assert "TS2306" in out, "the timeline must show the reason, not only the exit code"
    assert "is not a module" in out


def test_the_timeline_does_not_become_a_build_log() -> None:
    from types import SimpleNamespace

    from forge.report.progress import render_timeline

    event = SimpleNamespace(
        ts=1785417780.0,
        type="gate.failed",
        node_id="n",
        payload={"gate": "unit", "summary": "failed", "evidence": "\n".join(
            f"line {i}" for i in range(200)
        )},
    )

    out = render_timeline([event])
    assert out.count("line ") <= 8, "a long log must be truncated in the stream"
    assert "more line(s)" in out, "and must say that it was truncated"


def test_a_passing_event_prints_no_evidence_block() -> None:
    from types import SimpleNamespace

    from forge.report.progress import render_timeline

    event = SimpleNamespace(
        ts=1785417780.0, type="node.succeeded", node_id="n",
        payload={"title": "did the thing", "evidence": "should not appear"},
    )
    assert "should not appear" not in render_timeline([event])


def test_watch_prints_each_live_file_version_once(config) -> None:
    """The stream shows code while the model is still working."""
    from forge.cli import _render_live_code
    from forge.workspace.git import Repo

    main = Repo(config.workspace_dir).init()
    node_id = "node_01KYQ6HMAPMCJVRD3Q3KVHTP3Y"
    worktree = main.add_worktree(
        config.worktrees_dir / node_id,
        f"forge/node/{node_id}",
    )
    source = worktree.path / "src" / "answer.ts"
    source.parent.mkdir(parents=True)
    source.write_text("export const answer = 41;\n", encoding="utf-8")
    # Dependency tooling writes this, not the coding model. It should not swamp
    # the source an operator is trying to inspect.
    (worktree.path / "package-lock.json").write_text('{"lockfileVersion": 3}\n')

    seen: dict[tuple[str, str], str] = {}
    first = _render_live_code(config, [node_id], seen)
    assert "src/answer.ts" in first
    assert "export const answer = 41;" in first
    assert "package-lock" not in first
    assert _render_live_code(config, [node_id], seen) == ""

    source.write_text("export const answer = 42;\n", encoding="utf-8")
    second = _render_live_code(config, [node_id], seen)
    assert "export const answer = 42;" in second


def test_patch_applied_is_visible_in_the_timeline() -> None:
    from types import SimpleNamespace

    from forge.report.progress import render_timeline

    event = SimpleNamespace(
        ts=1785417780.0,
        type="patch.applied",
        node_id="node_01KYQ6HMAPMCJVRD3Q3KVHTP3Y",
        payload={"summary": "src/main.ts"},
    )
    assert "patch.applied" in render_timeline([event])
