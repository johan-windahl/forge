"""End-to-end orchestration: execution, durability, recovery, self-improvement.

These drive the real orchestrator against scripted model output. They are the
tests that would catch a regression in the property the whole platform exists
for: that a build survives being interrupted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import pytest

from forge.agents.base import Agent, AgentContext, AgentResult, ProposedNode
from forge.agents.coding import MAX_FIX_ROUNDS, CodingAgent
from forge.agents.registry import agent_registry
from forge.execution.opencode import OpenCodeExecutor, OpenCodeResult, OpenCodeUsage
from forge.kernel.graph import NodeStatus
from forge.kernel.orchestrator import Orchestrator
from forge.memory.store import fact
from forge.models.types import TaskClass

# --------------------------------------------------------------------------
# Test agents
# --------------------------------------------------------------------------


class RecordingAgent(Agent):
    """A deterministic agent used to drive the orchestrator without models."""

    kind = "t_record"
    task_class = TaskClass.IMPLEMENTATION
    commits = True
    runs: ClassVar[list[str]] = []
    behaviour: str = "succeed"

    def run(self, ctx: AgentContext) -> AgentResult:
        RecordingAgent.runs.append(ctx.node.id)
        (ctx.root / f"{ctx.node.id[-6:]}.txt").write_text(f"attempt {ctx.node.attempts}")

        match RecordingAgent.behaviour:
            case "fail_after_claude":
                from forge.kernel.events import Event, EventType

                ctx.models.ledger.append(
                    Event(
                        type=EventType.MODEL_RESPONSE,
                        node_id=ctx.node.id,
                        payload={"model": "claude"},
                    )
                )
                return AgentResult.failure("deliberate frontier failure")
            case "succeed_after_claude":
                from forge.kernel.events import Event, EventType

                ctx.models.ledger.append(
                    Event(
                        type=EventType.MODEL_RESPONSE,
                        node_id=ctx.node.id,
                        payload={"model": "claude"},
                    )
                )
            case "fail":
                return AgentResult.failure("deliberate failure")
            case "raise":
                raise RuntimeError("agent exploded")
            case "fail_then_succeed":
                if ctx.node.attempts < 2:
                    return AgentResult.failure("first attempt fails")
            case "needs_human":
                return AgentResult(success=True, summary="done but stuck",
                                   needs_human="what colour should it be?")
        return AgentResult(
            success=True,
            summary=f"did the work on attempt {ctx.node.attempts}",
            changed_files=[f"{ctx.node.id[-6:]}.txt"],
            commit_message=f"feat: {ctx.node.title[:40]}",
            memory=[fact(f"Observed during {ctx.node.id[-6:]}", "a durable fact")],
        )


class SpawningAgent(Agent):
    kind = "t_spawn"
    task_class = TaskClass.PLANNING
    commits = False

    def run(self, ctx: AgentContext) -> AgentResult:
        if ctx.spec.get("spawned"):
            return AgentResult(success=True, summary="leaf")
        return AgentResult(
            success=True,
            summary="spawned two children",
            nodes=[
                ProposedNode(kind="t_record", title="child one", spec={"spawned": True}),
                ProposedNode(kind="t_record", title="child two", spec={"spawned": True}, deps=[0]),
            ],
        )


@pytest.fixture(autouse=True)
def _register_test_agents():
    from forge.agents.registry import _load_builtins

    _load_builtins()
    agent_registry.register(RecordingAgent)
    agent_registry.register(SpawningAgent)
    RecordingAgent.runs = []
    RecordingAgent.behaviour = "succeed"
    yield


def _project(orchestrator: Orchestrator) -> None:
    orchestrator.create_project("Build a thing that works")
    # Remove the real planning and goal nodes; these tests drive the kernel.
    for node in orchestrator.graph.all_nodes():
        orchestrator.graph.cancel(node.id, "not used in this test")


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def test_a_node_runs_commits_and_records_memory(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    node = orchestrator.graph.add_node("t_record", "do the work")

    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    done = orchestrator.graph.get(node.id)
    assert done.status == NodeStatus.SUCCEEDED
    assert done.result["commit"]
    assert orchestrator.repo.log(limit=1)[0].node_id == node.id
    assert orchestrator.memory.search("durable fact")


def test_dependencies_are_respected_in_execution_order(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    first = orchestrator.graph.add_node("t_record", "first")
    second = orchestrator.graph.add_node("t_record", "second", deps=[first.id])

    orchestrator.run(install_signal_handlers=False)
    assert RecordingAgent.runs == [first.id, second.id]


def test_spawned_nodes_are_added_and_executed(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    orchestrator.graph.add_node("t_spawn", "plan it")
    orchestrator.run(install_signal_handlers=False)

    titles = {n.title for n in orchestrator.graph.all_nodes(status=NodeStatus.SUCCEEDED)}
    assert {"plan it", "child one", "child two"} <= titles


def test_failed_nodes_are_retried_then_succeed(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    RecordingAgent.behaviour = "fail_then_succeed"
    node = orchestrator.graph.add_node("t_record", "flaky work")

    orchestrator.run(install_signal_handlers=False)

    done = orchestrator.graph.get(node.id)
    assert done.status == NodeStatus.SUCCEEDED
    assert done.attempts == 2


def test_an_agent_that_raises_does_not_stop_the_run(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    RecordingAgent.behaviour = "raise"
    broken = orchestrator.graph.add_node("t_record", "broken")
    orchestrator.graph.add_node("t_spawn", "independent work")

    orchestrator.run(install_signal_handlers=False)

    assert orchestrator.graph.get(broken.id).status == NodeStatus.BLOCKED
    others = [n for n in orchestrator.graph.all_nodes() if n.title == "independent work"]
    assert others[0].status == NodeStatus.SUCCEEDED, "one bad node must not stall the graph"


def test_exhausted_retries_block_with_a_specific_question(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    RecordingAgent.behaviour = "fail"
    node = orchestrator.graph.add_node(
        "t_record", "impossible", spec={"acceptance": ["it must do the impossible"]}
    )

    orchestrator.run(install_signal_handlers=False)

    blocked = orchestrator.graph.get(node.id)
    assert blocked.status == NodeStatus.BLOCKED
    assert "impossible" in blocked.result["question"]
    assert orchestrator.status()["stalled"]


def test_needs_human_parks_the_node_even_on_success(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    RecordingAgent.behaviour = "needs_human"
    node = orchestrator.graph.add_node("t_record", "ambiguous work")

    orchestrator.run(install_signal_handlers=False)

    blocked = orchestrator.graph.get(node.id)
    assert blocked.status == NodeStatus.BLOCKED
    assert "colour" in blocked.result["question"]


def test_failures_are_attributed_to_the_model_that_served_them(orchestrator: Orchestrator) -> None:
    """Otherwise the router learns that local fails at tasks local never tried.

    `node.tier` only changes on an explicit escalation, so using it attributed a
    first-attempt frontier failure to `local`. Found in routing_stats after a
    live run.
    """
    _project(orchestrator)
    RecordingAgent.behaviour = "fail_after_claude"
    orchestrator.graph.add_node("t_record", "will fail")
    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    rungs = {row["rung"] for row in orchestrator.models.policy.table()}
    assert "claude" in rungs
    assert "local" not in rungs, "a rung that never ran must not be blamed"


def test_failure_does_not_reuse_a_model_response_from_an_earlier_attempt(
    orchestrator: Orchestrator,
) -> None:
    """Restarts without a fresh call must not multiply stale model failures."""
    from forge.kernel.events import Event, EventType

    _project(orchestrator)
    RecordingAgent.behaviour = "fail"
    node = orchestrator.graph.add_node("t_record", "fails before calling a model")
    orchestrator.ledger.append(
        Event(
            type=EventType.MODEL_RESPONSE,
            node_id=node.id,
            payload={"model": "opus"},
        )
    )

    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    assert orchestrator.models.policy.table() == []


def test_success_is_recorded_once_for_the_model_that_served_the_attempt(
    orchestrator: Orchestrator,
) -> None:
    _project(orchestrator)
    RecordingAgent.behaviour = "succeed_after_claude"
    orchestrator.graph.add_node("t_record", "will pass validation")

    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    rows = orchestrator.models.policy.table()
    assert len(rows) == 1
    assert rows[0]["rung"] == "claude"
    assert rows[0]["successes"] == 1
    assert rows[0]["failures"] == 0


def test_escalation_moves_the_node_up_the_ladder(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    RecordingAgent.behaviour = "fail"
    orchestrator.config.scheduler.escalate_after_attempts = 1
    node = orchestrator.graph.add_node("t_record", "hard work")

    orchestrator.run(install_signal_handlers=False)

    tiers = [
        e.payload.get("to_tier")
        for e in orchestrator.ledger.read(node_id=node.id, types=["node.escalated"])
    ]
    assert tiers and tiers[0] == "local_deep"


# --------------------------------------------------------------------------
# Durability and recovery
# --------------------------------------------------------------------------


def test_a_crashed_run_resumes_where_it_stopped(config, provider) -> None:
    """The property the whole platform is built around."""
    first = Orchestrator(config, worker_prefix="crash")
    for name in config.models.providers:
        first.models.registry.install(name, provider)
    first.create_project("Survive a crash")
    for node in first.graph.all_nodes():
        first.graph.cancel(node.id, "not used")

    done = first.graph.add_node("t_record", "already finished")
    pending = first.graph.add_node("t_record", "not yet started", deps=[done.id])
    first.run(max_nodes=1, install_signal_handlers=False)
    assert first.graph.get(done.id).status == NodeStatus.SUCCEEDED
    # Simulate a hard crash: no clean shutdown, connection just goes away.
    first.ledger.close()

    second = Orchestrator(config, worker_prefix="resumed")
    for name in config.models.providers:
        second.models.registry.install(name, provider)
    second.run(install_signal_handlers=False)

    assert second.graph.get(pending.id).status == NodeStatus.SUCCEEDED
    assert second.graph.get(done.id).attempts == 1, "finished work must not be redone"
    second.close()


def test_recovery_reclaims_a_node_held_by_a_dead_worker(config, provider) -> None:
    orchestrator = Orchestrator(config, worker_prefix="dead")
    for name in config.models.providers:
        orchestrator.models.registry.install(name, provider)
    orchestrator.create_project("Reclaim orphans")
    for node in orchestrator.graph.all_nodes():
        orchestrator.graph.cancel(node.id, "not used")

    node = orchestrator.graph.add_node("t_record", "orphaned by a crash")
    orchestrator.graph.claim(node.id, "worker-that-died", 3600)
    assert orchestrator.graph.get(node.id).status == NodeStatus.RUNNING

    orchestrator._recover()
    assert orchestrator.graph.get(node.id).status == NodeStatus.READY

    orchestrator.run(install_signal_handlers=False)
    assert orchestrator.graph.get(node.id).status == NodeStatus.SUCCEEDED
    orchestrator.close()


def test_recovery_cleans_a_dirty_workspace(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    (orchestrator.repo.path / "half-written.txt").write_text("garbage from a dead attempt")
    assert orchestrator.repo.is_dirty()

    orchestrator._recover()
    assert not orchestrator.repo.is_dirty()


def test_each_attempt_starts_from_a_clean_tree(orchestrator: Orchestrator) -> None:
    """At-least-once execution is only safe if attempts cannot compound."""
    _project(orchestrator)
    RecordingAgent.behaviour = "fail_then_succeed"
    node = orchestrator.graph.add_node("t_record", "retried")

    orchestrator.run(install_signal_handlers=False)

    content = (orchestrator.repo.path / f"{node.id[-6:]}.txt").read_text()
    assert content == "attempt 2", "the second attempt must not see the first's leftovers"


def test_checkpoints_are_created_and_can_be_rolled_back(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    orchestrator.graph.add_node("t_record", "first change")
    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    checkpoint = orchestrator.checkpoints.latest(kind="node")
    assert checkpoint is not None

    (orchestrator.repo.path / "later.txt").write_text("added after the checkpoint")
    orchestrator.repo.commit("chore: later change")

    orchestrator.checkpoints.rollback(checkpoint.id, reason="test")
    assert not (orchestrator.repo.path / "later.txt").exists()
    assert orchestrator.repo.head() == checkpoint.commit


def test_rollback_preserves_the_history_that_caused_it(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    orchestrator.graph.add_node("t_record", "work")
    orchestrator.run(max_nodes=1, install_signal_handlers=False)
    checkpoint = orchestrator.checkpoints.latest(kind="node")
    before = orchestrator.ledger.head_seq()

    orchestrator.checkpoints.rollback(checkpoint.id, reason="test")

    assert orchestrator.ledger.head_seq() > before, "the log is appended to, never truncated"
    assert orchestrator.ledger.read(types=["rollback.performed"])


def test_budget_exhaustion_stops_the_run_cleanly(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    orchestrator.config.budget.total_cost = 0.0
    for i in range(3):
        orchestrator.graph.add_node("t_record", f"work {i}")

    summary = orchestrator.run(install_signal_handlers=False)
    assert summary["counts"]["succeeded"] <= 3  # completes without raising


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_status_reports_progress_and_blocked_work(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    RecordingAgent.behaviour = "fail"
    orchestrator.graph.add_node("t_record", "will block")
    orchestrator.run(install_signal_handlers=False)

    status = orchestrator.status()
    assert status["stalled"]
    assert status["blocked"][0]["title"] == "will block"

    from forge.report.progress import render_status

    rendered = render_status(status)
    assert "NEEDS ATTENTION" in rendered


def test_metrics_are_computed_from_the_ledger(orchestrator: Orchestrator) -> None:
    _project(orchestrator)
    RecordingAgent.behaviour = "fail_then_succeed"
    orchestrator.graph.add_node("t_record", "retried work")
    orchestrator.run(install_signal_handlers=False)

    from forge.improve.metrics import compute_metrics

    metrics = compute_metrics(orchestrator.ledger, orchestrator.graph)
    assert metrics.nodes_retried == 1
    assert metrics.rework_ratio > 0
    assert "retried work" in metrics.render() or metrics.nodes_succeeded >= 1


def test_dashboard_is_self_contained(orchestrator: Orchestrator, tmp_path: Path) -> None:
    _project(orchestrator)
    orchestrator.graph.add_node("t_record", "work")
    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    from forge.report.dashboard import write_dashboard

    path = write_dashboard(
        tmp_path / "index.html",
        status=orchestrator.status(),
        nodes=orchestrator.graph.all_nodes(),
        events=orchestrator.ledger.tail(50),
    )
    html = path.read_text()
    assert "<html" in html
    assert "src=\"http" not in html and "@import" not in html, "must not fetch anything external"


def test_dashboard_labels_screenshots_with_capture_timestamp(
    orchestrator: Orchestrator, tmp_path: Path
) -> None:
    _project(orchestrator)
    screenshot = tmp_path / "screenshot_home.png"
    screenshot.write_bytes(b"png")
    os.utime(screenshot, (1_700_000_000, 1_700_000_000))

    from forge.report.dashboard import write_dashboard

    path = write_dashboard(
        tmp_path / "index.html",
        status=orchestrator.status(),
        nodes=[],
        events=[],
        screenshots=[screenshot],
    )

    rendered = path.read_text()
    assert "screenshot_home.png" in rendered
    assert "captured 2023-11-14T22:13:20Z" in rendered


def test_operator_guidance_reaches_later_prompts(orchestrator: Orchestrator) -> None:
    """`forge tell` is the supported way to steer qualities no gate measures."""
    from forge.memory.store import requirement

    _project(orchestrator)
    orchestrator.memory.write(
        requirement(
            "Operator guidance: flipper feel",
            "A full-power lower-flipper shot should reach the top rollovers in about 0.9s.",
            source="human",
        )
    )
    hits = orchestrator.memory.search("how strong should the flippers be", limit=3)
    assert hits and "flipper" in hits[0].title.lower()

    # And it lands in the project digest that forms every prompt's stable prefix.
    assert "flipper feel" in orchestrator._project_context()["digest"].lower()


def test_external_operator_guidance_invalidates_cached_project_digest(
    orchestrator: Orchestrator,
) -> None:
    """The daemon and ``forge tell`` use distinct MemoryStore instances."""
    from forge.memory.store import MemoryStore, requirement

    _project(orchestrator)
    assert "camera-safe renderer caching" not in orchestrator._project_context()["digest"].lower()

    external = MemoryStore(orchestrator.ledger, orchestrator.project.id)
    external.write(
        requirement(
            "Camera-safe renderer caching",
            "Keep gradients attached to geometry under camera scroll and shake.",
            source="human",
        )
    )

    assert "camera-safe renderer caching" in orchestrator._project_context()["digest"].lower()


def test_promotion_detects_a_repeated_problem(orchestrator: Orchestrator) -> None:
    """The 'replace model reasoning with tooling' signal is measured, not guessed."""
    _project(orchestrator)
    from forge.improve.promotion import detect_promotions
    from forge.memory.store import finding

    for i in range(4):
        orchestrator.memory.write(
            finding(
                f"Event listener not removed on unmount in Component{i}",
                "leaks a listener each mount",
                severity="medium",
                paths=[f"src/Component{i}.tsx"],
            )
        )

    candidates = detect_promotions(orchestrator.ledger, orchestrator.memory, threshold=3)
    assert candidates and candidates[0].occurrences >= 4


def test_usage_reports_show_the_window_not_the_cumulative_total(orchestrator) -> None:
    """Two reports in a row must not double-count the first window's tokens."""
    from forge.kernel.events import EventType

    budget = orchestrator.models.budget
    budget.record(model="local", tier="local", hosted="local", cost=0.001,
                  input_tokens=1000, output_tokens=500, cached_tokens=0,
                  node_id="n1", task_class="implementation", escalation=False)
    orchestrator._report_usage("run1")

    budget.record(model="local", tier="local", hosted="local", cost=0.002,
                  input_tokens=300, output_tokens=200, cached_tokens=0,
                  node_id="n2", task_class="implementation", escalation=False)
    orchestrator._report_usage("run1")

    reports = orchestrator.ledger.read(types=[EventType.USAGE_REPORT])
    first = next(m for m in reports[0].payload["models"] if m["model"] == "local")
    second = next(m for m in reports[1].payload["models"] if m["model"] == "local")

    assert first["output_tokens"] == 500
    assert second["output_tokens"] == 200, "the second window must exclude the first"
    assert second["total_output_tokens"] == 700, "cumulative is still reported alongside"


def test_a_window_with_no_calls_reports_zero_not_the_last_window(orchestrator) -> None:
    from forge.kernel.events import EventType

    orchestrator.models.budget.record(
        model="local", tier="local", hosted="local", cost=0.001,
        input_tokens=100, output_tokens=50, cached_tokens=0,
        node_id="n1", task_class="implementation", escalation=False,
    )
    orchestrator._report_usage("run1")
    orchestrator._report_usage("run1")

    reports = orchestrator.ledger.read(types=[EventType.USAGE_REPORT])
    quiet = next(m for m in reports[1].payload["models"] if m["model"] == "local")
    assert quiet["calls"] == 0 and quiet["output_tokens"] == 0


# --------------------------------------------------------------------------
# Stopping a run that does not want to stop
# --------------------------------------------------------------------------


class HangingAgent(Agent):
    """A node stuck in a call that cannot be interrupted.

    Not hypothetical: a model request against a peer that had silently gone away
    held a worker for 45 minutes, and three Ctrl-Cs did not end the process.
    """

    kind = "t_hang"
    task_class = TaskClass.IMPLEMENTATION
    commits = False
    release = None  # type: ignore[var-annotated]

    def run(self, ctx: AgentContext) -> AgentResult:
        HangingAgent.release.wait(30)
        return AgentResult(success=True, summary="finally")


def test_a_stop_does_not_wait_forever_for_a_stuck_node(orchestrator: Orchestrator) -> None:
    """The shutdown grace is what turns `forge stop` back into something that works.

    Waiting for the lease -- 40 minutes on the pinball project -- means a process
    that looks alive and does nothing, and an operator whose only remaining tool
    is `kill -9`.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    HangingAgent.release = threading.Event()
    agent_registry.register(HangingAgent)
    orchestrator.config.scheduler.shutdown_grace = 0.2
    exits: list[int] = []
    orchestrator._force_exit = exits.append  # type: ignore[method-assign]

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(HangingAgent.release.wait, 30)
        stuck = orchestrator._drain({future: "node_stuck"}, pool)
        assert stuck == ["node_stuck"], "a node still running must be reported, not waited on"
    finally:
        HangingAgent.release.set()
        pool.shutdown(wait=False)


def test_a_node_that_finishes_inside_the_grace_is_waited_for(orchestrator: Orchestrator) -> None:
    """The bound must not have turned a clean shutdown into an abandoned one."""
    from concurrent.futures import ThreadPoolExecutor

    orchestrator.config.scheduler.shutdown_grace = 10.0
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(lambda: "done")
    assert orchestrator._drain({future: "node_ok"}, pool) == []


def test_automatic_drain_covers_the_bounded_opencode_attempt(
    orchestrator: Orchestrator,
) -> None:
    """Default shutdown drains a real local tool loop instead of abandoning it."""
    node = orchestrator.graph.add_node("implement", "broad local work")
    orchestrator.graph.update(node.id, tier="local_deep")
    orchestrator.config.scheduler.shutdown_grace = 0.0
    # The shared fixture pins the backend to native so no test reaches a real
    # model. This test is specifically about the OpenCode window, so it has to
    # ask for that backend rather than inherit whatever the host happens to have.
    orchestrator.config.coding.backend = "opencode"
    expected = (
        orchestrator.config.coding.opencode_timeout
        * orchestrator.config.coding.opencode_rounds
        + orchestrator.config.sandbox.command_timeout
    )

    assert orchestrator._drain_grace({object(): node.id}) >= expected


def test_a_second_signal_exits_immediately(orchestrator: Orchestrator) -> None:
    """An operator who presses Ctrl-C twice means now.

    With a handler installed there is no KeyboardInterrupt to fall back on, so
    without this the only remaining option is `kill -9` from another terminal.
    """
    import signal

    exits: list[int] = []
    orchestrator._force_exit = exits.append  # type: ignore[method-assign]
    orchestrator._install_signals()

    handler = signal.getsignal(signal.SIGTERM)
    handler(signal.SIGTERM, None)  # type: ignore[operator]
    assert orchestrator._stop.is_set()
    assert exits == [], "the first signal winds down; in-flight work gets to finish"

    handler(signal.SIGTERM, None)  # type: ignore[operator]
    assert exits == [130]


def test_quiet_time_ignores_the_heartbeat(config, clock) -> None:
    """Heartbeats continue at full rate while a run makes no progress at all.

    Counting them as activity is what let a wedged worker look healthy for 45
    minutes: full heartbeat, unchanged counts, "working" on the status line.
    """
    from forge.kernel.events import EventType

    with Orchestrator(config, clock=clock, worker_prefix="test") as orch:
        orch.ledger.emit(EventType.NODE_STARTED, node_id="n1")
        clock.advance(3600)
        orch.ledger.emit(EventType.RUN_HEARTBEAT, run_id="r1")
        orch.ledger.emit(EventType.NODE_LEASE_RENEWED, node_id="n1")

        assert orch._quiet_for() >= 3600


def test_quiet_time_resets_on_real_progress(config, clock) -> None:
    from forge.kernel.events import EventType

    with Orchestrator(config, clock=clock, worker_prefix="test") as orch:
        orch.ledger.emit(EventType.NODE_STARTED, node_id="n1")
        clock.advance(3600)
        orch.ledger.emit(EventType.GATE_PASSED, node_id="n1", gate="types")

        assert orch._quiet_for() < 1.0


class SloppyAgent(Agent):
    """Fails, having left a broken edit behind -- the common real-world shape."""

    kind = "t_sloppy"
    task_class = TaskClass.IMPLEMENTATION
    commits = True
    seen: ClassVar[list[str]] = []

    def run(self, ctx: AgentContext) -> AgentResult:
        target = ctx.root / "broken.txt"
        SloppyAgent.seen.append(target.read_text() if target.exists() else "<absent>")
        target.write_text(f"garbage from attempt {ctx.node.attempts}")
        return AgentResult.failure("did not pass validation", changed_files=["broken.txt"])


def test_a_failed_attempt_does_not_poison_the_next_one(orchestrator: Orchestrator) -> None:
    """Every attempt starts clean, and the *last* one cleans up after itself.

    `restore_for_attempt` handles the retries. The gap it cannot reach is the
    failure that blocks the node: no attempt N+1 runs, so nothing resets, and
    the abandoned edits stay in a tree other nodes keep building in until
    `_commit_result`'s dirty-tree branch sweeps them into someone else's commit.
    """
    agent_registry.register(SloppyAgent)
    SloppyAgent.seen = []
    _project(orchestrator)
    orchestrator.graph.add_node("t_sloppy", "write something broken")

    orchestrator.run(install_signal_handlers=False)

    assert len(SloppyAgent.seen) > 1, "the node should have been retried"
    assert set(SloppyAgent.seen) == {"<absent>"}, (
        f"a retry inherited the previous attempt's edits: {SloppyAgent.seen}"
    )
    # The part `restore_for_attempt` cannot do: after the attempt that blocks
    # the node, no further attempt runs, so nothing else would clean this up.
    # Including after the attempt that blocks: the tree is shared, and the
    # next node's commit would otherwise sweep up the abandoned edit.
    assert not (orchestrator.repo.path / "broken.txt").exists()
    assert not orchestrator.repo.is_dirty()


def test_the_quiet_threshold_follows_the_slowest_rung(orchestrator: Orchestrator) -> None:
    """Otherwise it is a constant that silently expires when a model is swapped."""
    slowest = max(
        orchestrator.models.registry.spec(name).timeout
        for name in orchestrator.config.models.ladder
    )
    assert orchestrator.status()["quiet_threshold"] > slowest, (
        "the threshold must clear one full call, or a healthy call trips it"
    )


class PlaceholderAgent(CodingAgent):
    """Asks for files and writes a throwaway alongside, as the prompt invites."""

    kind = "t_placeholder"
    replies: ClassVar[list[dict]] = []

    def run(self, ctx: AgentContext) -> AgentResult:
        return self.implement(ctx, "write the solver", include_paths=[])

    def ask(self, ctx, builder, task, schema=None, **kwargs):  # type: ignore[no-untyped-def]
        return PlaceholderAgent.replies.pop(0)


class OpenCodeCodingAgent(CodingAgent):
    kind = "t_opencode_coding"

    def run(self, ctx: AgentContext) -> AgentResult:
        return self.implement(ctx, "write the local OpenCode result", include_paths=[])


class ManifestAgent(Agent):
    """Adds a dependency so integration ordering can be asserted."""

    kind = "t_manifest"
    task_class = TaskClass.IMPLEMENTATION
    commits = True

    def run(self, ctx: AgentContext) -> AgentResult:
        (ctx.root / "package.json").write_text(
            '{"name":"integration-order","devDependencies":{"example":"1.0.0"}}\n'
        )
        return AgentResult(
            success=True,
            summary="declared a dependency",
            changed_files=["package.json"],
            commit_message="test: declare dependency",
        )


def test_integration_installs_new_dependencies_before_running_gates(
    orchestrator: Orchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dependency available in a node worktree must also exist on main.

    The live failure was subtle: focused lint passed after the model installed
    ``@types/node``, then integrated lint failed because main ran gates before
    installing the newly merged manifest.
    """
    import forge.kernel.orchestrator as orchestrator_module

    agent_registry.register(ManifestAgent)
    integrated_installs: list[Path] = []

    def record_install(sandbox, _toolchain, **_kwargs):  # type: ignore[no-untyped-def]
        if sandbox.exists("package.json") and "integration-order" in sandbox.read("package.json"):
            integrated_installs.append(sandbox.root)
        return None

    monkeypatch.setattr(orchestrator_module, "ensure_dependencies", record_install)
    run_gates = orchestrator.gates.run

    def assert_installed_then_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert orchestrator.repo.path in integrated_installs
        return run_gates(*args, **kwargs)

    monkeypatch.setattr(orchestrator.gates, "run", assert_installed_then_run)
    _project(orchestrator)
    node = orchestrator.graph.add_node("t_manifest", "add a dependency")

    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    assert orchestrator.graph.get(node.id).status == NodeStatus.SUCCEEDED
    assert orchestrator.repo.path in integrated_installs


@pytest.mark.parametrize("tier", ["local", "local_deep"])
def test_opencode_edits_are_gated_committed_and_counted_as_local(
    orchestrator: Orchestrator, monkeypatch: pytest.MonkeyPatch, tier: str
) -> None:
    agent_registry.register(OpenCodeCodingAgent)
    orchestrator.config.coding.backend = "opencode"
    orchestrator.config.coding.fallback_to_native = False

    monkeypatch.setattr(OpenCodeExecutor, "available", lambda _self: True)

    def execute(executor: OpenCodeExecutor, _prompt: str) -> OpenCodeResult:
        (executor.sandbox.root / "from-opencode.py").write_text("VALUE = 1\n")
        return OpenCodeResult(
            ok=True,
            summary="implemented through the local tool loop",
            session_id="ses_forge",
            usage=OpenCodeUsage(
                input_tokens=120,
                output_tokens=30,
                reasoning_tokens=10,
                steps=3,
                measured=True,
            ),
        )

    monkeypatch.setattr(OpenCodeExecutor, "execute", execute)

    _project(orchestrator)
    node = orchestrator.graph.add_node("t_opencode_coding", "use OpenCode")
    orchestrator.graph.update(node.id, tier=tier)
    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    done = orchestrator.graph.get(node.id)
    assert done.status == NodeStatus.SUCCEEDED
    assert done.result["data"]["backend"] == "opencode"
    assert (orchestrator.repo.path / "from-opencode.py").read_text() == "VALUE = 1\n"
    row = orchestrator.ledger.conn.execute(
        "SELECT hosted, input_tokens, output_tokens FROM spend WHERE node_id = ?",
        (node.id,),
    ).fetchone()
    assert dict(row) == {"hosted": "local", "input_tokens": 120, "output_tokens": 40}


def test_a_plan_that_asks_for_files_never_reaches_disk(orchestrator: Orchestrator) -> None:
    """The worst outcome of the session, reproduced.

    haiku asked for three files and wrote, as the prompt invites, a throwaway:
    `// Placeholder - will implement after reading existing files`. Nothing could
    be granted, so the throwaway was applied, committed, and passed all eight
    gates because nothing imports the file yet. The node was marked succeeded on
    a one-line stub against acceptance criteria demanding a swept collision
    solver with soak tests. A false success is worse than a failure: the graph
    moves on and builds on the stub.
    """
    agent_registry.register(PlaceholderAgent)
    placeholder = {
        "summary": "Implement the solver",
        "need_files": ["does/not/exist.ts"],  # nothing grantable, as it was live
        "edits": [
            {
                "path": "collide.ts",
                "op": "write",
                "content": "// Placeholder - will implement after reading existing files",
            }
        ],
    }
    PlaceholderAgent.replies = [dict(placeholder) for _ in range(MAX_FIX_ROUNDS + 2)]

    _project(orchestrator)
    orchestrator.graph.add_node("t_placeholder", "write the solver")
    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    written = orchestrator.repo.path / "collide.ts"
    assert not written.exists(), (
        f"a provisional placeholder reached disk: {written.read_text()!r}"
    )


def test_a_rollback_reopens_the_work_it_discarded(orchestrator: Orchestrator) -> None:
    """A node still marked succeeded after its commit is gone is a lie that
    dependents act on.

    `runnable()` selects on status, and dependencies are resolved by promotion at
    succeed-time rather than re-checked, so a rolled-back node's dependents stay
    runnable and build against code that no longer exists. Found by rolling back
    a node that had "succeeded" by committing a one-line placeholder: the
    workspace reverted correctly and the graph still said the work was done.
    """
    _project(orchestrator)
    before = orchestrator.checkpoints.create("before the work", kind="manual")

    first = orchestrator.graph.add_node("t_record", "the work")
    orchestrator.run(install_signal_handlers=False)
    assert orchestrator.graph.get(first.id).status == NodeStatus.SUCCEEDED

    # Added after the fact so it is promoted to ready by the completed dep but
    # has not run: the state where a stale success does real damage.
    second = orchestrator.graph.add_node("t_record", "depends on it", deps=[first.id])
    orchestrator.graph.promote_ready()
    assert orchestrator.graph.get(second.id).status == NodeStatus.READY, "precondition"

    orchestrator.checkpoints.rollback(before.id, reason="test")

    assert orchestrator.graph.get(first.id).status == NodeStatus.READY, (
        "the node whose commit was thrown away must be redone"
    )
    assert orchestrator.graph.get(second.id).status == NodeStatus.PENDING, (
        "its dependent must not stay runnable against work that no longer exists"
    )


def test_a_dependent_that_also_succeeded_is_demoted_not_left_runnable(
    orchestrator: Orchestrator,
) -> None:
    """The case the first version of this missed.

    Events arrive parent-first, so when A was reopened its dependent B -- which
    had also succeeded after the checkpoint -- was still SUCCEEDED, and the
    demotion only touched READY nodes. B's own turn then set it READY. Both
    became runnable at once, so B could start against a tree that no longer
    contains A's work, which is the exact failure this whole mechanism exists
    to prevent.
    """
    _project(orchestrator)
    before = orchestrator.checkpoints.create("before the work", kind="manual")

    first = orchestrator.graph.add_node("t_record", "the work")
    second = orchestrator.graph.add_node("t_record", "depends on it", deps=[first.id])
    orchestrator.run(install_signal_handlers=False)
    assert orchestrator.graph.get(second.id).status == NodeStatus.SUCCEEDED, "precondition"

    orchestrator.checkpoints.rollback(before.id, reason="test")

    assert orchestrator.graph.get(first.id).status == NodeStatus.READY
    assert orchestrator.graph.get(second.id).status == NodeStatus.PENDING, (
        "a dependent must wait for its dependency to be redone, not race it"
    )


class EmptyPlanAgent(CodingAgent):
    """Returns a plan with no edits and no file request: a non-answer."""

    kind = "t_empty"

    def run(self, ctx: AgentContext) -> AgentResult:
        return self.implement(ctx, "do the work", include_paths=[])

    def ask(self, ctx, builder, task, schema=None, **kwargs):  # type: ignore[no-untyped-def]
        return {"summary": "nothing to do", "edits": []}


def test_a_plan_with_no_edits_is_not_a_success(orchestrator: Orchestrator) -> None:
    """The false success arrived at from the other direction.

    `edits` may now be empty so a model asking for files need not invent a
    throwaway. But an empty plan that is *not* a request is a non-answer, and
    `apply_edits` accepts it happily: the gates then pass because nothing
    changed, and the node is marked succeeded having done nothing.
    """
    agent_registry.register(EmptyPlanAgent)
    _project(orchestrator)
    node = orchestrator.graph.add_node("t_empty", "do the work")

    orchestrator.run(install_signal_handlers=False)

    assert orchestrator.graph.get(node.id).status != NodeStatus.SUCCEEDED, (
        "a node that produced no edits must not be marked succeeded"
    )


class UntestedCodeAgent(CodingAgent):
    """Writes an implementation and no tests, whatever it is asked."""

    kind = "t_untested"
    calls: ClassVar[list[int]] = []

    def run(self, ctx: AgentContext) -> AgentResult:
        return self.implement(ctx, "implement the plunger", include_paths=[])

    def ask(self, ctx, builder, task, schema=None, **kwargs):  # type: ignore[no-untyped-def]
        UntestedCodeAgent.calls.append(1)
        return {
            "summary": "implement it",
            "edits": [{"path": "plunger.ts", "op": "write",
                       "content": f"export const charge = {len(UntestedCodeAgent.calls)};\n"}],
        }


def test_a_node_told_to_prove_itself_by_test_does_not_pass_without_one(
    orchestrator: Orchestrator,
) -> None:
    """Gates measure generic health, not whether the node did what it was asked.

    The plunger node's criteria said "asserted by test" three times. It shipped
    52 lines and no test, passed every gate, and was marked succeeded having
    implemented about a third of its criteria.
    """
    agent_registry.register(UntestedCodeAgent)
    UntestedCodeAgent.calls = []
    _project(orchestrator)
    orchestrator.graph.add_node(
        "t_untested",
        "implement the plunger",
        spec={"acceptance": ["A weak launch drains back - asserted by test."]},
    )

    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    assert len(UntestedCodeAgent.calls) > 1, (
        "passing gates with no test must not end the node on the first round"
    )


class ConsultingAgent(CodingAgent):
    """Fails its gates once, so the repair path runs and a consult is bought."""

    kind = "t_consult"
    asks: ClassVar[list[str]] = []
    advice_seen: ClassVar[list[bool]] = []

    def run(self, ctx: AgentContext) -> AgentResult:
        return self.implement(ctx, "make it work", include_paths=[])

    def ask(self, ctx, builder, task, schema=None, **kwargs):  # type: ignore[no-untyped-def]
        ConsultingAgent.asks.append(task[:40])
        # A consult asks for prose, not an edit plan.
        if schema is None:
            return "The brace on line 12 of thing.ts is unbalanced."
        rendered = "\n".join(s.name for s in builder._sections)
        ConsultingAgent.advice_seen.append("diagnosis" in rendered)
        return {
            "summary": "attempt",
            "edits": [{"path": "thing.ts", "op": "write",
                       "content": f"export const n = {len(ConsultingAgent.asks)};\n"}],
        }


def test_a_failing_round_buys_a_diagnosis_and_feeds_it_back(orchestrator: Orchestrator) -> None:
    """The consult step: the strong rung finds the fault, the local rung fixes it.

    Every node that landed today was *written* by a cloud model. Finding a fault
    is the thing the local rung cannot do -- it failed to locate one unbalanced
    brace across three repair rounds -- while applying a precise instruction is
    something it can. Buying the diagnosis instead of the code is what keeps the
    bulk of the tokens local.
    """
    from forge.validation.gate import Gate, GateContext, gate_registry
    from forge.validation.types import Verdict

    class FailsOnce(Gate):
        name = "fails_once"
        cacheable = False
        runs: ClassVar[list[int]] = []

        def run(self, ctx: GateContext) -> Verdict:
            FailsOnce.runs.append(1)
            if len(FailsOnce.runs) == 1:
                return Verdict.failing(
                    "fails_once", "unbalanced brace", evidence="thing.ts(12,1): error"
                )
            return Verdict(gate="fails_once", passed=True)

    gate_registry.register(FailsOnce)
    orchestrator.config.validation.gates = ["fails_once"]

    agent_registry.register(ConsultingAgent)
    ConsultingAgent.asks = []
    ConsultingAgent.advice_seen = []
    _project(orchestrator)
    orchestrator.graph.add_node("t_consult", "make it work")

    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    consults = [a for a in ConsultingAgent.asks if a.startswith("Diagnose")]
    assert consults, f"no consult was made; asks were {ConsultingAgent.asks}"
    assert any(ConsultingAgent.advice_seen), (
        "the diagnosis never reached the repairing model's context"
    )


class RateLimitedAgent(Agent):
    """Fails the way a provider outage fails: transiently, without being asked."""

    kind = "t_ratelimited"

    def run(self, ctx: AgentContext) -> AgentResult:
        from forge.errors import RateLimited

        raise RateLimited("upstream 529")


def test_an_outage_does_not_train_the_router_against_the_rung(
    orchestrator: Orchestrator,
) -> None:
    """The router records what the model deserves the blame for.

    A rate limit, a dropped connection or a bug in Forge's own code says nothing
    about whether a rung can do this task class -- it never got to try. Recorded
    as a failure anyway, an hour of provider instability permanently biases
    routing away from a rung on evidence it did not generate.
    """
    agent_registry.register(RateLimitedAgent)
    _project(orchestrator)
    orchestrator.graph.add_node("t_ratelimited", "will be rate limited")

    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    decisions = [
        e for e in orchestrator.ledger.read(types=["route.decided"])
        if e.payload.get("outcome") == "failure"
    ]
    assert not decisions, f"an outage was recorded as a model failure: {decisions}"


class RereadingAgent(CodingAgent):
    """Writes a file, then asks to read that same file, forever."""

    kind = "t_reread"
    calls: ClassVar[list[str]] = []

    def run(self, ctx: AgentContext) -> AgentResult:
        # Per attempt, not per run: the scheduler may retry the node, and a
        # bound that counted across attempts would move with the backoff.
        RereadingAgent.calls = []
        return self.implement(ctx, "make it work", include_paths=[])

    def ask(self, ctx, builder, task, schema=None, **kwargs):  # type: ignore[no-untyped-def]
        RereadingAgent.calls.append(task[:20])
        if len(RereadingAgent.calls) > 15:
            raise RuntimeError("the re-read loop is unbounded")
        if schema is None:
            return "advice"
        if len(RereadingAgent.calls) == 1:
            return {"summary": "first", "edits": [
                {"path": "thing.ts", "op": "write", "content": "export const n = 1;\n"}]}
        return {"summary": "again", "edits": [], "need_files": ["thing.ts"]}


def test_a_model_that_keeps_asking_to_re_read_its_own_file_is_stopped(
    orchestrator: Orchestrator,
) -> None:
    """The re-read exemption had no counter.

    Re-reading a file this attempt wrote is continuity rather than discovery, so
    it rightly costs neither a grant nor a round. But serving one adds nothing
    the model could not already see -- the file is pinned and on disk -- so the
    next prompt is byte-identical and the ask repeats. With the round counter
    rolled back each time, that is a full-context call per turn, forever, and
    the node never fails either.
    """
    from forge.validation.gate import Gate, GateContext, gate_registry
    from forge.validation.types import Verdict

    class Broken(Gate):
        name = "broken"
        cacheable = False

        def run(self, ctx: GateContext) -> Verdict:
            return Verdict.failing("broken", "still failing", evidence="thing.ts(1,1): TS1005")

    gate_registry.register(Broken)
    orchestrator.config.validation.gates = ["broken"]
    agent_registry.register(RereadingAgent)
    RereadingAgent.calls = []
    _project(orchestrator)
    orchestrator.graph.add_node("t_reread", "make it work")

    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    assert len(RereadingAgent.calls) <= 8, (
        f"{len(RereadingAgent.calls)} model calls in a single attempt"
    )


class StaleDigestAgent(CodingAgent):
    """Fails the gates, then submits an unappliable plan.

    Two different failures in a row, which is what separates "the digest
    describes the current problem" from "the digest is whatever we last made".
    """

    kind = "t_stale"
    prompts: ClassVar[list[str]] = []

    def run(self, ctx: AgentContext) -> AgentResult:
        return self.implement(ctx, "make it work", include_paths=[])

    def ask(self, ctx, builder, task, schema=None, **kwargs):  # type: ignore[no-untyped-def]
        if schema is None:
            return "thing.ts:12 - THE DIGESTED TYPE ERROR."
        body = "\n".join(s.name + s.content for s in builder._sections)
        StaleDigestAgent.prompts.append(body)
        if len(StaleDigestAgent.prompts) == 1:
            return {"summary": "first", "edits": [
                {"path": "thing.ts", "op": "write", "content": "export const n = 1;\n"}]}
        # An anchor that is not in the file: rejected by the patcher, so the
        # gates never run and no new digest is made.
        return {"summary": "second", "edits": [
            {"path": "thing.ts", "op": "replace", "anchor": "NO SUCH ANCHOR",
             "content": "x"}]}


def test_a_digest_of_an_older_failure_is_not_shown_for_a_newer_one(
    orchestrator: Orchestrator,
) -> None:
    """`digest or report.render()` preferred a digest that was never refreshed.

    It is only recomputed after a gate run, and every other failure path --
    patch rejection, empty plan, an unmet acceptance criterion -- reaches the
    next round by `continue`. So round three was shown round one's digest under
    the heading "What the checks reported", and never saw the anchor mismatch
    that was actually blocking it. The model is then asked to fix a failure that
    is no longer happening.
    """
    from forge.validation.gate import Gate, GateContext, gate_registry
    from forge.validation.types import Verdict

    class Typed(Gate):
        name = "typed"
        cacheable = False
        runs: ClassVar[list[int]] = []

        def run(self, ctx: GateContext) -> Verdict:
            Typed.runs.append(1)
            return Verdict.failing("typed", "tsc failed", evidence="thing.ts(12,1): TS1005")

    gate_registry.register(Typed)
    Typed.runs = []
    orchestrator.config.validation.gates = ["typed"]
    agent_registry.register(StaleDigestAgent)
    StaleDigestAgent.prompts = []
    _project(orchestrator)
    orchestrator.graph.add_node("t_stale", "make it work")

    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    assert len(StaleDigestAgent.prompts) >= 3, "the third round must have happened"
    third = StaleDigestAgent.prompts[2]
    assert "anchor not found" in third, "the failure it must fix is the patch rejection"
    assert "THE DIGESTED TYPE ERROR" not in third, (
        "a digest of the previous, different failure was presented as the current one"
    )


class DigestingAgent(CodingAgent):
    """Fails once against a wall of compiler output, so the digest path runs."""

    kind = "t_digest"
    asks: ClassVar[list[str]] = []
    saw_raw_cascade: ClassVar[list[bool]] = []

    def run(self, ctx: AgentContext) -> AgentResult:
        return self.implement(ctx, "make it work", include_paths=[])

    def ask(self, ctx, builder, task, schema=None, **kwargs):  # type: ignore[no-untyped-def]
        DigestingAgent.asks.append(task[:30])
        if schema is None:
            if task.startswith("Summarise"):
                return "thing.ts:12 - unbalanced brace. The other 400 are consequences."
            return "The brace on line 12 is unbalanced."
        body = "\n".join(s.name + s.content for s in builder._sections)
        DigestingAgent.saw_raw_cascade.append("cascade error 300" in body)
        return {"summary": "attempt", "edits": [
            {"path": "thing.ts", "op": "write", "content": "export const n = 1;\n"}]}


def test_raw_compiler_output_is_digested_locally_not_pasted_into_the_repair(
    orchestrator: Orchestrator,
) -> None:
    """Four hundred lines of cascade must not compete for the repair budget.

    Truncating and hoping the surviving half holds the cause is a guess. Reading
    all of it in a separate local call and forwarding only the signal is not, and
    locally it costs nothing.
    """
    from forge.validation.gate import Gate, GateContext, gate_registry
    from forge.validation.types import Verdict

    cascade = "\n".join(f"src/other{i}.ts: cascade error {i}" for i in range(400))

    class Noisy(Gate):
        name = "noisy"
        cacheable = False
        runs: ClassVar[list[int]] = []

        def run(self, ctx: GateContext) -> Verdict:
            Noisy.runs.append(1)
            if len(Noisy.runs) == 1:
                return Verdict.failing(
                    "noisy", "tsc failed",
                    evidence=f"thing.ts(12,1): error TS1005\n{cascade}",
                )
            return Verdict(gate="noisy", passed=True)

    gate_registry.register(Noisy)
    orchestrator.config.validation.gates = ["noisy"]
    agent_registry.register(DigestingAgent)
    DigestingAgent.asks = []
    DigestingAgent.saw_raw_cascade = []
    _project(orchestrator)
    orchestrator.graph.add_node("t_digest", "make it work")

    orchestrator.run(max_nodes=1, install_signal_handlers=False)

    assert any(a.startswith("Summarise") for a in DigestingAgent.asks), (
        f"no digest was made; asks were {DigestingAgent.asks}"
    )
    assert not any(DigestingAgent.saw_raw_cascade), (
        "the raw cascade reached the repair prompt despite a digest being available"
    )


class FindingFixAgent(Agent):
    """Succeeds immediately, standing in for work that closes a finding."""

    kind = "t_finding_fix"

    def run(self, ctx: AgentContext) -> AgentResult:
        return AgentResult(success=True, summary="fixed")


def test_closing_the_work_closes_the_finding(orchestrator: Orchestrator) -> None:
    """`resolve_finding` shipped with no callers at all.

    Findings were created and never closed, so a real project reached 36 open
    findings, most of them long since fixed, and nothing downstream could
    distinguish an outstanding defect from a stale record.
    """
    from forge.memory.store import finding

    agent_registry.register(FindingFixAgent)
    _project(orchestrator)
    record = finding("Missing drain", "the ball never drains", severity="critical")
    orchestrator.memory.write_many([record])
    assert [r.title for r in orchestrator.memory.open_findings()] == ["Missing drain"]

    orchestrator.graph.add_node(
        "t_finding_fix", "Address: Missing drain", spec={"resolves_finding": record.id}
    )
    orchestrator.run(install_signal_handlers=False)

    assert orchestrator.memory.open_findings() == []


def test_a_drained_project_with_open_findings_is_re_opened(
    orchestrator: Orchestrator,
) -> None:
    """A quiescent graph plus an unresolved reviewer verdict is a dead end.

    `forge run` on a graph with nothing to do returns immediately, which is
    correct when the project is finished. A real project reached 100% and
    "quiescent" with four critical review findings still open, and there was no
    supported way back in: `forge tell` writes a memory record but creates no
    node, so nothing ever read it.
    """
    from forge.memory.store import finding

    _project(orchestrator)
    assert not [
        n for n in orchestrator.graph.all_nodes()
        if n.status in (NodeStatus.PENDING, NodeStatus.READY, NodeStatus.RUNNING)
    ]

    orchestrator.memory.write_many([
        finding("Presentation Layer Unapplied", "no lighting or ball shadow",
                severity="critical", source="visual:node_x")
    ])
    orchestrator._reopen_if_unfinished()

    reopened = [n for n in orchestrator.graph.all_nodes() if n.spec.get("reopened")]
    assert len(reopened) == 1
    assert reopened[0].kind == "goal"


def test_a_drained_project_with_nothing_open_stays_finished(
    orchestrator: Orchestrator,
) -> None:
    """Re-opening on every start would make a finished project never finish."""
    from forge.memory.store import finding

    _project(orchestrator)
    orchestrator.memory.write_many([
        # Resolved, and a stale gate record: neither contradicts "delivered".
        finding("types: error TS6133: 'add' is unused", "", severity="high",
                source="gate:types"),
    ])
    orchestrator._reopen_if_unfinished()

    assert not [n for n in orchestrator.graph.all_nodes() if n.spec.get("reopened")]


# --------------------------------------------------------------------------
# Environmental failure: an endpoint that stops answering
# --------------------------------------------------------------------------


def _running_node(orchestrator: Orchestrator) -> None:
    """Put one node into RUNNING so a silent window is suspicious.

    The shared fixture makes every provider an echo stub; a stub has no endpoint
    to probe, so this scenario needs the local one to look like real HTTP.
    """
    orchestrator.config.models.providers["local"].kind = "openai_compat"
    _project(orchestrator)
    node = orchestrator.graph.add_node(
        kind="implement", title="something long-running", spec={}
    )
    orchestrator.graph.start(node.id, tier="local", worker_id="test-worker")


def test_a_silent_window_with_a_running_node_probes_the_endpoint(
    orchestrator: Orchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this exists for: the server went away and nothing said so.

    A run whose model endpoint stops answering does not fail, it waits. The
    executor blocks on its socket and the only symptom is a usage report with
    no calls in it -- which is also what a model mid-generation looks like.
    """
    from forge.kernel import orchestrator as orch_module

    _running_node(orchestrator)
    monkeypatch.setattr(
        orch_module, "probe_provider", lambda provider, **kw: (False, "http://nope/v1 unreachable")
    )

    orchestrator._check_silent_window("run-1")

    warning = orchestrator._latest_warning()
    assert warning is not None
    assert warning["kind"] == "model_endpoint_unreachable"
    assert "unreachable" in warning["detail"]


def test_a_silent_window_with_nothing_running_is_not_an_error(
    orchestrator: Orchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idle is not broken. Probing here would cry wolf on every quiet project."""
    from forge.kernel import orchestrator as orch_module

    orchestrator.config.models.providers["local"].kind = "openai_compat"
    _project(orchestrator)
    called = []
    monkeypatch.setattr(
        orch_module,
        "probe_provider",
        lambda provider, **kw: called.append(1) or (False, "unreachable"),
    )

    orchestrator._check_silent_window("run-1")

    assert not called
    assert orchestrator._latest_warning() is None


def test_the_warning_clears_once_the_endpoint_answers_again(
    orchestrator: Orchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recovered endpoint must stop explaining a stall that no longer exists."""
    from forge.kernel import orchestrator as orch_module

    _running_node(orchestrator)
    monkeypatch.setattr(
        orch_module, "probe_provider", lambda provider, **kw: (False, "unreachable")
    )
    orchestrator._check_silent_window("run-1")
    assert orchestrator._latest_warning() is not None

    monkeypatch.setattr(
        orch_module, "probe_provider", lambda provider, **kw: (True, "responded 200")
    )
    orchestrator._check_silent_window("run-1")

    assert orchestrator._latest_warning() is None
