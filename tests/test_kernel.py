"""Kernel behaviour: the ledger, the graph, leases, and recovery.

These are the tests that matter most. Everything above the kernel can fail and
be retried; if the kernel loses or duplicates work, nothing above it can help.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.errors import ConcurrencyError, InvariantError
from forge.kernel.events import Event, EventType
from forge.kernel.graph import NodeKind, NodeStatus, TaskGraph
from forge.kernel.ledger import Ledger
from forge.kernel.scheduler import Disposition, Scheduler
from forge.util.clock import ManualClock

# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------


def test_append_assigns_monotonic_sequence(ledger: Ledger) -> None:
    first = ledger.emit("test.one", value=1)
    second = ledger.emit("test.two", value=2)
    assert second.seq > first.seq
    assert [e.type for e in ledger.read()] == ["test.one", "test.two"]


def test_duplicate_event_id_is_rejected(ledger: Ledger) -> None:
    event = Event(type="test.dup", id="evt_fixed")
    ledger.append(event)
    with pytest.raises(ConcurrencyError):
        ledger.append(Event(type="test.dup", id="evt_fixed"))


def test_append_many_is_atomic(ledger: Ledger) -> None:
    before = ledger.head_seq()
    with pytest.raises(ConcurrencyError):
        ledger.append_many(
            [Event(type="a", id="evt_x"), Event(type="b", id="evt_x")]  # same id twice
        )
    assert ledger.head_seq() == before, "a failed batch must leave no partial writes"


def test_kv_compare_and_swap(ledger: Ledger) -> None:
    version = ledger.kv_set("k", {"v": 1})
    with pytest.raises(ConcurrencyError):
        ledger.kv_set("k", {"v": 2}, expected_version=version + 5)
    ledger.kv_set("k", {"v": 2}, expected_version=version)
    assert ledger.kv_get("k") == {"v": 2}


def test_projections_rebuild_exactly(ledger: Ledger, clock: ManualClock) -> None:
    """The whole event-sourcing bet: derived state is always recomputable."""
    graph = TaskGraph(ledger, "proj_test", clock)
    a = graph.add_node(NodeKind.IMPLEMENT, "first")
    b = graph.add_node(NodeKind.REVIEW, "second", deps=[a.id])
    graph.succeed(a.id, {"ok": True})

    before = {n.id: (n.status, n.attempts, n.deps) for n in graph.all_nodes()}
    ledger.rebuild_projections()
    after = {n.id: (n.status, n.attempts, n.deps) for n in graph.all_nodes()}
    assert before == after
    assert after[b.id][0] == NodeStatus.READY


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


def test_dependencies_gate_readiness(ledger: Ledger, clock: ManualClock) -> None:
    graph = TaskGraph(ledger, "proj_test", clock)
    a = graph.add_node(NodeKind.IMPLEMENT, "a")
    b = graph.add_node(NodeKind.IMPLEMENT, "b", deps=[a.id])

    assert graph.get(a.id).status == NodeStatus.READY
    assert graph.get(b.id).status == NodeStatus.PENDING
    assert [n.id for n in graph.runnable()] == [a.id]

    graph.succeed(a.id)
    assert graph.get(b.id).status == NodeStatus.READY


def test_cycles_are_refused_at_insert(ledger: Ledger, clock: ManualClock) -> None:
    graph = TaskGraph(ledger, "proj_test", clock)
    a = graph.add_node(NodeKind.IMPLEMENT, "a")
    b = graph.add_node(NodeKind.IMPLEMENT, "b", deps=[a.id])
    with pytest.raises(InvariantError):
        graph.add_node(NodeKind.IMPLEMENT, "c", deps=[b.id], node_id=a.id)


def test_add_many_resolves_index_deps_and_drops_self_edges(ledger: Ledger, clock: ManualClock) -> None:
    graph = TaskGraph(ledger, "proj_test", clock)
    created = graph.add_many(
        [
            {"kind": "implement", "title": "one", "deps": [0]},  # self-reference
            {"kind": "review", "title": "two", "deps": [0]},
        ]
    )
    assert created[0].deps == []
    assert created[1].deps == [created[0].id]


def test_terminal_failure_cascades_to_dependents(ledger: Ledger, clock: ManualClock) -> None:
    graph = TaskGraph(ledger, "proj_test", clock)
    a = graph.add_node(NodeKind.IMPLEMENT, "a")
    b = graph.add_node(NodeKind.REVIEW, "b", deps=[a.id])
    c = graph.add_node(NodeKind.DOCUMENT, "c", deps=[b.id])

    graph.fail(a.id, "unrecoverable", terminal=True)
    assert graph.get(b.id).status == NodeStatus.BLOCKED
    assert graph.get(c.id).status == NodeStatus.BLOCKED, "blocking must be transitive"


def test_barrier_nodes_wait_for_everything_else(ledger: Ledger, clock: ManualClock) -> None:
    graph = TaskGraph(ledger, "proj_test", clock)
    work = graph.add_node(NodeKind.IMPLEMENT, "work")
    barrier = graph.add_node(NodeKind.GOAL, "goal", spec={"barrier": True}, priority=999)

    assert [n.id for n in graph.runnable()] == [work.id]
    graph.succeed(work.id)
    assert [n.id for n in graph.runnable()] == [barrier.id]


def test_critical_path_follows_the_longest_unfinished_chain(ledger: Ledger, clock: ManualClock) -> None:
    graph = TaskGraph(ledger, "proj_test", clock)
    a = graph.add_node(NodeKind.IMPLEMENT, "a")
    b = graph.add_node(NodeKind.IMPLEMENT, "b", deps=[a.id])
    graph.add_node(NodeKind.IMPLEMENT, "c", deps=[b.id])
    graph.add_node(NodeKind.IMPLEMENT, "short", deps=[a.id])

    path = [n.title for n in graph.critical_path()]
    assert path == ["a", "b", "c"]


# --------------------------------------------------------------------------
# Leases
# --------------------------------------------------------------------------


def test_only_one_worker_can_claim_a_node(ledger: Ledger, clock: ManualClock) -> None:
    graph = TaskGraph(ledger, "proj_test", clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "contended")

    graph.claim(node.id, "worker-a", 60)
    with pytest.raises(ConcurrencyError):
        graph.claim(node.id, "worker-b", 60)


def test_expired_lease_returns_the_node_to_the_queue(ledger: Ledger, clock: ManualClock) -> None:
    """The crash-recovery primitive: a dead worker's node becomes runnable."""
    graph = TaskGraph(ledger, "proj_test", clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "orphaned")
    graph.claim(node.id, "doomed-worker", 60)
    assert graph.get(node.id).status == NodeStatus.RUNNING
    assert graph.runnable() == []

    clock.advance(61)
    reaped = graph.reap_expired_leases()

    assert reaped == [node.id]
    assert graph.get(node.id).status == NodeStatus.READY
    assert [n.id for n in graph.runnable()] == [node.id]


def test_stolen_lease_is_detected_by_the_original_holder(ledger: Ledger, clock: ManualClock) -> None:
    from forge.errors import LeaseLost

    graph = TaskGraph(ledger, "proj_test", clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "stolen")
    lease = graph.claim(node.id, "worker-a", 60)

    clock.advance(61)
    graph.reap_expired_leases()
    graph.claim(node.id, "worker-b", 60)

    with pytest.raises(LeaseLost):
        graph.verify_lease(lease)


def test_attempts_survive_lease_loss(ledger: Ledger, clock: ManualClock) -> None:
    """Retries must stay bounded across crashes, or a bad node loops forever."""
    graph = TaskGraph(ledger, "proj_test", clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "flaky")

    graph.claim(node.id, "w1", 60)
    graph.start(node.id, tier="local", worker_id="w1")
    clock.advance(61)
    graph.reap_expired_leases()

    graph.claim(node.id, "w2", 60)
    graph.start(node.id, tier="local", worker_id="w2")
    assert graph.get(node.id).attempts == 2


# --------------------------------------------------------------------------
# Scheduler policy
# --------------------------------------------------------------------------


def _scheduler(ledger: Ledger, clock: ManualClock) -> tuple[Scheduler, TaskGraph]:
    from forge.config import SchedulerConfig

    graph = TaskGraph(ledger, "proj_test", clock)
    config = SchedulerConfig(max_attempts=4, escalate_after_attempts=2, backoff_base=1.0)
    return Scheduler(graph, config, clock=clock, ladder=["local", "local_deep", "frontier"], seed=1), graph


def test_transient_failures_do_not_escalate(ledger: Ledger, clock: ManualClock) -> None:
    from forge.errors import ProviderUnavailable

    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")
    graph.start(node.id, tier="local", worker_id="w")

    plan = scheduler.plan_failure(graph.get(node.id), ProviderUnavailable("connection refused"))
    assert plan.disposition is Disposition.RETRY
    assert "transient" in plan.reason


def test_an_integration_conflict_does_not_buy_a_stronger_model(
    ledger: Ledger, clock: ManualClock
) -> None:
    """Git concurrency is not evidence that the local model cannot code."""
    from forge.errors import ConcurrencyError

    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")
    graph.start(node.id, tier="local", worker_id="w")

    plan = scheduler.plan_failure(
        graph.get(node.id),
        ConcurrencyError("isolated work conflicts with newer integrated work"),
    )

    assert plan.disposition is Disposition.RETRY
    assert plan.refund_attempt
    assert plan.escalate_to is None


def test_repeated_failure_escalates_up_the_ladder(ledger: Ledger, clock: ManualClock) -> None:
    from forge.errors import MalformedOutput

    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")
    for _ in range(2):
        graph.start(node.id, tier="local", worker_id="w")

    plan = scheduler.plan_failure(graph.get(node.id), MalformedOutput("wrong output"))
    assert plan.disposition is Disposition.ESCALATE
    assert plan.escalate_to == "local_deep"


def test_a_platform_bug_does_not_promote_the_node_to_a_pricier_model(
    ledger: Ledger, clock: ManualClock
) -> None:
    """An exception from Forge's own code is not evidence that the model is weak.

    Measured on the pinball run: attempts consumed by four platform defects --
    a schema bug, a retry storm, an impossible output budget, a broken HTTP
    handler -- escalated one node to the most expensive rung and kept it there,
    against a cloud-spend target of 18%.
    """
    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")
    for _ in range(2):  # already past escalate_after_attempts
        graph.start(node.id, tier="local", worker_id="w")

    plan = scheduler.plan_failure(graph.get(node.id), AttributeError("bug in Forge itself"))
    assert plan.disposition is Disposition.RETRY
    assert "platform fault" in plan.reason


def test_a_permanently_broken_platform_still_stops(ledger: Ledger, clock: ManualClock) -> None:
    """Retrying our own bug forever would be worse than blocking."""
    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")
    for _ in range(4):
        graph.start(node.id, tier="local", worker_id="w")

    plan = scheduler.plan_failure(graph.get(node.id), AttributeError("bug in Forge itself"))
    assert plan.disposition is Disposition.BLOCK


def test_exhausted_attempts_block_rather_than_loop(ledger: Ledger, clock: ManualClock) -> None:
    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")
    for _ in range(4):
        graph.start(node.id, tier="frontier", worker_id="w")

    plan = scheduler.plan_failure(graph.get(node.id), ValueError("still broken"))
    assert plan.disposition is Disposition.BLOCK

    scheduler.apply(graph.get(node.id), plan, "still broken")
    blocked = graph.get(node.id)
    assert blocked.status == NodeStatus.BLOCKED
    assert "acceptance" in (blocked.result or {}).get("question", "").lower() or blocked.result


def test_same_failure_blocks_after_every_strategy_is_exhausted(
    ledger: Ledger, clock: ManualClock
) -> None:
    from forge.config import SchedulerConfig

    graph = TaskGraph(ledger, "proj_test", clock)
    scheduler = Scheduler(
        graph,
        SchedulerConfig(max_attempts=50, max_no_progress_attempts=3),
        clock=clock,
        ladder=["local", "local_deep", "opus"],
    )
    node = graph.add_node(
        NodeKind.IMPLEMENT,
        "hard task",
        spec={
            "decomposed": True,
            "_same_failure_count": 3,
            "_strategies_tried": ["local", "local_deep", "coach", "decompose", "cloud-solve"],
        },
    )
    graph.update(node.id, tier="opus")

    plan = scheduler.plan_failure(graph.get(node.id), ValueError("same deterministic error"))

    assert plan.disposition is Disposition.BLOCK
    assert "practically unsolvable" in plan.reason


def test_unrecoverable_errors_are_not_retried(ledger: Ledger, clock: ManualClock) -> None:
    from forge.errors import ConfigError

    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")
    plan = scheduler.plan_failure(graph.get(node.id), ConfigError("no such gate"))
    assert plan.disposition is Disposition.BLOCK


def test_backoff_grows_and_is_bounded(ledger: Ledger, clock: ManualClock) -> None:
    scheduler, _ = _scheduler(ledger, clock)
    delays = [scheduler.backoff(i) for i in range(1, 8)]
    assert delays[0] < delays[3]
    assert all(d <= scheduler.config.backoff_max * 1.3 for d in delays)


def test_deferred_node_is_not_runnable_until_its_time(ledger: Ledger, clock: ManualClock) -> None:
    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")
    graph.defer(node.id, clock.now() + 30)

    assert scheduler.next_node() is None
    clock.advance(31)
    assert scheduler.next_node() is not None


def test_metrics_sink_is_never_fatal(ledger: Ledger) -> None:
    ledger.record_metric("x", 1.0, "counter", {"a": "b"})
    ledger.record_metric("x", 2.0, "counter", {"a": "b"})
    rows = {row["name"]: row for row in ledger.metrics_snapshot()}
    assert rows["x"]["count"] == 2
    assert rows["x"]["total"] == 3.0


def test_event_projection_ignores_unknown_types(ledger: Ledger) -> None:
    ledger.emit("node.some_future_event", node_id="node_x")
    ledger.emit(EventType.METRIC, name="x")
    # Rebuilding must tolerate types this version does not understand.
    ledger.rebuild_projections()


def test_the_block_reason_names_the_budget_not_just_the_count(
    ledger: Ledger, clock: ManualClock
) -> None:
    """"exhausted 14 attempts" against a max_attempts of 4 reads as nonsense.

    The counter is cumulative across operator unblocks, so it routinely exceeds
    the limit. The message has to say which number is the budget.
    """
    from forge.errors import MalformedOutput

    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "hard thing")
    graph.update(node.id, attempts=14)
    plan = scheduler.plan_failure(graph.get(node.id), MalformedOutput("nope"))

    assert plan.disposition is Disposition.BLOCK
    assert "4-attempt budget" in plan.reason
    assert "14 attempts on this node in total" in plan.reason


def test_answering_a_blocked_node_restarts_the_local_strategy_cycle(
    ledger: Ledger, clock: ManualClock
) -> None:
    """Guidance must get a full local retry, not resume a sticky cloud tier.

    `plan_failure` compares the lifetime attempt count against `max_attempts`, so
    a node unblocked at 14 attempts re-blocked on its first failure -- and the
    operator's answer was never given a real chance.  Likewise, retaining Opus
    skipped the local/OpenCode worker that should apply the new instruction.
    """
    from forge.errors import MalformedOutput

    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "hard thing")
    graph.update(
        node.id,
        attempts=14,
        tier="frontier",
        spec={
            "_failure_signature": "old",
            "_same_failure_count": 4,
            "_strategies_tried": ["local", "coach", "cloud-solve"],
        },
    )

    # What `forge unblock` does.
    spec = dict(graph.get(node.id).spec)
    for key in ("_failure_signature", "_same_failure_count", "_strategies_tried"):
        spec.pop(key, None)
    graph.update(
        node.id,
        status="ready",
        attempts=0,
        tier="local",
        spec=spec,
        not_before=0.0,
        actor="human",
    )

    fresh = graph.get(node.id)
    assert fresh.tier == "local"
    assert not any(key.startswith("_failure") or key == "_strategies_tried" for key in fresh.spec)
    plan = scheduler.plan_failure(fresh, MalformedOutput("nope"))
    assert plan.disposition is not Disposition.BLOCK, (
        "a freshly unblocked node must get a real retry budget, not one last chance"
    )


def test_a_block_explains_that_the_stronger_rungs_were_priced_out(
    ledger: Ledger, clock: ManualClock
) -> None:
    """"try a stronger model" is not advice when no stronger model is reachable.

    Past `budget.per_node_cost` every cloud rung is unaffordable, so each retry
    re-runs the strongest *local* model while the route reason still says
    "climbing one rung". One node spent 16 attempts and 12.43 against an 8.00
    ceiling that way, on a syntax error, and nothing mentioned money.
    """
    from forge.config import SchedulerConfig
    from forge.errors import MalformedOutput

    graph = TaskGraph(ledger, "proj_test", clock)
    scheduler = Scheduler(
        graph, SchedulerConfig(max_attempts=2), clock=clock, per_node_cost=8.0
    )
    node = graph.add_node(NodeKind.IMPLEMENT, "expensive thing")
    graph.update(node.id, attempts=2)
    # Cost accumulates from real spend, so it is not settable by hand.
    ledger.append(
        Event(type=EventType.BUDGET_SPENT, node_id=node.id, payload={"cost": 12.43})
    )

    fresh = graph.get(node.id)
    plan = scheduler.plan_failure(fresh, MalformedOutput("bad"))
    scheduler.apply(fresh, plan, "still broken")

    question = graph.get(node.id).result["question"]
    assert "12.43" in question and "8.00" in question
    assert "per_node_cost" in question, "it must name the knob that would change this"


def test_a_node_inside_its_budget_gets_no_such_note(ledger: Ledger, clock: ManualClock) -> None:
    from forge.config import SchedulerConfig
    from forge.errors import MalformedOutput

    graph = TaskGraph(ledger, "proj_test", clock)
    scheduler = Scheduler(
        graph, SchedulerConfig(max_attempts=2), clock=clock, per_node_cost=8.0
    )
    node = graph.add_node(NodeKind.IMPLEMENT, "cheap thing")
    graph.update(node.id, attempts=2)
    ledger.append(
        Event(type=EventType.BUDGET_SPENT, node_id=node.id, payload={"cost": 0.5})
    )

    fresh = graph.get(node.id)
    scheduler.apply(fresh, scheduler.plan_failure(fresh, MalformedOutput("bad")), "broken")
    assert "per-node ceiling" not in graph.get(node.id).result["question"]


def test_a_persistent_outage_backs_off_instead_of_spinning(
    ledger: Ledger, clock: ManualClock
) -> None:
    """A subscription rate limit spun for ninety minutes at ~20s intervals.

    Transient failures rightly skip the attempt budget, but the backoff was
    capped at `min(attempts, 3)`, so a genuine outage produced ~270 identical
    retries. The delay must grow with the outage.
    """
    from forge.errors import RateLimited

    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")

    # An outage is a run of consecutive transient failures, which is what the
    # backoff must grow with. It cannot be measured by `node.attempts` any more:
    # those are refunded precisely so an outage leaves the node's budget alone.
    graph.start(node.id, tier="local", worker_id="w")
    early = scheduler.plan_failure(graph.get(node.id), RateLimited("plan limit"))
    for _ in range(9):
        graph.start(node.id, tier="local", worker_id="w")
        late = scheduler.plan_failure(graph.get(node.id), RateLimited("plan limit"))

    assert late.delay > early.delay * 5, (
        f"backoff did not grow with the outage: {early.delay:.1f}s then {late.delay:.1f}s"
    )


def test_an_outage_that_never_clears_parks_the_node(ledger: Ledger, clock: ManualClock) -> None:
    """Otherwise nothing ever tells the operator the run has stopped progressing."""
    from forge.config import SchedulerConfig
    from forge.errors import RateLimited

    graph = TaskGraph(ledger, "proj_test", clock)
    config = SchedulerConfig(max_attempts=4, max_transient_attempts=5, backoff_base=1.0)
    scheduler = Scheduler(graph, config, clock=clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")

    for _ in range(5):
        graph.start(node.id, tier="local", worker_id="w")
        plan = scheduler.plan_failure(graph.get(node.id), RateLimited("plan limit"))

    assert plan.disposition is Disposition.BLOCK
    assert "RateLimited" in plan.reason
    assert "no stronger model" in plan.reason, "it must say escalation cannot help"


def test_an_outage_does_not_spend_the_attempt_budget(ledger: Ledger, clock: ManualClock) -> None:
    """"Transient failures do not consume an attempt" was only true in here.

    `graph.start` charges one on every claim, and `plan_failure` compares that
    same counter against `max_attempts`. So ten minutes of provider 529s left
    the node at its ceiling, and the first genuine failure after the provider
    recovered blocked it instantly -- no retry, no escalation, over an outage it
    had no part in. The refund is what makes the exemption real.
    """
    from forge.errors import RateLimited

    scheduler, graph = _scheduler(ledger, clock)  # max_attempts=4
    node = graph.add_node(NodeKind.IMPLEMENT, "n")

    for _ in range(6):
        graph.start(node.id, tier="local", worker_id="w")
        plan = scheduler.plan_failure(graph.get(node.id), RateLimited("upstream 529"))
        scheduler.apply(graph.get(node.id), plan, "rate limited")

    assert graph.get(node.id).attempts <= 1, "an outage must leave the budget alone"

    # And the node is still workable afterwards.
    graph.start(node.id, tier="local", worker_id="w")
    after = scheduler.plan_failure(graph.get(node.id), None)
    assert after.disposition is not Disposition.BLOCK, (
        "a first real failure after an outage must still get its retries"
    )


def test_a_genuine_failure_still_spends_one(ledger: Ledger, clock: ManualClock) -> None:
    """The refund is narrow: only the environment's failures are free."""
    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")

    graph.start(node.id, tier="local", worker_id="w")
    plan = scheduler.plan_failure(graph.get(node.id), None)
    scheduler.apply(graph.get(node.id), plan, "the model was wrong")

    assert graph.get(node.id).attempts == 1


def test_a_cleared_outage_resets_the_streak(ledger: Ledger, clock: ManualClock) -> None:
    """Otherwise a node that hit trouble twice in a long run parks on the third."""
    from forge.config import SchedulerConfig
    from forge.errors import RateLimited

    graph = TaskGraph(ledger, "proj_test", clock)
    config = SchedulerConfig(max_attempts=20, max_transient_attempts=3, backoff_base=1.0)
    scheduler = Scheduler(graph, config, clock=clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")

    for _ in range(2):
        graph.start(node.id, tier="local", worker_id="w")
        scheduler.plan_failure(graph.get(node.id), RateLimited("blip"))

    # The provider comes back and the node fails on its own merits.
    graph.start(node.id, tier="local", worker_id="w")
    scheduler.plan_failure(graph.get(node.id), None)

    graph.start(node.id, tier="local", worker_id="w")
    plan = scheduler.plan_failure(graph.get(node.id), RateLimited("blip"))
    assert plan.disposition is Disposition.RETRY, "the earlier outage is over and forgotten"


def test_the_config_forge_init_writes_can_reach_its_own_top_rung(tmp_path: Path) -> None:
    """`forge init` wrote a 4-rung ladder with a 4-attempt budget.

    Escalation is one rung per failure after `escalate_after_attempts`, so the
    top rung is first served on attempt 5 and the node blocked on attempt 4 --
    "exhausted the 4-attempt budget", having never once been given to the model
    that would have finished it. The `forge doctor` check meant to catch this
    used the wrong formula and did not.
    """
    from forge.config import attempts_needed_for_ladder, load_config, write_default_config

    write_default_config(tmp_path / ".forge" / "config.toml")
    config = load_config(tmp_path)
    assert str(tmp_path) in " ".join(config.sources), "precondition: the file was read"

    needed = attempts_needed_for_ladder(
        len(config.models.ladder), config.scheduler.escalate_after_attempts
    )
    assert config.scheduler.max_attempts >= needed, (
        f"{len(config.models.ladder)} rungs need {needed} attempts, "
        f"the starter config allows {config.scheduler.max_attempts}"
    )
    assert config.models.ladder == ["local", "local_deep", "haiku", "sonnet", "opus"]
    assert all(
        config.models.models[name].hosted == "cloud"
        for name in ("haiku", "sonnet", "opus")
    )
    assert not config.budget.enforce_cost_limits
    assert config.budget.max_cloud_fraction == 0.60
    assert config.coding.backend == "auto"
    assert not config.coding.opencode_subagents


def test_the_ladder_budget_counts_one_attempt_per_rung_not_one_interval(tmp_path: Path) -> None:
    """The old formula multiplied: 4 rungs x 2 = 8, where 5 is the answer."""
    from forge.config import attempts_needed_for_ladder

    assert attempts_needed_for_ladder(4, 2) == 5
    assert attempts_needed_for_ladder(1, 2) == 2, "a single-rung ladder never escalates"


def test_a_brief_blip_still_costs_nothing(ledger: Ledger, clock: ManualClock) -> None:
    """The exemption must survive: one bad minute must not consume a node."""
    from forge.errors import RateLimited

    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node(NodeKind.IMPLEMENT, "n")
    graph.start(node.id, tier="local", worker_id="w")

    plan = scheduler.plan_failure(graph.get(node.id), RateLimited("blip"))
    assert plan.disposition is Disposition.RETRY


def test_a_merge_conflict_consumes_an_attempt(ledger: Ledger, clock: ManualClock) -> None:
    """A conflicting branch is not a passing outage, so it must not be refunded.

    Classified transient, every merge conflict refunded its attempt, so
    `max_attempts` never applied and the node retried without bound. A real run
    logged 343 failures over ten hours with `attempts` still reading 3.
    """
    from forge.errors import MergeConflict

    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node("implement", "conflicting work")

    plan = scheduler.plan_failure(node, MergeConflict("conflicts", branch="forge/node/x"))

    assert plan.disposition == Disposition.RETRY
    assert not plan.refund_attempt


def test_repeated_merge_conflicts_eventually_stop(ledger: Ledger, clock: ManualClock) -> None:
    """The attempt budget has to actually bind, which is the whole point."""
    from forge.errors import MergeConflict

    scheduler, graph = _scheduler(ledger, clock)
    node = graph.add_node("implement", "conflicting work")
    graph.update(node.id, attempts=scheduler.config.max_attempts)

    plan = scheduler.plan_failure(graph.get(node.id), MergeConflict("conflicts"))

    assert plan.disposition == Disposition.BLOCK


def test_cancelled_work_is_not_on_the_critical_path(ledger: Ledger, clock: ManualClock) -> None:
    """Cancelled is finished, just not by being done.

    Treating only SUCCEEDED as terminal made a completed project report that it
    was gated on a proposal it had already declined: "Set default model for
    'implementation' task class to haiku", which read as though routing had
    been changed to start on a cloud rung.
    """
    graph = TaskGraph(ledger, "proj_test", clock)
    done = graph.add_node("implement", "the real work")
    graph.succeed(done.id, {"summary": "done"})
    declined = graph.add_node("improve", "Set default model to haiku")
    graph.cancel(declined.id, "declined")

    assert graph.critical_path() == []
