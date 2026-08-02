"""The model layer: routing policy, budget, structured output, caching."""

from __future__ import annotations

import json

import pytest

from forge.config import BudgetConfig, ModelSpec, default_models
from forge.errors import BudgetExhausted, MalformedOutput
from forge.kernel.ledger import Ledger
from forge.models.budget import Budget
from forge.models.cache import ResponseCache, cache_key
from forge.models.policy import RoutingPolicy
from forge.models.registry import Registry
from forge.models.router import Router
from forge.models.structured import extract_json, object_schema, parse_and_validate, string
from forge.models.types import Message, Request, TaskClass, TaskProfile
from forge.util.clock import ManualClock
from forge.util.jsonschema import validate

# --------------------------------------------------------------------------
# Structured output
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        'Here you go:\n```json\n{"a": 1}\n```',
        'Sure! {"a": 1} -- let me know if you need changes.',
        '```\n{"a": 1}\n```',
    ],
)
def test_json_is_recovered_from_chatty_output(raw: str) -> None:
    assert extract_json(raw) == {"a": 1}


def test_trailing_commas_and_python_literals_are_repaired() -> None:
    assert extract_json('prefix {"a": True, "b": None,} suffix') == {"a": True, "b": None}


def test_empty_output_is_an_error() -> None:
    with pytest.raises(MalformedOutput):
        extract_json("   ")


def test_schema_validation_reports_a_repairable_message() -> None:
    schema = object_schema({"name": string(), "count": {"type": "integer"}})
    _, errors = parse_and_validate('{"name": "x"}', schema)
    assert errors and "count" in errors[0]


def test_validator_distinguishes_bool_from_int() -> None:
    assert validate(True, {"type": "integer"})
    assert not validate(True, {"type": "boolean"})


def test_validator_walks_nested_structures() -> None:
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "object",
                     "properties": {"n": {"type": "integer"}}, "required": ["n"]}}},
        "required": ["items"],
    }
    errors = validate({"items": [{"n": 1}, {"m": 2}]}, schema)
    assert len(errors) == 1 and "$.items[1]" in errors[0]


def test_additional_properties_are_rejected_when_strict() -> None:
    schema = object_schema({"a": string()}, strict=True)
    errors = validate({"a": "x", "b": "y"}, schema)
    assert errors and "unexpected property 'b'" in errors[0]


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def _budget(ledger: Ledger, clock: ManualClock, **kwargs) -> Budget:
    config = BudgetConfig(total_cost=10.0, daily_cost=4.0, per_node_cost=1.0,
                          escalation_reserve=0.2, **kwargs)
    return Budget(config, ledger, clock)


def test_local_calls_bypass_cloud_ceilings(ledger: Ledger, clock: ManualClock) -> None:
    budget = _budget(ledger, clock)
    budget.record(model="local", tier="local", hosted="local", cost=3.9,
                  input_tokens=1000, output_tokens=100)
    # The daily cloud ceiling is 4.0 and is already nearly spent, but local work
    # must keep flowing.
    budget.check(1.0, hosted="local")
    with pytest.raises(BudgetExhausted):
        budget.check(1.0, hosted="cloud")


def test_escalation_reserve_is_protected_from_routine_work(ledger: Ledger, clock: ManualClock) -> None:
    # Daily ceiling raised out of the way so the reserve is the binding limit.
    budget = _budget(ledger, clock)
    budget.config.daily_cost = 100.0
    budget.record(model="f", tier="frontier", hosted="cloud", cost=7.5,
                  input_tokens=100, output_tokens=100)
    with pytest.raises(BudgetExhausted, match="escalation reserve"):
        budget.check(0.6, hosted="cloud", escalation=False)
    budget.check(0.6, hosted="cloud", escalation=True)  # allowed to use the reserve


def test_total_ceiling_stops_everything(ledger: Ledger, clock: ManualClock) -> None:
    budget = _budget(ledger, clock)
    budget.record(model="f", tier="frontier", hosted="cloud", cost=9.9,
                  input_tokens=1, output_tokens=1)
    with pytest.raises(BudgetExhausted, match="project cost ceiling"):
        budget.check(0.2, hosted="local")


def test_per_node_ceiling_is_enforced(ledger: Ledger, clock: ManualClock) -> None:
    budget = _budget(ledger, clock)
    budget.record(model="f", tier="frontier", hosted="cloud", cost=0.9,
                  input_tokens=1, output_tokens=1, node_id="node_a")
    with pytest.raises(BudgetExhausted, match="per-node"):
        budget.check(0.2, hosted="cloud", node_id="node_a")
    budget.check(0.2, hosted="cloud", node_id="node_b")


def test_spend_survives_a_restart(config, clock: ManualClock) -> None:
    """Budgets are derived from the ledger, so a crash does not reset them."""
    first = Ledger(config.ledger_path, clock=clock)
    _budget(first, clock).record(model="f", tier="frontier", hosted="cloud", cost=2.5,
                                 input_tokens=10, output_tokens=10)
    first.close()

    second = Ledger(config.ledger_path, clock=clock)
    assert _budget(second, clock).snapshot().total == pytest.approx(2.5)
    second.close()


def test_cloud_pressure_rises_above_target(ledger: Ledger, clock: ManualClock) -> None:
    budget = _budget(ledger, clock)
    budget.config.cloud_fraction_target = 0.1
    budget.record(model="f", tier="frontier", hosted="cloud", cost=0.1,
                  input_tokens=30_000, output_tokens=0)
    budget.record(model="l", tier="local", hosted="local", cost=0.0,
                  input_tokens=10_000, output_tokens=0)
    assert budget.cloud_pressure() > 0.5


def test_hard_cloud_fraction_rejects_a_project_call_before_crossing(
    ledger: Ledger, clock: ManualClock
) -> None:
    budget = _budget(ledger, clock)
    budget.config.max_cloud_fraction = 0.60
    budget.record(
        model="local",
        tier="local",
        hosted="local",
        cost=0.0,
        input_tokens=0,
        output_tokens=1_000,
    )
    budget.record(
        model="f",
        tier="frontier",
        hosted="cloud",
        cost=0.1,
        input_tokens=0,
        output_tokens=1_000,
    )

    budget.check(
        0.1,
        hosted="cloud",
        estimated_output_tokens=400,
        node_id="allowed",
        escalation=True,
    )
    with pytest.raises(BudgetExhausted, match="cloud-generated-token ceiling"):
        budget.check(
            0.1,
            hosted="cloud",
            estimated_output_tokens=1_000,
            node_id="blocked",
            escalation=True,
        )


# --------------------------------------------------------------------------
# Routing policy
# --------------------------------------------------------------------------


def _policy(ledger: Ledger) -> tuple[RoutingPolicy, dict[str, ModelSpec], list[str]]:
    models = default_models()
    policy = RoutingPolicy(ledger, seed=7)
    ladder = models.ladder
    specs = {name: models.models[name] for name in ladder}
    return policy, specs, ladder


def test_low_stakes_work_starts_cheap(ledger: Ledger) -> None:
    policy, specs, ladder = _policy(ledger)
    profile = TaskProfile(task_class=TaskClass.CLASSIFICATION, difficulty=0.1, stakes=0.1)
    assert policy.decide(profile, ladder, specs).model == "local"


def test_high_stakes_work_starts_stronger(ledger: Ledger) -> None:
    policy, specs, ladder = _policy(ledger)
    profile = TaskProfile(task_class=TaskClass.ARCHITECTURE, difficulty=0.9, stakes=0.95)
    assert policy.decide(profile, ladder, specs).model != "local"


def test_prior_attempts_force_a_higher_rung(ledger: Ledger) -> None:
    policy, specs, ladder = _policy(ledger)
    base = TaskProfile(task_class=TaskClass.IMPLEMENTATION, difficulty=0.3, stakes=0.3)
    first = policy.decide(base, ladder, specs).model
    retried = policy.decide(
        TaskProfile(task_class=TaskClass.IMPLEMENTATION, difficulty=0.3, stakes=0.3, attempt=2),
        ladder,
        specs,
    ).model
    assert ladder.index(retried) > ladder.index(first)


def test_observed_success_pulls_routing_back_down(ledger: Ledger) -> None:
    """The learning claim: evidence that local works should make local chosen."""
    policy, specs, ladder = _policy(ledger)
    profile = TaskProfile(task_class=TaskClass.CODE_REVIEW, difficulty=0.5, stakes=0.7)
    assert policy.decide(profile, ladder, specs).model != "local"

    for _ in range(40):
        policy.record(TaskClass.CODE_REVIEW, "local", success=True)
    assert policy.decide(profile, ladder, specs).model == "local"


def test_observed_failure_pushes_routing_up(ledger: Ledger) -> None:
    policy, specs, ladder = _policy(ledger)
    profile = TaskProfile(task_class=TaskClass.EXTRACTION, difficulty=0.2, stakes=0.2)
    assert policy.decide(profile, ladder, specs).model == "local"

    for _ in range(40):
        policy.record(TaskClass.EXTRACTION, "local", success=False)
    assert policy.decide(profile, ladder, specs).model != "local"


def test_cloud_pressure_makes_the_policy_more_frugal(ledger: Ledger) -> None:
    policy, _specs, _ladder = _policy(ledger)
    profile = TaskProfile(task_class=TaskClass.IMPLEMENTATION, difficulty=0.5, stakes=0.5)
    relaxed = policy.required_success(profile, cloud_pressure=0.0)
    pressured = policy.required_success(profile, cloud_pressure=1.0)
    assert pressured < relaxed


def test_vision_requirement_filters_the_ladder(ledger: Ledger) -> None:
    policy, specs, ladder = _policy(ledger)
    profile = TaskProfile(task_class=TaskClass.VISUAL_JUDGEMENT, needs_vision=True)
    chosen = policy.decide(profile, ladder, specs).model
    assert specs[chosen].supports_vision


def test_recommendations_appear_once_there_is_evidence(ledger: Ledger) -> None:
    policy, _, _ = _policy(ledger)
    for _ in range(12):
        policy.record(TaskClass.DEBUGGING, "local", success=False)
    notes = policy.recommendations()
    assert any("debugging" in note for note in notes)


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


def test_router_skips_models_that_cannot_hold_the_prompt(ledger: Ledger, clock: ManualClock) -> None:
    models = default_models()
    for spec in models.models.values():
        spec.context_window = 5_000
    models.models["opus"].context_window = 400_000
    for provider in models.providers.values():
        provider.kind, provider.api_key_env = "echo", ""

    router = Router(Registry(models), RoutingPolicy(ledger, seed=3), _budget(ledger, clock))
    huge = [Message("user", "x" * 200_000)]
    route = router.select(TaskProfile(task_class=TaskClass.SUMMARIZATION), messages=huge)
    assert route.model == "opus"


def test_router_falls_back_to_local_when_cloud_is_unaffordable(ledger: Ledger, clock: ManualClock) -> None:
    models = default_models()
    for provider in models.providers.values():
        provider.kind, provider.api_key_env = "echo", ""
    budget = _budget(ledger, clock)
    budget.record(model="f", tier="frontier", hosted="cloud", cost=3.99,
                  input_tokens=1, output_tokens=1)  # daily ceiling is 4.0

    router = Router(Registry(models), RoutingPolicy(ledger, seed=3), budget)
    route = router.select(
        TaskProfile(task_class=TaskClass.ARCHITECTURE, difficulty=0.9, stakes=0.95),
        messages=[Message("user", "design it")],
    )
    assert route.spec.hosted == "local", "must degrade to local rather than stop working"


def test_escalation_never_returns_a_lower_rung(ledger: Ledger, clock: ManualClock) -> None:
    models = default_models()
    for provider in models.providers.values():
        provider.kind, provider.api_key_env = "echo", ""
    router = Router(Registry(models), RoutingPolicy(ledger, seed=3), _budget(ledger, clock))
    profile = TaskProfile(task_class=TaskClass.IMPLEMENTATION)

    route = router.escalate(profile, "local", messages=[Message("user", "hi")])
    assert route is not None and models.ladder.index(route.model) > 0
    assert router.escalate(profile, models.ladder[-1]) is None


def test_escalation_depends_on_the_rung_not_on_the_attempt_count(
    ledger: Ledger, clock: ManualClock
) -> None:
    """Two skip mechanisms were compounding.

    `escalate` excludes every rung at or below the current one -- which is the
    whole of "do not repeat what just failed" -- and then also passed
    `profile.escalated()`, whose bumped `attempt` becomes an *index floor* into
    the candidate list. That list has just had its bottom removed, so the floor
    counted the same rungs a second time and the same escalation landed higher
    the later it happened. Where you are on the ladder is the only thing that
    should decide where you go next.
    """
    models = default_models()
    for provider in models.providers.values():
        provider.kind, provider.api_key_env = "echo", ""
    router = Router(Registry(models), RoutingPolicy(ledger, seed=3), _budget(ledger, clock))

    messages = [Message("user", "hi")]
    early = router.escalate(
        TaskProfile(task_class=TaskClass.IMPLEMENTATION, attempt=0), "local", messages=messages
    )
    late = router.escalate(
        TaskProfile(task_class=TaskClass.IMPLEMENTATION, attempt=3), "local", messages=messages
    )

    assert early is not None and late is not None
    assert early.model == late.model, (
        f"the same escalation chose {early.model!r} early and {late.model!r} late"
    )


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _request(text: str) -> Request:
    return Request(messages=[Message("user", text)], profile=TaskProfile(task_class=TaskClass.SUMMARIZATION))


def test_cache_key_covers_everything_that_changes_the_answer() -> None:
    a, b = _request("hello"), _request("hello")
    assert cache_key(a, "local") == cache_key(b, "local")
    assert cache_key(a, "local") != cache_key(a, "frontier")
    assert cache_key(a, "local") != cache_key(_request("goodbye"), "local")

    schema_request = _request("hello")
    schema_request.schema = {"type": "object"}
    assert cache_key(schema_request, "local") != cache_key(a, "local")


def test_cached_completions_report_zero_cost(tmp_path, clock: ManualClock) -> None:
    """Otherwise a retried node would be billed repeatedly for one call."""
    from forge.models.types import Completion, Usage

    cache = ResponseCache(tmp_path / "c", clock=clock)
    cache.put("k", Completion(text="hi", model="local", tier="local",
                              usage=Usage(10, 5), cost=1.23))
    hit = cache.get("k")
    assert hit is not None and hit.cached and hit.cost == 0.0


def test_cache_entries_expire(tmp_path, clock: ManualClock) -> None:
    from forge.models.types import Completion

    cache = ResponseCache(tmp_path / "c", ttl=100, clock=clock)
    cache.put("k", Completion(text="hi", model="local", tier="local"))
    assert cache.get("k") is not None
    clock.advance(101)
    assert cache.get("k") is None


# --------------------------------------------------------------------------
# Client integration
# --------------------------------------------------------------------------


def test_client_repairs_invalid_structured_output(config, ledger: Ledger, provider) -> None:
    from forge.models.client import ModelClient

    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, provider)

    schema = object_schema({"answer": string()})
    provider.responses = ['{"wrong_field": 1}', '{"answer": "correct"}']

    result = client.structured(
        [Message("user", "q")], TaskProfile(task_class=TaskClass.EXTRACTION), schema
    )
    assert result == {"answer": "correct"}


def test_client_gives_up_after_bounded_repairs(config, ledger: Ledger, provider) -> None:
    from forge.models.client import ModelClient

    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, provider)
    provider.handler = lambda request, spec: '{"nope": 1}'

    with pytest.raises(MalformedOutput):
        client.structured(
            [Message("user", "q")],
            TaskProfile(task_class=TaskClass.EXTRACTION),
            object_schema({"answer": string()}),
        )


def test_client_reroutes_around_an_unavailable_provider(config, ledger: Ledger, provider) -> None:
    from forge.errors import ProviderUnavailable
    from forge.models.client import ModelClient

    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, provider)
    provider.fail_with = ProviderUnavailable("connection refused")
    provider.fail_times = 1
    provider.responses = ["recovered"]

    completion = client.complete([Message("user", "q")], TaskProfile(task_class=TaskClass.SUMMARIZATION))
    assert completion.text == "recovered"


def test_client_does_not_probe_stronger_models_after_a_rate_limit(
    config, ledger: Ledger, provider
) -> None:
    """A shared subscription limit applies to sibling models too.

    The live pinball run tried Sonnet and then Opus on every retry while the
    Claude session was exhausted.  A rate limit must leave the scheduler to
    back off instead of turning one failed request into a ladder sweep.
    """
    from forge.errors import RateLimited
    from forge.models.client import ModelClient

    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, provider)
    provider.fail_with = RateLimited("session limit")
    provider.fail_times = 1
    provider.responses = ["must not be reached"]

    with pytest.raises(RateLimited):
        client.complete(
            [Message("user", "q")],
            TaskProfile(task_class=TaskClass.SUMMARIZATION),
        )

    requests = ledger.read(types=["model.request"])
    assert len(requests) == 1, "a 429 must not probe another model on the same quota"


def test_spend_is_recorded_for_every_call(config, ledger: Ledger, provider) -> None:
    from forge.models.client import ModelClient

    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, provider)
    client.complete([Message("user", "q")], TaskProfile(task_class=TaskClass.SUMMARIZATION), node_id="node_1")

    assert client.budget.snapshot().calls == 1
    assert client.budget.node_spend("node_1") >= 0.0


# --------------------------------------------------------------------------
# Dry-run isolation
#
# `forge run --dry-run` swaps every provider for the echo stub. Because the stub
# is installed *under the real provider's name*, everything downstream -- the
# spend ledger, the response cache, the routing posteriors -- would otherwise
# treat rehearsal output as if a model had produced it. The cache case is the
# dangerous one: a poisoned entry is served to a later real run and looks
# exactly like a cheap success.
# --------------------------------------------------------------------------


def _stub_client(config, ledger: Ledger, provider):
    from forge.models.client import ModelClient

    client = ModelClient(config, ledger)
    provider.stub = True
    for name in config.models.providers:
        client.registry.install(name, provider)
    return client


def test_dry_run_output_costs_nothing(config, ledger: Ledger, provider) -> None:
    client = _stub_client(config, ledger, provider)
    completion = client.complete(
        [Message("user", "q")], TaskProfile(task_class=TaskClass.SUMMARIZATION), node_id="node_1"
    )

    assert completion.stub
    assert completion.cost == 0.0
    assert client.budget.snapshot().calls == 0, "a rehearsal must not appear in the spend ledger"
    assert client.budget.snapshot().cloud_tokens == 0


def test_dry_run_never_writes_the_response_cache(config, ledger: Ledger, provider) -> None:
    """Otherwise the next real run silently inherits skeleton answers."""
    config.models.cache_enabled = True  # the shared fixture disables it
    client = _stub_client(config, ledger, provider)
    provider.responses = ["stub answer"]
    client.complete([Message("user", "q")], TaskProfile(task_class=TaskClass.SUMMARIZATION))

    assert client.cache.stats()["misses"] == 0, "a stubbed call must not consult the cache at all"
    assert not list(client.cache.root.rglob("*.json"))


def test_dry_run_does_not_train_the_router(config, ledger: Ledger, provider) -> None:
    client = _stub_client(config, ledger, provider)
    schema = object_schema({"answer": string()})
    provider.responses = ['{"answer": "x"}']
    client.structured(
        [Message("user", "q")], TaskProfile(task_class=TaskClass.EXTRACTION), schema=schema
    )

    rows = [r for r in client.policy.table() if r["task_class"] == str(TaskClass.EXTRACTION)]
    assert not rows, "fake successes would bias real routing"


# --------------------------------------------------------------------------
# Explicit nulls for optional fields
# --------------------------------------------------------------------------


def test_an_explicit_null_for_an_optional_field_means_absent() -> None:
    """JSON has no `undefined`, so models write null where they mean absent.

    Observed live: a scaffold node failed 70 times because codex returned
    `content: null` on a `create_dir` edit that needed no content. The edit was
    perfectly well formed. OpenAI's strict mode makes this unavoidable -- there,
    optionality *is* a nullable type -- so rejecting nulls made the codex rung
    unusable for every structured call.
    """
    from forge.workspace.patch import EDIT_PLAN_SCHEMA

    text = json.dumps({
        "summary": "scaffold the project",
        "edits": [{"path": "src", "op": "create_dir", "content": None,
                   "anchor": None, "occurrence": None, "reason": None}],
    })
    value, errors = parse_and_validate(text, EDIT_PLAN_SCHEMA)

    assert errors == [], f"optional nulls must not fail validation: {errors}"
    assert value["edits"][0]["path"] == "src"
    assert "content" not in value["edits"][0], "the key should be dropped, not kept as None"


def test_a_null_for_a_required_field_is_still_an_error() -> None:
    """Dropping nulls must not paper over a genuinely missing answer."""
    from forge.workspace.patch import EDIT_PLAN_SCHEMA

    text = json.dumps({"summary": None, "edits": [{"path": "a.ts", "op": "write", "content": "x"}]})
    _, errors = parse_and_validate(text, EDIT_PLAN_SCHEMA)
    assert errors, "a required field set to null is a real failure"


def test_a_genuinely_nullable_field_keeps_its_null() -> None:
    schema = object_schema({"note": {"type": ["string", "null"]}})
    value, errors = parse_and_validate(json.dumps({"note": None}), schema)
    assert errors == [] and value["note"] is None


def test_the_strict_transform_and_the_validator_agree_end_to_end() -> None:
    """What we ask codex for must be what our own validator accepts back."""
    from forge.models.structured import strict_schema
    from forge.workspace.patch import EDIT_PLAN_SCHEMA

    sent = strict_schema(EDIT_PLAN_SCHEMA)
    # A response that is valid against the strict schema codex was given...
    # Strict mode expresses optionality as null-admitting rather than by omission,
    # so every optional key arrives explicitly as null.
    response = {
        "summary": "scaffold",
        "need_files": None,
        "edits": [{"path": "src", "op": "create_dir", "content": None,
                   "anchor": None, "occurrence": None, "reason": "new module"}],
    }
    assert validate(response, sent) == [], "precondition: valid under the strict schema"
    # ...must also survive Forge's validation against the original.
    _, errors = parse_and_validate(json.dumps(response), EDIT_PLAN_SCHEMA)
    assert errors == [], f"the two ends disagree: {errors}"
    # ...and must survive being turned into an EditPlan. `int(None)` used to raise
    # TypeError here, which reached the scheduler as an internal platform fault.
    from forge.workspace.patch import EditPlan

    plan = EditPlan.from_payload(response)
    # `occurrence` stays None rather than becoming 1: an unanswered "which
    # match" is not a claim that the first one is right, and the ambiguity check
    # in `_apply_anchor` is the only thing standing between a three-way anchor
    # match and a silent edit to the wrong one.
    assert plan.edits[0].occurrence is None and plan.edits[0].content == ""
    assert plan.need_files == []


# --------------------------------------------------------------------------
# Reasoning overflow: the failure that looks exactly like a hang
# --------------------------------------------------------------------------


def _overflow(model: str = "local_deep", **context):  # type: ignore[no-untyped-def]
    from forge.errors import ReasoningBudgetExhausted

    return ReasoningBudgetExhausted(
        "model spent its entire output budget on reasoning and returned no answer",
        model=model,
        reasoning_chars=40_000,
        **context,
    )


def _route_for(client, model: str, **spec_overrides):  # type: ignore[no-untyped-def]
    """A route for `model`, optionally with an altered spec.

    Overrides are applied to a *copy*: the registry hands out one shared
    ModelSpec per rung, and mutating it in place leaked a 100-second timeout
    into every later test in the file.
    """
    import dataclasses

    from forge.models.policy import Decision
    from forge.models.router import Route

    spec = client.registry.spec(model)
    if spec_overrides:
        spec = dataclasses.replace(spec, **spec_overrides)
    return Route(
        model=model,
        spec=spec,
        decision=Decision(model=model, reason="test", expected_success=0.9, required_success=0.5),
        estimated_cost=0.0,
        estimated_input_tokens=0,
    )


def test_the_output_budget_is_bumped_when_there_is_room(config, ledger: Ledger, provider) -> None:
    from forge.models.client import ModelClient
    from forge.models.types import Request

    client = ModelClient(config, ledger)
    # `tokens_per_second=0` means "rate unmeasured", so the context window is
    # the only ceiling -- the case this test is about. The shipped local_deep
    # rung has a measured rate that leaves no useful room, which is a separate
    # question covered by test_a_bump_too_small_to_matter_disables_thinking.
    route = _route_for(client, "local_deep", tokens_per_second=0.0)
    request = Request(messages=[Message("user", "short prompt")],
                      profile=TaskProfile(task_class=TaskClass.IMPLEMENTATION))

    assert client._bump_budget(request, route, _overflow(), 0) is True
    assert request.max_output_tokens > route.spec.max_output_tokens


def test_the_bump_never_exceeds_what_the_context_window_can_hold(config, ledger: Ledger, provider) -> None:
    """Asking for more output than physically fits cannot succeed.

    On the pinball run the budget was doubled to 131k output tokens against a
    131k context window. Tokens are wall-clock on a local model, so each
    impossible attempt cost forty minutes and failed identically.
    """
    from forge.models.client import ModelClient
    from forge.models.types import Request, estimate_messages

    client = ModelClient(config, ledger)
    # Unmeasured rate: this test is about the context window, not the clock.
    route = _route_for(client, "local_deep", tokens_per_second=0.0)
    spec = route.spec
    # Sized so that doubling the budget overflows the window but a 1.75x
    # increase still fits: big enough to be worth attempting, small enough that
    # a blind doubling would be silently wrong. That gap is the whole test.
    prompt = "x" * (4 * (spec.context_window - 2048 - int(spec.max_output_tokens * 1.75)))
    request = Request(messages=[Message("user", prompt)],
                      profile=TaskProfile(task_class=TaskClass.IMPLEMENTATION),
                      max_output_tokens=spec.max_output_tokens)

    assert client._bump_budget(request, route, _overflow(), 0) is True
    prompt_tokens = estimate_messages(request.messages)
    assert request.max_output_tokens > spec.max_output_tokens, "it should still grow"
    assert prompt_tokens + request.max_output_tokens <= spec.context_window, (
        f"asked for {request.max_output_tokens} output on top of a "
        f"{prompt_tokens}-token prompt in a {spec.context_window}-token window"
    )


def test_thinking_is_disabled_when_no_headroom_is_left(config, ledger: Ledger, provider) -> None:
    """The opposite of a bigger budget, and the last thing to try before paying.

    The failure is that the model thought until it ran out of room. A model that
    answers directly usually answers.
    """
    from forge.models.client import ModelClient
    from forge.models.types import Request

    client = ModelClient(config, ledger)
    route = _route_for(client, "local_deep")
    assert route.spec.extra.get("thinking"), "fixture must have a thinking rung"
    prompt = "x" * (route.spec.context_window - 1000) * 4
    request = Request(messages=[Message("user", prompt)],
                      profile=TaskProfile(task_class=TaskClass.IMPLEMENTATION))

    assert client._bump_budget(request, route, _overflow(), 0) is True
    params = request.extra["provider_params"]
    assert params["chat_template_kwargs"]["enable_thinking"] is False
    # And it is not offered twice: the second time, escalate instead.
    assert client._bump_budget(request, route, _overflow(), 0) is False


def test_a_generation_that_produced_nothing_is_still_accounted(config, ledger: Ledger, provider) -> None:
    """Minutes of GPU time with no ledger entry made a busy run look idle."""
    from forge.models.client import ModelClient
    from forge.models.types import Request

    client = ModelClient(config, ledger)
    route = _route_for(client, "local_deep")
    request = Request(messages=[Message("user", "q")],
                      profile=TaskProfile(task_class=TaskClass.IMPLEMENTATION),
                      node_id="n1")

    client._record_overflow(request, route, _overflow(input_tokens=12_000, output_tokens=32_768))
    rows = client.budget.report()["by_model"]
    row = next(r for r in rows if r["model"] == "local_deep")
    assert row["output_tokens"] == 32_768 and row["input_tokens"] == 12_000


def test_a_budget_beyond_what_the_timeout_allows_is_not_deliverable() -> None:
    """Output tokens are wall-clock, and the socket deadline is the other wall.

    A dense 27B generates at ~12 tok/s where the mixture it replaced did ~85.
    Every local rung became impossible at once and reported a read timeout,
    which names the network rather than the arithmetic.
    """
    from forge.config import ModelSpec

    slow = ModelSpec(name="slow", timeout=3600.0, tokens_per_second=12.0)
    assert slow.deliverable_tokens() == int(3600 * 0.85 * 12)

    unmeasured = ModelSpec(name="unknown", timeout=3600.0)
    assert unmeasured.deliverable_tokens() == 0, "no rate means no opinion, not zero capacity"


def test_the_bump_never_asks_for_more_than_the_clock_allows(config, ledger: Ledger, provider) -> None:
    """The context window has room to spare; the timeout does not."""
    from forge.models.client import ModelClient
    from forge.models.types import Request

    client = ModelClient(config, ledger)
    # A short timeout at a measured rate: plenty of context, almost no clock.
    route = _route_for(client, "local_deep", timeout=100.0, tokens_per_second=12.0)
    ceiling = route.spec.deliverable_tokens()

    request = Request(
        messages=[Message("user", "short prompt")],
        profile=TaskProfile(task_class=TaskClass.IMPLEMENTATION),
        max_output_tokens=256,
    )

    assert client._bump_budget(request, route, _overflow(), 0) is True
    assert request.max_output_tokens > 256, "it should still have grown"
    assert request.max_output_tokens <= ceiling, (
        f"asked for {request.max_output_tokens} tokens the model cannot emit in "
        f"{route.spec.timeout:.0f}s (ceiling {ceiling})"
    )


def test_a_bump_too_small_to_matter_disables_thinking_instead(config, ledger: Ledger, provider) -> None:
    """Observed live on the pinball run, at the cost of fifty minutes.

    The clock ceiling left 12% more room than the budget that had just
    overflowed. A model that filled 32k tokens and stopped mid-thought does not
    finish inside 36k -- it stops 12% later. Turning thinking off changes the
    shape of the answer rather than its length, so that is what to try.
    """
    from forge.models.client import ModelClient
    from forge.models.types import Request

    client = ModelClient(config, ledger)
    # 36720 deliverable against a 32768 budget: real headroom, useless headroom.
    route = _route_for(client, "local_deep", timeout=3600.0, tokens_per_second=12.0)
    request = Request(
        messages=[Message("user", "short prompt")],
        profile=TaskProfile(task_class=TaskClass.IMPLEMENTATION),
        max_output_tokens=32_768,
    )

    assert client._bump_budget(request, route, _overflow(), 0) is True
    assert request.max_output_tokens == 32_768, "the budget should not have crept up"
    assert client._thinking_disabled(request), "it should have turned thinking off instead"


def test_the_overflow_diagnostic_names_the_wall_it_actually_hit() -> None:
    """It reported "no context headroom left" with 115k of 131k tokens free.

    Two different walls stop the budget growing -- the context window and the
    clock -- and for a while both printed the context message. Naming the wrong
    one sends the reader to `context_window` when the fix is `timeout`, which is
    how several hours went today.
    """
    from forge.models.client import _limiting_wall

    # Clock binds: plenty of window left, but the timeout cannot deliver it.
    assert _limiting_wall(headroom=115_000, deliverable=36_720) == "clock"
    # Window binds: the model is fast enough, the prompt has eaten the room.
    assert _limiting_wall(headroom=4_000, deliverable=36_720) == "context"
    # Rate unmeasured means no opinion about the clock, so it cannot be blamed.
    assert _limiting_wall(headroom=4_000, deliverable=0) == "context"


def test_one_provider_slot_serialises_models_that_share_hardware() -> None:
    """Per-model limits cannot express "these rungs are one GPU".

    `local` and `local_deep` are the same weights on the same box. Their separate
    semaphores allowed two concurrent generations against hardware saturated by
    one: 26.7 tok/s alone versus 13.7 each doubled, identical aggregate
    throughput, every call twice as close to its timeout for no extra work.
    """
    import threading

    from forge.config import ModelSpec, ProviderConfig
    from forge.models.provider import Provider
    from forge.models.types import Completion, Usage

    overlap = {"max": 0, "now": 0}
    guard = threading.Lock()
    release = threading.Event()

    class _Counting(Provider):
        kind = "counting"

        def _complete(self, request, spec):  # type: ignore[no-untyped-def]
            with guard:
                overlap["now"] += 1
                overlap["max"] = max(overlap["max"], overlap["now"])
            release.wait(timeout=5.0)
            with guard:
                overlap["now"] -= 1
            return Completion(text="ok", usage=Usage(), model=spec.name, tier=spec.tier)

    provider = _Counting("local", ProviderConfig(max_concurrency=1, timeout=5.0))
    # Two *different* rungs, each permitting concurrency on its own.
    specs = [
        ModelSpec(name="local", concurrency=4, timeout=5.0),
        ModelSpec(name="local_deep", concurrency=2, timeout=5.0),
    ]
    request = Request(
        messages=[Message("user", "hi")],
        profile=TaskProfile(task_class=TaskClass.IMPLEMENTATION),
    )

    threads = [
        threading.Thread(target=provider.complete, args=(request, spec)) for spec in specs
    ]
    for t in threads:
        t.start()
    # Give the second thread every chance to slip through before letting go.
    threading.Event().wait(0.3)
    release.set()
    for t in threads:
        t.join(timeout=5.0)

    assert overlap["max"] == 1, (
        f"{overlap['max']} generations ran at once against a one-slot provider"
    )


def test_the_thinking_rung_starts_with_room_for_an_answer() -> None:
    """Four consecutive calls spent 88% of a 32768 budget reasoning and returned
    nothing, at 21 minutes each. A starting budget that reliably overflows is a
    tax on every attempt, not a safety margin.

    The two invariants that keep it honest: enough left for an answer after the
    reasoning this model actually does, and still short enough that the overflow
    bump remains available if a task reasons longer than any seen so far.
    """
    from forge.config import default_models

    spec = default_models().models["local_deep"]
    observed_reasoning = 29_228  # the largest of the four

    assert spec.max_output_tokens - observed_reasoning > 15_000, (
        "no room left for the answer after this model's observed reasoning"
    )
    assert spec.max_output_tokens <= spec.deliverable_tokens(), (
        "the budget cannot be delivered before the timeout"
    )
    assert spec.deliverable_tokens() >= spec.max_output_tokens * 1.5, (
        "no headroom left for a bump, so an unusually long reasoning run has no recourse"
    )


def _total_only_budget(ledger: Ledger, clock: ManualClock) -> Budget:
    """A budget where the *total* ceiling is the binding one."""
    config = BudgetConfig(total_cost=10.0, daily_cost=1000.0, per_node_cost=1000.0,
                          escalation_reserve=0.0)
    return Budget(config, ledger, clock)


def test_two_workers_cannot_both_pass_the_same_remaining_budget(
    ledger: Ledger, clock: ManualClock
) -> None:
    """Admission and reservation have to be one step, not two.

    `check` then `reserve` is a time-of-check/time-of-use race: with more than
    one worker every caller observes the same pre-reservation state and all of
    them pass, which is exactly what the module docstring promises cannot
    happen. Only one of these two calls fits in the remaining 1.0.
    """
    budget = _total_only_budget(ledger, clock)
    budget.record(model="opus", tier="cloud", hosted="cloud", cost=9.0,
                  input_tokens=10, output_tokens=10)

    admitted = 0
    for _ in range(2):
        try:
            budget.check_and_reserve(0.6, hosted="cloud", node_id="node_a", escalation=True)
            admitted += 1
        except BudgetExhausted:
            pass

    assert admitted == 1


def test_a_reservation_outlives_the_call_until_the_spend_is_recorded(
    ledger: Ledger, clock: ManualClock
) -> None:
    """Releasing before recording leaves a window with no accounting at all.

    In that gap the call is neither reserved nor visible in `spend`, so a
    concurrent worker sees a budget that has already been consumed as free.
    """
    budget = _total_only_budget(ledger, clock)
    budget.check_and_reserve(9.5, hosted="cloud", node_id="node_a", escalation=True)

    # Still reserved: a second admission must not see this as spare capacity.
    with pytest.raises(BudgetExhausted):
        budget.check_and_reserve(1.0, hosted="cloud", node_id="node_b", escalation=True)

    budget.record(model="opus", tier="cloud", hosted="cloud", cost=9.5,
                  input_tokens=10, output_tokens=10, node_id="node_a")
    budget.release(9.5, hosted="cloud", node_id="node_a")

    # And once recorded the ceiling is still enforced, now from the ledger.
    with pytest.raises(BudgetExhausted):
        budget.check_and_reserve(1.0, hosted="cloud", node_id="node_b", escalation=True)


def test_per_node_ceiling_counts_in_flight_reservations(
    ledger: Ledger, clock: ManualClock
) -> None:
    """The per-node ceiling was the weakest of the three: no reservation at all."""
    budget = _budget(ledger, clock)
    budget.check_and_reserve(0.8, hosted="cloud", node_id="node_a", escalation=True)
    with pytest.raises(BudgetExhausted):
        budget.check_and_reserve(0.8, hosted="cloud", node_id="node_a", escalation=True)
