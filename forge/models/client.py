"""The model client: one entry point for every completion in the platform.

Everything that must happen around a model call happens here, once:

* route the request,
* consult the response cache,
* admit it against the budget,
* call the provider with retries already handled beneath,
* validate structured output and repair it if needed,
* escalate to a stronger model when the current one cannot deliver,
* record spend, outcome and enough context for the retrospective.

Agents therefore contain prompt construction and domain logic and nothing else.
That is the difference between a platform where adding a specialist agent is
fifty lines and one where it is five hundred.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..errors import (
    BudgetExhausted,
    ContextOverflow,
    MalformedOutput,
    ModelError,
    ProviderUnavailable,
    RateLimited,
    ReasoningBudgetExhausted,
)
from ..kernel.events import Event, EventType
from ..kernel.ledger import Ledger
from ..obs.log import get_logger
from ..obs.metrics import Metrics
from ..util.clock import Clock, default_clock
from .budget import Budget
from .cache import ResponseCache, cache_key
from .policy import RoutingPolicy
from .registry import Registry
from .router import Route, Router
from .structured import parse_and_validate, repair_instruction, schema_instruction
from .types import Completion, Message, Request, TaskProfile, ToolSpec, estimate_messages

log = get_logger("models.client")

#: Repairs after a schema violation, before escalating. One is usually enough;
#: a second rarely helps and a third never has in practice.
MAX_REPAIRS = 2

#: Times the output budget is doubled when a reasoning model overruns it before
#: giving up and escalating. Two doublings covers every case observed; beyond
#: that the prompt, not the budget, is the problem.
MAX_BUDGET_BUMPS = 2

#: The smallest budget increase worth an attempt, as a multiple of the budget
#: that just overflowed. A thinking model that filled 32k tokens and stopped
#: mid-thought does not become decisive with 36k; it stops 12% later, and on a
#: 12 tok/s local rung that costs fifty minutes to establish. Anything under
#: this falls through to disabling thinking, which actually changes the shape of
#: the answer rather than its length.
_MIN_USEFUL_BUMP = 1.5


def _limiting_wall(headroom: int, deliverable: int) -> str:
    """Which constraint is stopping the output budget from growing.

    Two different walls produce one symptom, and for a while both printed "no
    context headroom left" -- which reads as a full context window when the real
    limit can be wall-clock, with 115k of a 131k window still free. Naming the
    wrong constraint sends the reader to the wrong knob: `context_window` when
    the fix is `timeout`.
    """
    if deliverable and deliverable < headroom:
        return "clock"
    return "context"

#: Tokens held back from the context window when deciding how much output budget
#: can still fit. Covers a repair instruction, a schema reminder and the slack in
#: a character-count token estimate.
_CONTEXT_MARGIN = 2048


class ModelClient:
    def __init__(
        self,
        config: Config,
        ledger: Ledger,
        *,
        registry: Registry | None = None,
        policy: RoutingPolicy | None = None,
        budget: Budget | None = None,
        cache: ResponseCache | None = None,
        metrics: Metrics | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self._clock = clock or default_clock()
        self.registry = registry or Registry(config.models, self._clock)
        self.policy = policy or RoutingPolicy(ledger, config=config.improvement, clock=self._clock)
        self.budget = budget or Budget(config.budget, ledger, self._clock)
        self.router = Router(self.registry, self.policy, self.budget)
        self.cache = cache or ResponseCache(
            config.cache_dir / "responses",
            ttl=config.models.cache_ttl_seconds,
            clock=self._clock,
            enabled=config.models.cache_enabled,
        )
        self.metrics = metrics or Metrics(ledger, self._clock)

    # -- public API ------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        profile: TaskProfile,
        *,
        tools: list[ToolSpec] | None = None,
        schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        node_id: str | None = None,
        no_cache: bool = False,
        allow_escalation: bool = True,
    ) -> Completion:
        """Run a completion, escalating up the ladder as needed.

        Raises :class:`ModelError` only when every rung has been tried and
        failed. Callers should treat that as a node-level failure, not as a
        reason to abandon the run.
        """
        request = Request(
            messages=list(messages),
            profile=profile,
            tools=list(tools or []),
            schema=schema,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stop=list(stop or []),
            no_cache=no_cache,
            node_id=node_id,
        )
        tried: set[str] = set()
        last_error: Exception | None = None
        budget_bumps = 0
        route = self.router.select(
            profile, messages=request.messages, max_output_tokens=max_output_tokens, node_id=node_id
        )

        while True:
            tried.add(route.model)
            self._emit_route(route, request)
            try:
                completion = self._attempt(request, route)
            except ReasoningBudgetExhausted as exc:
                # The model can do this; it just ran out of room to say so.
                # Escalating here would spend a frontier call to fix a number.
                last_error = exc
                self._record_overflow(request, route, exc)
                bumped = self._bump_budget(request, route, exc, budget_bumps)
                if bumped:
                    budget_bumps += 1
                    continue
                next_route = self._next_route(request, route, tried, reason="reasoning overflow")
                if next_route is None:
                    raise
                route = next_route
                continue
            except RateLimited as exc:
                # A quota is normally shared by every model behind one provider
                # (Sonnet and Opus use the same Claude subscription, and the two
                # local rungs share one llama.cpp server).  Treating a 429 like
                # an unavailable *model* made one node probe every sibling on
                # every retry.  During a Claude session-limit outage that
                # produced hundreds of guaranteed-to-fail Opus calls.
                #
                # Let the scheduler apply its outage backoff.  The request has
                # not demonstrated a capability failure, so climbing the model
                # ladder is both wasteful and contrary to local-first routing.
                last_error = exc
                log.warn("provider rate limited; deferring without rerouting", model=route.model)
                self.metrics.incr("model.rate_limited", model=route.model)
                raise
            except ProviderUnavailable as exc:
                # The model is fine; the endpoint is not. Move sideways/up
                # rather than failing the node on an infrastructure blip.
                last_error = exc
                log.warn("provider unavailable, rerouting", model=route.model, error=str(exc))
                self.metrics.incr("model.provider_unavailable", model=route.model)
                next_route = self._next_route(request, route, tried, reason="provider unavailable")
                if next_route is None:
                    raise
                route = next_route
                continue
            except (MalformedOutput, ContextOverflow) as exc:
                last_error = exc
                self.policy.record(
                    profile.task_class, route.model, success=False, node_id=node_id
                )
                if not allow_escalation:
                    raise
                next_route = self._next_route(request, route, tried, reason=type(exc).__name__)
                if next_route is None:
                    raise
                route = next_route
                continue
            except BudgetExhausted:
                raise
            except ModelError as exc:
                last_error = exc
                self._emit_error(request, route, exc)
                next_route = self._next_route(request, route, tried, reason="model error")
                if next_route is None:
                    raise
                route = next_route
                continue

            return completion

        raise last_error or ModelError("completion failed")  # pragma: no cover

    def structured(
        self,
        messages: list[Message],
        profile: TaskProfile,
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Complete and return the validated parsed value, not the raw text."""
        completion = self.complete(messages, profile, schema=schema, **kwargs)
        if completion.parsed is None:
            raise MalformedOutput("structured completion produced no value", model=completion.model)
        return completion.parsed

    # -- internals -------------------------------------------------------

    #: Task classes where a weak-then-strong pair teaches something reusable.
    #: Planning and summarization are excluded: their output is project-specific
    #: prose, so retaining it costs ledger space and yields no transferable rule.
    LEARNABLE = frozenset({
        "implementation", "debugging", "refactoring", "test_authoring", "code_review",
    })

    def _transcript(self, request: Request, completion: Completion) -> dict[str, Any]:
        """The output text to retain on the response event, if any.

        Retained only for node-attached calls in learnable classes, and
        truncated. The point is to make escalation pairs recoverable -- knowing
        that local failed and codex succeeded is useless without knowing what
        each produced -- while keeping the ledger a log rather than an archive.
        """
        memory = self.config.memory
        if not memory.keep_transcripts or not request.node_id:
            return {}
        if str(request.profile.task_class) not in self.LEARNABLE:
            return {}
        limit = max(0, memory.transcript_max_chars)
        if not limit:
            return {}
        text = completion.text or ""
        return {
            "text": text[:limit],
            "text_truncated": len(text) > limit,
        }

    def _attempt(self, request: Request, route: Route) -> Completion:
        spec = route.spec
        provider = self.registry.provider_for(spec)
        messages = list(request.messages)

        # A server that cannot constrain decoding needs the schema in the
        # prompt. One that can does not, and adding it there wastes tokens on
        # every single call -- which adds up to real money over a week.
        if request.schema and not spec.supports_json_schema:
            messages.append(Message("system", schema_instruction(request.schema)))

        call = Request(
            messages=messages,
            profile=request.profile,
            tools=request.tools,
            schema=request.schema,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            stop=request.stop,
            no_cache=request.no_cache,
            node_id=request.node_id,
            extra=request.extra,
        )

        # The echo stub is installed *under the real provider's name*, so its
        # cache key is byte-identical to the real model's. Sharing the cache
        # would let a `--dry-run` poison a subsequent real run with skeleton
        # answers -- and the poisoning is invisible, because a cache hit looks
        # exactly like a cheap success. Stubs neither read nor write the cache.
        stubbed = getattr(provider, "stub", False)
        key = cache_key(call, spec.name) if not request.no_cache and not stubbed else ""
        if key:
            hit = self.cache.get(key)
            if hit is not None:
                self.metrics.incr("model.cache_hit", model=spec.name)
                self.ledger.append(
                    Event(
                        type=EventType.MODEL_CACHE_HIT,
                        node_id=request.node_id,
                        payload={"model": spec.name, "task_class": str(request.profile.task_class)},
                    )
                )
                if request.schema:
                    value, errors = parse_and_validate(hit.text, request.schema)
                    if errors:
                        # A cached entry that no longer validates means the
                        # schema changed. Drop it and do the work.
                        log.debug("cached response no longer valid for schema", model=spec.name)
                    else:
                        hit.parsed = value
                        return hit
                else:
                    return hit

        estimated = route.estimated_cost
        reserved_output = request.max_output_tokens or min(spec.max_output_tokens // 4, 4096)
        self.budget.check_and_reserve(
            estimated,
            hosted=spec.hosted,
            estimated_output_tokens=reserved_output,
            node_id=request.node_id,
            escalation=request.profile.attempt > 0,
        )
        try:
            completion = self._call_with_repair(provider, call, route)

            # Record inside the reservation, not after it. Releasing first left
            # a window in which this call was neither reserved nor yet visible
            # in `spend`, so a concurrent worker could be admitted past a
            # ceiling this call had already consumed.
            #
            # Stub tokens are not real tokens: recording them would skew the
            # local/cloud split that drives `cloud_fraction_target`.
            if not completion.stub:
                self.budget.record(
                    model=spec.name,
                    tier=spec.tier,
                    hosted=spec.hosted,
                    cost=completion.cost,
                    input_tokens=completion.usage.input_tokens,
                    output_tokens=completion.usage.output_tokens,
                    cached_tokens=completion.usage.cached_input_tokens,
                    node_id=request.node_id,
                    task_class=str(request.profile.task_class),
                    escalation=request.profile.attempt > 0,
                )
        finally:
            self.budget.release(
                estimated,
                hosted=spec.hosted,
                output_tokens=reserved_output,
                node_id=request.node_id,
            )
        self.metrics.observe("model.latency", completion.latency, model=spec.name)
        self.metrics.observe("model.output_tokens", completion.usage.output_tokens, model=spec.name)
        self.ledger.append(
            Event(
                type=EventType.MODEL_RESPONSE,
                node_id=request.node_id,
                payload={
                    **completion.to_dict(),
                    "task_class": str(request.profile.task_class),
                    "label": request.profile.label,
                    **self._transcript(request, completion),
                },
            )
        )

        if key:
            self.cache.put(key, completion)

        # A valid schema proves only that the model followed the response
        # format, not that the proposed code or plan was correct.  Treating it
        # as task success double-counted calls (schema success here, then a gate
        # verdict in the orchestrator) and trained routing on formatting rather
        # than outcomes.  The orchestrator records the one final verdict for
        # the attempt after validation.  Malformed output is still recorded as
        # a failure above because no downstream verdict can exist for it.
        return completion

    def _call_with_repair(self, provider: Any, call: Request, route: Route) -> Completion:
        """Call the provider, then validate and repair structured output."""
        messages = list(call.messages)
        attempts = 0
        while True:
            attempts += 1
            attempt_call = Request(
                messages=messages,
                profile=call.profile,
                tools=call.tools,
                schema=call.schema,
                max_output_tokens=call.max_output_tokens,
                temperature=call.temperature,
                stop=call.stop,
                node_id=call.node_id,
                extra=call.extra,
            )
            completion = provider.complete(attempt_call, route.spec)
            completion.attempts = attempts

            if completion.truncated and not call.schema:
                log.warn("completion hit the output limit", model=route.spec.name)

            if not call.schema:
                return completion

            try:
                value, errors = parse_and_validate(completion.text, call.schema)
            except MalformedOutput as exc:
                errors = [str(exc)]
                value = None

            if not errors:
                completion.parsed = value
                return completion

            self.metrics.incr("model.schema_violation", model=route.spec.name)
            log.debug(
                "structured output failed validation",
                model=route.spec.name,
                attempt=attempts,
                errors=errors[:3],
            )
            if attempts > MAX_REPAIRS:
                raise MalformedOutput(
                    "structured output invalid after repair attempts",
                    model=route.spec.name,
                    errors=errors[:5],
                )
            messages = [
                *messages,
                Message("assistant", completion.text[:4000]),
                Message("user", repair_instruction(errors, call.schema)),
            ]

    def _next_route(
        self, request: Request, current: Route, tried: set[str], *, reason: str
    ) -> Route | None:
        route = self.router.escalate(
            request.profile,
            current.model,
            messages=request.messages,
            node_id=request.node_id,
            reason=reason,
        )
        if route is None or route.model in tried:
            return None
        return route

    def _emit_route(self, route: Route, request: Request) -> None:
        self.ledger.append(
            Event(
                type=EventType.MODEL_REQUEST,
                node_id=request.node_id,
                payload={
                    "route": route.to_dict(),
                    "task_class": str(request.profile.task_class),
                    "label": request.profile.label,
                    "estimated_input_tokens": estimate_messages(request.messages),
                    "has_schema": request.schema is not None,
                    "tools": [t.name for t in request.tools],
                },
            )
        )

    def _bump_budget(
        self, request: Request, route: Route, exc: Exception, bumps: int
    ) -> bool:
        """Give the model more room, or turn thinking off. False when neither helps.

        Doubling blindly is wrong on a local model in two ways. First, tokens are
        wall-clock: on the pinball run a bump to 131k output tokens meant a
        forty-minute generation, and three of them in sequence looked exactly
        like a hang. Second, the ceiling is not the rung's configured maximum but
        what is left of the context window after the prompt -- asking for more
        output than can physically fit is a request that cannot succeed, so it
        buys another forty minutes and fails identically.

        When there is no headroom left, the last thing to try before paying for a
        frontier rung is the opposite of a bigger budget: no thinking at all. The
        failure is that the model thought until it ran out of room, and a model
        that answers directly usually answers.
        """
        if bumps >= MAX_BUDGET_BUMPS:
            log.warn("output budget still exhausted after bumps", model=route.model)
            return False

        spec = route.spec
        prompt_tokens = estimate_messages(request.messages)
        # Leave the prompt room to grow: a repair instruction or a schema
        # reminder is appended on later attempts.
        headroom = spec.context_window - prompt_tokens - _CONTEXT_MARGIN
        current = request.max_output_tokens or spec.max_output_tokens
        ceiling = min(spec.max_output_tokens * 4, headroom)

        # The context window is not the only wall. Output tokens are wall-clock,
        # and a budget the model cannot emit before its own socket deadline is a
        # request that always fails -- after burning the entire timeout. At 12
        # tok/s the 4x ceiling is three hours against a one-hour timeout, so
        # without this the bump reliably converts a recoverable overflow into a
        # read timeout that reads as a network fault.
        deliverable = spec.deliverable_tokens()
        if deliverable:
            ceiling = min(ceiling, deliverable)

        # A bump has to be big enough to change the outcome. A model that just
        # spent 29k tokens thinking and stopped mid-thought does not finish
        # because it was handed 12% more; it thinks 12% longer and stops again.
        # Observed live: the clock ceiling left exactly that much slack, and
        # taking it would have cost fifty minutes to learn nothing. Below the
        # threshold, skip straight to the thing that does work -- no thinking.
        if ceiling >= current * _MIN_USEFUL_BUMP:
            request.max_output_tokens = min(ceiling, max(current * 2, 8192))
            self.metrics.incr("model.reasoning_overflow", model=route.model)
            log.info(
                "retrying with a larger output budget",
                model=route.model,
                max_output_tokens=request.max_output_tokens,
                reasoning_chars=getattr(exc, "context", {}).get("reasoning_chars"),
            )
            return True

        if not self._thinking_disabled(request) and spec.extra.get("thinking"):
            params = dict(request.extra.get("provider_params") or {})
            params["chat_template_kwargs"] = {
                **(params.get("chat_template_kwargs") or {}),
                "enable_thinking": False,
            }
            request.extra["provider_params"] = params
            self.metrics.incr("model.thinking_disabled", model=route.model)
            # Name the wall that was actually hit. "No context headroom" was
            # printed for both, and read as a full context window when the real
            # limit was the clock: 15k of a 131k window used, and the budget
            # still could not usefully grow. A diagnostic that names the wrong
            # constraint sends the next person to the wrong knob.
            log.info(
                "no useful budget increase available; retrying with thinking disabled",
                model=route.model,
                limited_by=_limiting_wall(headroom, deliverable),
                prompt_tokens=prompt_tokens,
                context_window=spec.context_window,
                current_budget=current,
                ceiling=ceiling,
            )
            return True

        log.warn(
            "reasoning overflow with no headroom and no thinking to disable",
            model=route.model,
            limited_by=_limiting_wall(headroom, deliverable),
            prompt_tokens=prompt_tokens,
            current_budget=current,
            ceiling=ceiling,
        )
        return False

    @staticmethod
    def _thinking_disabled(request: Request) -> bool:
        params = request.extra.get("provider_params") or {}
        kwargs = params.get("chat_template_kwargs") or {}
        return kwargs.get("enable_thinking") is False

    def _record_overflow(self, request: Request, route: Route, exc: Exception) -> None:
        """Account for a generation that produced nothing.

        The tokens were spent -- on a local model, minutes of them. Recording
        only successful calls made a run that was working hard look completely
        idle: no spend, no usage, no ledger event, and an operator reasonably
        concluding the process had hung.
        """
        context = getattr(exc, "context", {}) or {}
        input_tokens = int(context.get("input_tokens") or 0)
        output_tokens = int(context.get("output_tokens") or 0)
        if not (input_tokens or output_tokens):
            return
        self.budget.record(
            model=route.model,
            tier=str(route.spec.tier),
            hosted=str(route.spec.hosted),
            cost=route.spec.cost(input_tokens, output_tokens),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=0,
            node_id=request.node_id,
            task_class=str(request.profile.task_class),
            escalation=False,
        )

    def _emit_error(self, request: Request, route: Route, exc: Exception) -> None:
        from ..errors import ForgeError

        payload = {
            "model": route.model,
            "task_class": str(request.profile.task_class),
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, ForgeError):
            payload["retryable"] = exc.retryable
        self.ledger.append(
            Event(type=EventType.MODEL_ERROR, node_id=request.node_id, payload=payload)
        )

    # -- reporting -------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "cache": self.cache.stats(),
            "budget": self.budget.report(),
            "routing": self.policy.table(),
        }
