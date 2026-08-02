"""Model routing: policy plus availability plus budget.

The policy decides what *should* serve a request. The router decides what
*can*: it filters the ladder to models that are reachable and affordable, asks
the policy to choose among those, and records the decision.

Keeping these separate matters because they fail differently. A policy mistake
is a quality or cost regression that shows up in the retrospective. An
availability mistake -- routing to a provider whose key is missing -- is an
immediate hard failure. Filtering first means the policy never has to reason
about the world's state, only about evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..config import ModelSpec
from ..errors import BudgetExhausted, ConfigError
from ..obs.log import get_logger
from .budget import Budget
from .policy import Decision, RoutingPolicy
from .registry import Registry
from .types import TaskProfile, estimate_messages

log = get_logger("models.router")


@dataclass(slots=True)
class Route:
    """A concrete decision: this model, for this reason, at this price."""

    model: str
    spec: ModelSpec
    decision: Decision
    estimated_cost: float
    estimated_input_tokens: int
    #: Rungs the ladder could have used but could not pay for. Empty on a normal
    #: decision. Recorded because "why did no stronger model help?" is otherwise
    #: unanswerable from the ledger: the route reason says "climbing one rung",
    #: which is true and sounds like escalation even when the rungs above are
    #: gone.
    priced_out: list[str] = field(default_factory=list)

    @property
    def tier(self) -> str:
        return self.spec.tier

    def to_dict(self) -> dict[str, Any]:
        data = {
            "model": self.model,
            "tier": self.tier,
            "estimated_cost": round(self.estimated_cost, 5),
            "estimated_input_tokens": self.estimated_input_tokens,
            **self.decision.to_dict(),
        }
        if self.priced_out:
            data["priced_out"] = self.priced_out
        return data


class Router:
    def __init__(
        self,
        registry: Registry,
        policy: RoutingPolicy,
        budget: Budget,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.budget = budget

    def select(
        self,
        profile: TaskProfile,
        *,
        messages: list | None = None,
        max_output_tokens: int | None = None,
        node_id: str | None = None,
        exclude: set[str] | None = None,
    ) -> Route:
        """Choose a model for this request.

        Returns a :class:`Route` whose model is guaranteed reachable and within
        budget at the estimated size. If nothing on the ladder qualifies, the
        cheapest reachable model is returned and the caller may still fail on
        the hard total ceiling -- Forge prefers doing the work badly-but-locally
        over not doing it at all.
        """
        exclude = set(exclude or ())
        input_tokens = estimate_messages(messages or [])
        ladder = self.registry.usable_ladder()

        # Drop rungs that cannot hold the prompt at all. Context overflow is a
        # deterministic property, so it is a filter, not a risk to weigh.
        fitting = [
            name
            for name in ladder
            if self.registry.spec(name).context_window
            >= input_tokens + (max_output_tokens or self.registry.spec(name).max_output_tokens)
        ]
        if not fitting:
            largest = max(ladder, key=lambda n: self.registry.spec(n).context_window)
            log.warn(
                "prompt exceeds every context window; using the largest",
                estimated_tokens=input_tokens,
                model=largest,
            )
            fitting = [largest]

        specs = {name: self.registry.spec(name) for name in fitting}

        # Subscription quota is a hard constraint that money cannot relieve, so
        # it filters before the cost check rather than being weighed against it.
        within_quota = [name for name in fitting if self._within_quota(specs[name])]
        if within_quota:
            fitting = within_quota
        else:
            log.warn("every rung is out of hourly quota; proceeding and letting the provider decide")

        affordable = [name for name in fitting if self._affordable(specs[name], input_tokens, max_output_tokens, node_id, profile)]
        priced_out = [name for name in fitting if name not in affordable]
        required = specs.get(profile.min_tier or "")
        if required is not None and required.hosted == "cloud" and profile.min_tier not in affordable:
            # A coach or final-solver request must never silently turn into
            # another local attempt and then be recorded as a cloud strategy.
            output_tokens = max_output_tokens or min(required.max_output_tokens // 4, 4096)
            self.budget.check(
                required.cost(input_tokens, output_tokens),
                hosted=required.hosted,
                estimated_output_tokens=output_tokens,
                node_id=node_id,
                escalation=profile.attempt > 0,
            )
            raise BudgetExhausted(
                "required cloud rung is unavailable within budget",
                model=profile.min_tier,
            )
        if not affordable:
            # Everything cloud-side is out of budget; fall back to local, which
            # only the total ceiling can block.
            affordable = [name for name in fitting if specs[name].hosted == "local"] or fitting[:1]
            log.info("budget constrained routing to cheapest rungs", available=affordable)

        # A ladder that has lost its top rungs still reports "climbing one rung"
        # when it escalates, which reads as help arriving. It is not: once a node
        # passes `budget.per_node_cost`, every cloud rung is unaffordable and the
        # strongest model it can reach is the local one it just failed with. One
        # node spent 16 attempts and $12.43 against an $8 ceiling that way, on an
        # error a frontier model would have fixed at once, and nothing in the
        # ledger or the blocked question mentioned money. Recorded so the reason a
        # stronger model never arrived is answerable after the fact.
        if priced_out:
            log.warn(
                "rungs priced out of this node's budget",
                priced_out=priced_out,
                remaining=affordable,
                node=node_id,
            )

        decision = self.policy.decide(
            profile,
            affordable,
            specs,
            cloud_pressure=self.budget.cloud_pressure(),
            exclude=exclude,
        )
        spec = specs.get(decision.model) or self.registry.spec(decision.model)
        estimated = spec.cost(input_tokens, max_output_tokens or spec.max_output_tokens // 4)
        return Route(
            model=decision.model,
            spec=spec,
            decision=decision,
            estimated_cost=estimated,
            estimated_input_tokens=input_tokens,
            priced_out=priced_out,
        )

    def _within_quota(self, spec: ModelSpec) -> bool:
        """Has this model any subscription allowance left this hour?"""
        remaining = self.budget.quota_remaining(spec.name, spec.quota_per_hour)
        if remaining == 0:
            log.info(
                "rung is out of hourly quota, routing around it",
                model=spec.name,
                quota_per_hour=spec.quota_per_hour,
            )
            return False
        return True

    def _affordable(
        self,
        spec: ModelSpec,
        input_tokens: int,
        max_output_tokens: int | None,
        node_id: str | None,
        profile: TaskProfile,
    ) -> bool:
        estimated = spec.cost(input_tokens, max_output_tokens or spec.max_output_tokens // 4)
        try:
            output_tokens = max_output_tokens or min(spec.max_output_tokens // 4, 4096)
            self.budget.check(
                estimated,
                hosted=spec.hosted,
                estimated_output_tokens=output_tokens,
                node_id=node_id,
                escalation=profile.attempt > 0,
            )
        except BudgetExhausted:
            return False
        return True

    def escalate(
        self,
        profile: TaskProfile,
        current: str,
        *,
        messages: list | None = None,
        node_id: str | None = None,
        reason: str = "",
    ) -> Route | None:
        """The next rung up, or ``None`` when already at the top.

        Escalation always excludes every rung at or below the current one. A
        retry that lands on the same model with the same context will produce
        the same answer often enough that the tokens are better spent going up.
        """
        ladder = self.registry.usable_ladder()
        try:
            index = ladder.index(current)
        except ValueError:
            index = -1
        if index >= len(ladder) - 1:
            return None
        exclude = set(ladder[: index + 1])
        # The attempt floor is dropped here, and only here. Its job in `select`
        # is "do not land on a rung this node has already failed on", which
        # `exclude` now states exactly -- by name, against the real ladder.
        # Keeping both compounded them: the floor is an *index* into the
        # candidate list, and that list has just had the bottom rungs removed,
        # so a node on its second attempt escalating from `local` skipped past
        # `local_deep` entirely. That is the one free rung on the ladder, and it
        # was being jumped precisely when cost matters most. The difficulty bump
        # from `escalated()` stays: this task has proved harder than it looked.
        harder = replace(profile.escalated(), attempt=0)
        route = self.select(
            harder,
            messages=messages,
            node_id=node_id,
            exclude=exclude,
        )
        if route.model in exclude:  # pragma: no cover - select honours exclude
            return None
        log.info("escalating", from_model=current, to_model=route.model, reason=reason)
        return route

    def describe(self) -> dict[str, Any]:
        try:
            ladder = self.registry.usable_ladder()
        except ConfigError:
            ladder = []
        return {
            "ladder": ladder,
            "unusable": [n for n in self.registry.ladder if n not in ladder],
            "models": self.registry.describe(),
            "policy": self.policy.table(),
            "budget": self.budget.report(),
            "quota_used": {
                name: self.budget.calls_in_last_hour(name) for name in self.registry.names()
            },
        }
