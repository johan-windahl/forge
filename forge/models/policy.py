"""The adaptive routing policy.

The question this module answers is: *given what we have observed so far, what
is the cheapest model likely to get this task right?*

It is deliberately not "how hard does this look?" -- that framing asks a model to
guess about a model, which is both expensive and unreliable. Instead Forge
measures. Every routed call ends in a recorded success or failure, where
"success" is defined by whatever deterministic gate followed it. Those outcomes
accumulate into a Beta posterior per (task class, ladder rung), and the policy
picks the cheapest rung whose posterior clears a requirement derived from the
task's stakes, difficulty and prior attempts.

Why Beta posteriors rather than a running average:

* They express *uncertainty*, so eight successes out of eight is treated
  differently from eighty out of eighty. The policy stays willing to try the
  cheap rung while evidence is thin, and stops second-guessing once it is not.
* They give a principled exploration rule (Thompson sampling) instead of an
  arbitrary epsilon.
* They start from a prior, so a brand-new project is not paralysed by having no
  data -- it starts with sensible per-class beliefs and moves off them fast.

The policy also reads back-pressure from the budget: if cloud usage is running
above the operator's target, the success requirement for escalating rises, so
the system quietly becomes more frugal without anyone editing a config file.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from typing import Any

from ..config import ImprovementConfig, ModelSpec
from ..kernel.ledger import Ledger
from ..obs.log import get_logger
from ..util.clock import Clock, default_clock
from .types import TaskClass, TaskProfile

log = get_logger("models.policy")


#: Prior belief, per task class, that the *cheapest* rung succeeds. Encoded as
#: pseudo-observations so real evidence overwhelms it within a dozen samples.
#: These numbers are starting points, not conclusions -- the whole design is
#: that they stop mattering once the system has run for a day.
_PRIORS: dict[str, tuple[float, float]] = {
    TaskClass.CLASSIFICATION: (9.0, 1.0),
    TaskClass.EXTRACTION: (8.0, 2.0),
    TaskClass.SUMMARIZATION: (8.0, 2.0),
    TaskClass.DOCUMENTATION: (7.0, 3.0),
    TaskClass.TEST_AUTHORING: (6.0, 4.0),
    TaskClass.IMPLEMENTATION: (6.0, 4.0),
    TaskClass.REFACTORING: (6.0, 4.0),
    TaskClass.CODE_REVIEW: (5.0, 5.0),
    TaskClass.DEBUGGING: (4.0, 6.0),
    TaskClass.VISUAL_JUDGEMENT: (4.0, 6.0),
    TaskClass.RETROSPECTIVE: (4.0, 6.0),
    TaskClass.ARCHITECTURE: (3.0, 7.0),
    TaskClass.PLANNING: (3.0, 7.0),
}
_DEFAULT_PRIOR = (5.0, 5.0)

#: Higher rungs inherit a share of the belief in lower rungs plus a bonus: a
#: stronger model is assumed at least as capable, which stops the policy from
#: needing to re-learn everything for every rung.
_RUNG_BONUS = 0.6

#: Cold-start ceiling. Below this stakes level, a first attempt with no evidence
#: is capped at this rung index so it produces evidence cheaply instead of
#: spending the most expensive resource on a guess. See RoutingPolicy.decide.
COLD_START_MAX_RUNG = 1
COLD_START_CONFIDENCE = 0.35
COLD_START_STAKES_EXEMPTION = 0.85


@dataclass(slots=True)
class Observation:
    successes: float = 0.0
    failures: float = 0.0
    cost: float = 0.0
    latency: float = 0.0

    @property
    def n(self) -> float:
        return self.successes + self.failures

    def mean_cost(self) -> float:
        return self.cost / self.n if self.n else 0.0


@dataclass(slots=True)
class Decision:
    """A routing choice, with the reasoning that produced it.

    The reasoning is written to the ledger. When the retrospective later asks
    "why did we spend 40% of the budget on debugging?", the answer is in the
    log as data, not reconstructed by asking a model to speculate.
    """

    model: str
    reason: str
    expected_success: float
    required_success: float
    considered: list[dict[str, Any]] = field(default_factory=list)
    escalation: bool = False
    explored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "reason": self.reason,
            "expected_success": round(self.expected_success, 4),
            "required_success": round(self.required_success, 4),
            "escalation": self.escalation,
            "explored": self.explored,
            "considered": self.considered,
        }


class RoutingPolicy:
    """Chooses a ladder rung from measured outcomes.

    Statistics live in the ledger (``routing_stats``), so they persist across
    restarts and are shared by every worker. A small in-process cache avoids a
    query per call; it is invalidated on every recorded outcome.
    """

    def __init__(
        self,
        ledger: Ledger,
        *,
        config: ImprovementConfig | None = None,
        clock: Clock | None = None,
        seed: int | None = None,
    ) -> None:
        self.ledger = ledger
        self.config = config or ImprovementConfig()
        self._clock = clock or default_clock()
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], Observation] | None = None
        self._rng = random.Random(seed)

    # -- statistics ------------------------------------------------------

    def _stats(self) -> dict[tuple[str, str], Observation]:
        with self._lock:
            if self._cache is None:
                cache: dict[tuple[str, str], Observation] = {}
                for row in self.ledger.conn.execute("SELECT * FROM routing_stats_v2"):
                    cache[(row["task_class"], row["tier"])] = Observation(
                        successes=float(row["successes"]),
                        failures=float(row["failures"]),
                        cost=float(row["cost"]),
                        latency=float(row["latency"]),
                    )
                self._cache = cache
            return self._cache

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    def posterior(self, task_class: str, rung: str, rung_index: int) -> tuple[float, float]:
        """Beta parameters for P(success | class, rung)."""
        stats = self._stats()
        prior_a, prior_b = _PRIORS.get(task_class, _DEFAULT_PRIOR)
        # Stronger rungs start from a more optimistic prior, scaled by position.
        shift = _RUNG_BONUS * rung_index
        prior_a += shift
        prior_b = max(0.5, prior_b - shift)
        observed = stats.get((task_class, rung), Observation())
        return prior_a + observed.successes, prior_b + observed.failures

    def success_estimate(self, task_class: str, rung: str, rung_index: int) -> float:
        a, b = self.posterior(task_class, rung, rung_index)
        return a / (a + b)

    def sample_estimate(self, task_class: str, rung: str, rung_index: int) -> float:
        """Thompson sample: draw from the posterior instead of using its mean.

        This is what makes exploration self-limiting. While a rung's posterior
        is wide, draws vary a lot and the cheap rung sometimes wins even when
        its mean is below requirement -- so it gets tried, and the posterior
        narrows. Once the posterior is tight, draws cluster at the mean and
        exploration stops on its own. No decay schedule to tune.
        """
        a, b = self.posterior(task_class, rung, rung_index)
        return self._rng.betavariate(max(0.01, a), max(0.01, b))

    def confidence(self, task_class: str, rung: str) -> float:
        """How much evidence backs this cell, normalised to [0, 1]."""
        observed = self._stats().get((task_class, rung), Observation())
        target = max(1, self.config.min_samples_for_routing_update)
        return min(1.0, observed.n / (target * 2))

    # -- requirement -----------------------------------------------------

    def required_success(self, profile: TaskProfile, cloud_pressure: float = 0.0) -> float:
        """The success probability a rung must clear to be chosen.

        Rises with stakes (a wrong architecture is expensive to unwind), with
        difficulty, and with each failed attempt. Falls back down when cloud
        spend is running hot, which makes the cheap rung acceptable more often.
        """
        base = 0.55
        requirement = base + 0.25 * profile.stakes + 0.15 * profile.difficulty
        requirement += 0.12 * min(3, profile.attempt)
        # Under cloud pressure, tolerate a lower expected success rate locally
        # rather than escalating: a retry on the local model is nearly free.
        requirement -= 0.20 * cloud_pressure
        return max(0.30, min(0.97, requirement))

    # -- selection -------------------------------------------------------

    def decide(
        self,
        profile: TaskProfile,
        ladder: list[str],
        specs: dict[str, ModelSpec],
        *,
        cloud_pressure: float = 0.0,
        exclude: set[str] | None = None,
    ) -> Decision:
        """Pick a rung. ``ladder`` must be ordered cheapest to strongest."""
        exclude = exclude or set()
        candidates = [
            (index, name)
            for index, name in enumerate(ladder)
            if name not in exclude and self._eligible(profile, specs[name])
        ]
        if not candidates:
            # Every preference has been excluded; fall back to the strongest
            # thing still allowed rather than failing the node outright.
            fallback = next((n for n in reversed(ladder) if n not in exclude), ladder[-1])
            return Decision(
                model=fallback,
                reason="all preferred rungs excluded; using last resort",
                expected_success=self.success_estimate(profile.task_class, fallback, len(ladder) - 1),
                required_success=0.0,
            )

        # Prior attempts force a floor: repeating a failed attempt on the same
        # rung with the same context is the definition of wasted tokens.
        floor = 0
        if profile.attempt > 0:
            floor = min(profile.attempt, len(candidates) - 1)
        if profile.min_tier:
            floor = max(floor, next((i for i, (_, n) in enumerate(candidates) if n == profile.min_tier), 0))

        # Cold-start ceiling.
        #
        # With no evidence, every rung's posterior is its prior, and the priors
        # are deliberately pessimistic about the cheap rungs -- so a first-ever
        # request would jump straight to the strongest rung, spend the most
        # expensive resource available, and learn nothing about whether the
        # cheap one would have worked.
        #
        # That is backwards. The first attempt at a low-stakes task should be
        # cheap precisely *because* there is no evidence: its job is to produce
        # some. Failure escalates on the next attempt, which is the mechanism
        # that is supposed to allocate expensive capacity.
        #
        # High-stakes work is exempt. A bad plan or architecture is not caught
        # by any gate, so there is no cheap failure to learn from, and the whole
        # point of the escalation ladder does not apply.
        ceiling = len(candidates) - 1
        if profile.attempt == 0 and profile.stakes < COLD_START_STAKES_EXEMPTION:
            evidence = max(
                (self.confidence(profile.task_class, name) for _, name in candidates),
                default=0.0,
            )
            if evidence < COLD_START_CONFIDENCE:
                ceiling = min(ceiling, COLD_START_MAX_RUNG)

        requirement = self.required_success(profile, cloud_pressure)
        considered: list[dict[str, Any]] = []
        chosen: tuple[int, str] | None = None
        explored = False

        for position, (rung_index, name) in enumerate(candidates):
            estimate = self.success_estimate(profile.task_class, name, rung_index)
            sampled = self.sample_estimate(profile.task_class, name, rung_index)
            spec = specs[name]
            considered.append(
                {
                    "model": name,
                    "tier": spec.tier,
                    "estimate": round(estimate, 4),
                    "sampled": round(sampled, 4),
                    "confidence": round(self.confidence(profile.task_class, name), 3),
                }
            )
            if position < floor:
                continue
            if position > ceiling and chosen is None and position > floor:
                # Capped by the cold-start rule; stop looking upward.
                break
            if chosen is not None:
                continue
            if sampled >= requirement:
                chosen = (rung_index, name)
                explored = sampled >= requirement > estimate
            elif estimate >= requirement:
                chosen = (rung_index, name)

        if chosen is None:
            # Nothing cleared the bar. What to do next depends on why we are
            # here, and the two cases want opposite things.
            if profile.attempt > 0:
                # We are escalating after a real failure. Climb exactly one rung
                # per attempt: the floor already encodes the attempt count.
                # Leaping to the top on the first failure would spend the
                # scarcest resource before trying the free one above it, and
                # would never generate evidence about the rungs in between.
                index = min(floor, len(candidates) - 1)
                reason = f"attempt {profile.attempt}: climbing one rung to {candidates[index][1]!r}"
            else:
                index = min(ceiling, len(candidates) - 1)
                reason = (
                    "no rung met the requirement; capped at the cold-start ceiling "
                    "to produce evidence cheaply"
                    if index < len(candidates) - 1
                    else "no rung met the requirement; using the strongest available"
                )
            chosen = (candidates[index][0], candidates[index][1])
        elif explored:
            reason = "exploring a cheaper rung whose posterior is still wide"
        elif chosen[1] == candidates[floor][1] and floor > 0:
            reason = f"forced up {floor} rung(s) by {profile.attempt} prior attempt(s)"
        else:
            reason = "cheapest rung meeting the success requirement"

        return Decision(
            model=chosen[1],
            reason=reason,
            expected_success=self.success_estimate(profile.task_class, chosen[1], chosen[0]),
            required_success=requirement,
            considered=considered,
            escalation=profile.attempt > 0 or chosen[0] > 0,
            explored=explored,
        )

    @staticmethod
    def _eligible(profile: TaskProfile, spec: ModelSpec) -> bool:
        if profile.needs_vision and not spec.supports_vision:
            return False
        if profile.needs_tools and not spec.supports_tools:
            return False
        return not (
            profile.max_tier
            and spec.tier == "frontier"
            and profile.max_tier != "frontier"
        )

    # -- feedback --------------------------------------------------------

    def record(
        self,
        task_class: str,
        rung: str,
        *,
        success: bool,
        cost: float = 0.0,
        latency: float = 0.0,
        node_id: str | None = None,
    ) -> None:
        """Record the outcome of a routed call.

        Called by whatever *deterministically* determined the outcome -- a gate
        verdict, a schema validation, a patch that applied. Outcomes judged only
        by another model are recorded with reduced weight by the caller, because
        letting model opinion train the router would close a loop with no ground
        truth in it.
        """
        from ..kernel.events import Event, EventType

        self.ledger.append(
            Event(
                type=EventType.ROUTE_DECIDED,
                node_id=node_id,
                payload={
                    "task_class": str(task_class),
                    # routing_stats.tier stores the ladder *rung*, which is a
                    # model name. Kept under this key for schema stability.
                    "tier": rung,
                    "outcome": "success" if success else "failure",
                    "feedback_version": 2,
                    "cost": cost,
                    "latency": latency,
                },
            )
        )
        self.invalidate()

    # -- reporting -------------------------------------------------------

    def table(self) -> list[dict[str, Any]]:
        """Current beliefs, for ``forge policy`` and the retrospective."""
        rows: list[dict[str, Any]] = []
        for (task_class, rung), observed in sorted(self._stats().items()):
            a, b = self.posterior(task_class, rung, 0)
            rows.append(
                {
                    "task_class": task_class,
                    "rung": rung,
                    "successes": observed.successes,
                    "failures": observed.failures,
                    "posterior_mean": round(a / (a + b), 4),
                    "confidence": round(self.confidence(task_class, rung), 3),
                    "mean_cost": round(observed.mean_cost(), 5),
                }
            )
        return rows

    def recommendations(self) -> list[str]:
        """Human-readable policy observations for the retrospective.

        Turning statistics into sentences is not decoration: these lines become
        lessons that persist across projects, so a future run starts already
        knowing that, say, debugging on this codebase needs a stronger model.
        """
        notes: list[str] = []
        minimum = self.config.min_samples_for_routing_update
        for (task_class, rung), observed in self._stats().items():
            if observed.n < minimum:
                continue
            rate = observed.successes / observed.n
            if rate < 0.4:
                notes.append(
                    f"{task_class} on '{rung}' succeeds {rate:.0%} of the time over "
                    f"{int(observed.n)} attempts -- consider starting this class one rung higher."
                )
            elif rate > 0.9 and rung != "local":
                notes.append(
                    f"{task_class} on '{rung}' succeeds {rate:.0%} of the time -- try the rung "
                    f"below it to cut cost."
                )
        return notes
