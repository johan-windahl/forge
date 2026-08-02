"""Spend accounting and ceilings.

Three properties an autonomous system needs from a budget, in order of
importance:

1. **It cannot be exceeded by accident.** Every model call passes through
   :meth:`Budget.reserve` before it happens, not after.
2. **It survives restarts.** Spend is derived from the ledger, so a crash and
   restart resumes with the same numbers rather than a fresh allowance.
3. **It degrades rather than stops.** Hitting the daily ceiling should mean
   "keep working locally", not "halt". Only the total ceiling stops the run.

The escalation reserve deserves a note: a fixed fraction of the total budget is
spendable *only* by escalations. Without it, a project that routinely nudges
frontier models for routine work arrives at its hardest problem with nothing
left to spend on it -- precisely backwards.
"""

from __future__ import annotations

import datetime
import threading
from dataclasses import dataclass
from typing import Any

from ..config import BudgetConfig
from ..errors import BudgetExhausted
from ..kernel.events import EventType
from ..kernel.ledger import Ledger
from ..obs.log import get_logger
from ..util.clock import Clock, default_clock

log = get_logger("models.budget")


@dataclass(slots=True)
class SpendSnapshot:
    total: float = 0.0
    today: float = 0.0
    cloud_tokens: int = 0
    local_tokens: int = 0
    escalation_spend: float = 0.0
    calls: int = 0

    @property
    def cloud_fraction(self) -> float:
        """Share of generated tokens produced by cloud models."""
        total = self.cloud_tokens + self.local_tokens
        return self.cloud_tokens / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "today": round(self.today, 4),
            "cloud_tokens": self.cloud_tokens,
            "local_tokens": self.local_tokens,
            "cloud_fraction": round(self.cloud_fraction, 4),
            "escalation_spend": round(self.escalation_spend, 4),
            "calls": self.calls,
        }


class Budget:
    """Enforces cost ceilings and reports how spend is distributed."""

    def __init__(self, config: BudgetConfig, ledger: Ledger, clock: Clock | None = None) -> None:
        self.config = config
        self.ledger = ledger
        self._clock = clock or default_clock()
        self._lock = threading.Lock()
        self._reserved: float = 0.0  # in-flight, not yet recorded
        self._reserved_cloud_tokens: int = 0
        self._reserved_by_node: dict[str, float] = {}
        self._warned: set[str] = set()

    # -- queries ---------------------------------------------------------

    def snapshot(self) -> SpendSnapshot:
        today = datetime.datetime.fromtimestamp(self._clock.now(), tz=datetime.UTC).strftime("%Y-%m-%d")
        row = self.ledger.conn.execute(
            """SELECT COALESCE(SUM(cost), 0) AS total,
                      COALESCE(SUM(CASE WHEN day = ? THEN cost ELSE 0 END), 0) AS today,
                      COALESCE(SUM(CASE WHEN hosted = 'local' THEN output_tokens ELSE 0 END), 0) AS local_tokens,
                      COALESCE(SUM(CASE WHEN hosted != 'local' THEN output_tokens ELSE 0 END), 0) AS cloud_tokens,
                      COALESCE(SUM(CASE WHEN escalation = 1 THEN cost ELSE 0 END), 0) AS escalation_spend,
                      COUNT(*) AS calls
               FROM spend""",
            (today,),
        ).fetchone()
        return SpendSnapshot(
            total=float(row["total"]),
            today=float(row["today"]),
            cloud_tokens=int(row["cloud_tokens"]),
            local_tokens=int(row["local_tokens"]),
            escalation_spend=float(row["escalation_spend"]),
            calls=int(row["calls"]),
        )

    def calls_in_last_hour(self, model: str) -> int:
        """How many calls a model has served in the past rolling hour.

        Derived from the ledger rather than an in-process counter, so a restart
        does not hand a subscription-backed model a fresh allowance it has not
        actually earned. That matters on a platform designed to be restarted.
        """
        cutoff = self._clock.now() - 3600.0
        row = self.ledger.conn.execute(
            "SELECT COUNT(*) AS c FROM spend WHERE model = ? AND ts >= ?", (model, cutoff)
        ).fetchone()
        return int(row["c"])

    def quota_remaining(self, model: str, quota_per_hour: int) -> int:
        """Calls left this hour. ``-1`` means unlimited."""
        if quota_per_hour <= 0:
            return -1
        return max(0, quota_per_hour - self.calls_in_last_hour(model))

    def node_spend(self, node_id: str) -> float:
        row = self.ledger.conn.execute(
            "SELECT COALESCE(SUM(cost), 0) AS c FROM spend WHERE node_id = ?", (node_id,)
        ).fetchone()
        return float(row["c"])

    def remaining(self) -> float:
        return max(0.0, self.config.total_cost - self.snapshot().total - self._reserved)

    # -- admission control ----------------------------------------------

    def check(
        self,
        estimated_cost: float,
        *,
        hosted: str,
        estimated_output_tokens: int = 0,
        node_id: str | None = None,
        escalation: bool = False,
    ) -> None:
        """Raise :class:`BudgetExhausted` if this call must not be made.

        Prefer :meth:`check_and_reserve`. A bare check is advisory: between it
        and the caller's ``reserve`` another worker can pass the same check, so
        two concurrent calls can jointly cross a ceiling neither would cross
        alone. This entry point remains for callers that only want to know.
        """
        with self._lock:
            self._check_locked(
                estimated_cost,
                hosted=hosted,
                estimated_output_tokens=estimated_output_tokens,
                node_id=node_id,
                escalation=escalation,
            )

    def check_and_reserve(
        self,
        estimated_cost: float,
        *,
        hosted: str,
        estimated_output_tokens: int = 0,
        node_id: str | None = None,
        escalation: bool = False,
    ) -> None:
        """Admit and reserve under one lock, or raise :class:`BudgetExhausted`.

        The check and the reservation have to be one atomic step. Separately
        they are a time-of-check/time-of-use race: with ``scheduler.workers``
        above one, every worker can observe the same pre-reservation state and
        all of them pass. That defeats the module's first stated property.
        """
        with self._lock:
            self._check_locked(
                estimated_cost,
                hosted=hosted,
                estimated_output_tokens=estimated_output_tokens,
                node_id=node_id,
                escalation=escalation,
            )
            self._reserve_locked(
                estimated_cost,
                hosted=hosted,
                output_tokens=estimated_output_tokens,
                node_id=node_id,
            )

    def _check_locked(
        self,
        estimated_cost: float,
        *,
        hosted: str,
        estimated_output_tokens: int = 0,
        node_id: str | None = None,
        escalation: bool = False,
    ) -> None:
        """Admission logic. ``self._lock`` must already be held.

        Locally hosted calls are only ever refused by the *total* ceiling.
        Throttling them would trade the resource we have plenty of -- local
        compute -- for the one we are trying to conserve, which is backwards.
        """
        snapshot = self.snapshot()
        projected = snapshot.total + self._reserved + estimated_cost

        if self.config.enforce_cost_limits and projected > self.config.total_cost:
            raise BudgetExhausted(
                "project cost ceiling reached",
                limit=self.config.total_cost,
                spent=round(snapshot.total, 4),
                requested=round(estimated_cost, 4),
            )

        if hosted == "local":
            return

        # Reserve: routine cloud work may not eat into the escalation pool.
        reserve = self.config.total_cost * self.config.escalation_reserve
        if (
            self.config.enforce_cost_limits
            and not escalation
            and projected > self.config.total_cost - reserve
        ):
            raise BudgetExhausted(
                "only the escalation reserve remains",
                reserve=round(reserve, 4),
                spent=round(snapshot.total, 4),
            )

        # In-flight reservations are all by definition from right now, so they
        # belong to today's total. Leaving them out let concurrent workers walk
        # past the daily ceiling the same way they walked past the total one.
        if (
            self.config.enforce_cost_limits
            and snapshot.today + self._reserved + estimated_cost > self.config.daily_cost
        ):
            raise BudgetExhausted(
                "daily cloud cost ceiling reached",
                limit=self.config.daily_cost,
                today=round(snapshot.today, 4),
            )

        if self.config.enforce_cost_limits and node_id:
            spent = self.node_spend(node_id) + self._reserved_by_node.get(node_id, 0.0)
            if spent + estimated_cost > self.config.per_node_cost:
                raise BudgetExhausted(
                    "per-node cost ceiling reached",
                    limit=self.config.per_node_cost,
                    node_spent=round(spent, 4),
                    node_id=node_id,
                )

        # A hard admission boundary, unlike cloud_fraction_target's routing
        # feedback. Cost ceilings above retain their more specific diagnostics.
        # Reserve projected output across concurrent calls so two workers cannot
        # each pass the check and jointly cross the ceiling.
        projected_cloud = (
            snapshot.cloud_tokens
            + self._reserved_cloud_tokens
            + max(0, estimated_output_tokens)
        )
        projected_total = (
            snapshot.cloud_tokens
            + snapshot.local_tokens
            + self._reserved_cloud_tokens
            + max(0, estimated_output_tokens)
        )
        projected_fraction = projected_cloud / projected_total if projected_total else 0.0
        # Direct callers predating token admission pass no estimate; retain
        # their cost-only semantics. ModelClient and Router always provide one.
        if (
            node_id is not None
            and estimated_output_tokens > 0
            and projected_fraction > self.config.max_cloud_fraction
        ):
            raise BudgetExhausted(
                "hard cloud-generated-token ceiling would be exceeded",
                limit=self.config.max_cloud_fraction,
                current=round(snapshot.cloud_fraction, 4),
                projected=round(projected_fraction, 4),
            )

    def reserve(
        self,
        estimated_cost: float,
        *,
        hosted: str = "local",
        output_tokens: int = 0,
        node_id: str | None = None,
    ) -> None:
        with self._lock:
            self._reserve_locked(
                estimated_cost, hosted=hosted, output_tokens=output_tokens, node_id=node_id
            )

    def _reserve_locked(
        self,
        estimated_cost: float,
        *,
        hosted: str = "local",
        output_tokens: int = 0,
        node_id: str | None = None,
    ) -> None:
        """``self._lock`` must already be held."""
        self._reserved += estimated_cost
        if node_id:
            self._reserved_by_node[node_id] = (
                self._reserved_by_node.get(node_id, 0.0) + estimated_cost
            )
        if hosted != "local":
            self._reserved_cloud_tokens += max(0, output_tokens)

    def release(
        self,
        estimated_cost: float,
        *,
        hosted: str = "local",
        output_tokens: int = 0,
        node_id: str | None = None,
    ) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - estimated_cost)
            if node_id:
                remaining = self._reserved_by_node.get(node_id, 0.0) - estimated_cost
                if remaining > 1e-12:
                    self._reserved_by_node[node_id] = remaining
                else:
                    self._reserved_by_node.pop(node_id, None)
            if hosted != "local":
                self._reserved_cloud_tokens = max(
                    0, self._reserved_cloud_tokens - max(0, output_tokens)
                )

    # -- recording -------------------------------------------------------

    def record(
        self,
        *,
        model: str,
        tier: str,
        hosted: str,
        cost: float,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        node_id: str | None = None,
        task_class: str | None = None,
        escalation: bool = False,
    ) -> None:
        self.ledger.append(
            _spend_event(
                model=model,
                tier=tier,
                hosted=hosted,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                node_id=node_id,
                task_class=task_class,
                escalation=escalation,
            )
        )
        self._maybe_warn()

    def _maybe_warn(self) -> None:
        snapshot = self.snapshot()
        for threshold in (0.5, 0.8, 0.95):
            marker = f"total:{threshold}"
            if snapshot.total >= self.config.total_cost * threshold and marker not in self._warned:
                self._warned.add(marker)
                log.warn(
                    "budget threshold crossed",
                    fraction=threshold,
                    spent=round(snapshot.total, 4),
                    limit=self.config.total_cost,
                )
                self.ledger.emit(
                    EventType.BUDGET_WARNING,
                    fraction=threshold,
                    spent=snapshot.total,
                    limit=self.config.total_cost,
                )

    # -- routing feedback -------------------------------------------------

    def cloud_pressure(self) -> float:
        """How far cloud usage is running above target, in [0, 1].

        The router adds this to its escalation threshold, so a project drifting
        cloud-heavy quietly becomes more reluctant to escalate, and a project
        that has been frugal becomes more willing. It is a proportional
        controller on a quantity the operator actually cares about, rather than
        a hard quota that fails at the worst moment.
        """
        snapshot = self.snapshot()
        generated_total = snapshot.cloud_tokens + snapshot.local_tokens
        if generated_total < 1_000:
            # Bootstrap fallback for providers that report no output usage.
            # This is routing feedback only; the hard ceiling always uses
            # generated tokens and therefore is not distorted by CLI harnesses.
            row = self.ledger.conn.execute(
                """SELECT
                     COALESCE(SUM(CASE WHEN hosted = 'local' THEN input_tokens ELSE 0 END), 0) AS local_input,
                     COALESCE(SUM(CASE WHEN hosted != 'local' THEN input_tokens ELSE 0 END), 0) AS cloud_input
                   FROM spend"""
            ).fetchone()
            local_input = int(row["local_input"])
            cloud_input = int(row["cloud_input"])
            if local_input + cloud_input < 20_000:
                return 0.0
            observed_fraction = cloud_input / (local_input + cloud_input)
        else:
            observed_fraction = snapshot.cloud_fraction
        target = max(1e-6, self.config.cloud_fraction_target)
        excess = (observed_fraction - target) / target
        return max(0.0, min(1.0, excess))

    def report(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        by_model = [
            dict(row)
            for row in self.ledger.conn.execute(
                """SELECT model, tier, hosted, COUNT(*) AS calls,
                          SUM(input_tokens) AS input_tokens,
                          SUM(output_tokens) AS output_tokens,
                          SUM(cached_tokens) AS cached_tokens,
                          ROUND(SUM(cost), 4) AS cost
                   FROM spend GROUP BY model, tier, hosted ORDER BY cost DESC"""
            )
        ]
        return {
            **snapshot.to_dict(),
            "limits": {
                "total": self.config.total_cost,
                "daily": self.config.daily_cost,
                "per_node": self.config.per_node_cost,
                "enforce_cost_limits": self.config.enforce_cost_limits,
                "cloud_fraction_target": self.config.cloud_fraction_target,
                "max_cloud_fraction": self.config.max_cloud_fraction,
            },
            "remaining": round(self.remaining(), 4),
            "cloud_pressure": round(self.cloud_pressure(), 4),
            "by_model": by_model,
        }


def _spend_event(**payload: Any):
    from ..kernel.events import Event

    node_id = payload.pop("node_id", None)
    return Event(type=EventType.BUDGET_SPENT, node_id=node_id, payload=payload)
