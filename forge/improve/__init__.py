"""Self-improvement: measuring the platform's own work and changing how it works.

The division of labour here is the point. Everything that can be *computed* from
the ledger is computed -- cost per milestone, escalation rate per task class,
gate flakiness, rework ratio, where the wall-clock went. Only the interpretation
is asked of a model, and it is asked with the numbers in hand.

That ordering matters because a retrospective that asks "how did that go?" gets
a fluent narrative that may be entirely wrong. A retrospective that asks "here
are the numbers; what do they mean and what should change?" gets something a
system can act on.
"""

from .metrics import MilestoneMetrics, compute_metrics
from .promotion import PromotionCandidate, detect_promotions

__all__ = ["MilestoneMetrics", "PromotionCandidate", "compute_metrics", "detect_promotions"]
