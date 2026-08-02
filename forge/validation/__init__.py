"""Deterministic validation.

Forge's central quality claim is that *machines decide whether the work is
done*. A compiler, a test runner, a pixel comparison and an HTTP status code are
not opinions. Model judgement is used only where no deterministic check exists
-- "does this level feel good to play?" -- and even there it supplements rather
than replaces the objective gates that ran first.
"""

from .gate import Gate, GateContext, gate_registry
from .runner import GateRunner
from .types import Severity, Verdict

__all__ = ["Gate", "GateContext", "GateRunner", "Severity", "Verdict", "gate_registry"]
