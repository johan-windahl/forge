"""Specialist agents.

An agent is a thin, focused thing: it builds a context, asks a model for a
structured answer, applies that answer to the workspace, and reports. It does
not choose a model, manage a budget, retry, checkpoint or decide what to work on
next -- the kernel and the model layer own those, once, for everybody.

Adding a specialist means writing a class with a ``kind``, a ``task_class`` and
a ``run`` method, then registering it. Nothing else in the platform changes.
"""

from .base import Agent, AgentContext, AgentResult
from .registry import agent_registry, build_agent

__all__ = ["Agent", "AgentContext", "AgentResult", "agent_registry", "build_agent"]
