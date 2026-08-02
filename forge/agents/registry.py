"""Agent registry: node kind to specialist.

Registration is by decorator so a project or a plugin can add a specialist
without editing Forge, and the orchestrator resolves purely by node kind. The
consequence worth noting: extending the platform with, say, an accessibility
reviewer means writing one class and adding one kind to the planner's
vocabulary. Nothing in the kernel, the model layer or the validation layer
learns about it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import NotSupported

if TYPE_CHECKING:  # pragma: no cover
    from .base import Agent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, type[Agent]] = {}

    def register(self, agent_cls: type[Agent]) -> type[Agent]:
        kind = agent_cls.kind
        if not kind or kind == "agent":
            raise ValueError(f"{agent_cls.__name__} must define a unique kind")
        self._agents[kind] = agent_cls
        return agent_cls

    def create(self, kind: str, **kwargs: Any) -> Agent:
        try:
            return self._agents[kind](**kwargs)
        except KeyError:
            raise NotSupported(f"no agent handles node kind {kind!r}", known=sorted(self._agents)) from None

    def has(self, kind: str) -> bool:
        return kind in self._agents

    def kinds(self) -> list[str]:
        return sorted(self._agents)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "kind": cls.kind,
                "task_class": str(cls.task_class),
                "difficulty": cls.difficulty,
                "stakes": cls.stakes,
                "commits": cls.commits,
            }
            for cls in sorted(self._agents.values(), key=lambda c: c.kind)
        ]


agent_registry = AgentRegistry()


def register(agent_cls: type[Agent]) -> type[Agent]:
    return agent_registry.register(agent_cls)


def build_agent(kind: str, **kwargs: Any) -> Agent:
    _load_builtins()
    return agent_registry.create(kind, **kwargs)


_loaded = False


def _load_builtins() -> None:
    """Import the modules that register the built-in agents.

    Done lazily rather than at package import to keep ``import forge`` cheap and
    to avoid a circular import between the registry and the agents that use it.
    """
    global _loaded
    if _loaded:
        return
    from . import (  # noqa: F401
        coding,
        goal,
        improving,
        planning,
        reviewing,
        shipping,
        validating,
    )

    _loaded = True


def all_kinds() -> list[str]:
    _load_builtins()
    return agent_registry.kinds()
