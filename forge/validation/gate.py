"""The gate interface and registry.

A gate is a named, cacheable, deterministic check. Writing one means subclassing
:class:`Gate` and implementing :meth:`Gate.run`; registering it makes it
available to every project by name in config, with no other change anywhere.

The ``cache_key`` design is what makes long runs affordable. A gate declares
which inputs it depends on -- usually a subset of the tree -- and the runner
skips any gate whose inputs are unchanged since it last passed. Over a
multi-day run where most edits touch one module, this removes the large majority
of test-suite executions without weakening the guarantee, because an unchanged
input provably produces an unchanged result for a deterministic check.

Gates that are *not* deterministic must say so by setting ``cacheable = False``
(the browser gate does, since a real server and a real renderer are involved).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import NotSupported
from ..obs.log import get_logger
from ..util.hashing import content_hash, tree_hash
from .types import Verdict

if TYPE_CHECKING:  # pragma: no cover
    from ..memory.store import MemoryStore
    from ..workspace.sandbox import Sandbox

log = get_logger("validation.gate")


@dataclass(slots=True)
class GateContext:
    """Everything a gate is allowed to know.

    Note what is absent: no model client, no ledger write access. A gate that
    could call a model would not be deterministic, and a gate that could write
    to the ledger could rewrite history to make itself pass. Both are excluded
    structurally rather than by convention.
    """

    sandbox: Sandbox
    root: Path
    artifacts_dir: Path
    #: Detected toolchain: languages, package manager, discovered commands.
    toolchain: dict[str, Any] = field(default_factory=dict)
    #: Per-gate settings from project config.
    settings: dict[str, Any] = field(default_factory=dict)
    #: Files the current node changed, for gates that can scope their work.
    changed_files: list[str] = field(default_factory=list)
    node_id: str | None = None
    timeout: float = 900.0
    #: Ceiling for anything driving a real browser. Separate from ``timeout``
    #: because the two measure different things: a build may legitimately take
    #: fifteen minutes, a page that has not loaded in one is hung.
    browser_timeout: float = 60.0
    memory: MemoryStore | None = None

    def artifact_path(self, name: str) -> Path:
        target = self.artifacts_dir / (self.node_id or "shared") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def setting(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)


class Gate(ABC):
    """A deterministic check over the workspace."""

    #: Name used in config and in verdicts.
    name: str = "gate"
    #: Human-readable purpose, shown in ``forge gates``.
    description: str = ""
    #: Whether an unchanged tree may reuse a previous verdict.
    cacheable: bool = True
    #: Gates with a lower order run first; cheap gates should run before
    #: expensive ones so a syntax error is caught before a browser boots.
    order: int = 100
    #: A failure here means the node cannot succeed.
    blocking: bool = True
    #: Glob patterns whose contents affect this gate's result. Empty means the
    #: whole tree.
    inputs: tuple[str, ...] = ()

    @abstractmethod
    def run(self, ctx: GateContext) -> Verdict: ...

    def applicable(self, ctx: GateContext) -> bool:
        """Whether this gate makes sense for this project.

        A ``types`` gate on a project with no type checker should skip, not
        fail. Skipping is reported and does not block, but it *is* recorded, so
        the retrospective can notice that a project has been shipping without
        type checking for three days.
        """
        return True

    def cache_key(self, ctx: GateContext) -> str:
        return content_hash(
            self.name,
            self.version(),
            ctx.settings,
            tree_hash(ctx.root, include=self.inputs or None),
        )

    def version(self) -> str:
        """Bump when a gate's logic changes, to invalidate old cached verdicts."""
        return "1"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Gate {self.name}>"


class GateRegistry:
    """Name-to-gate mapping. The extension point for new checks."""

    def __init__(self) -> None:
        self._gates: dict[str, type[Gate]] = {}

    def register(self, gate_cls: type[Gate]) -> type[Gate]:
        if not gate_cls.name or gate_cls.name == "gate":
            raise ValueError(f"{gate_cls.__name__} must define a unique name")
        self._gates[gate_cls.name] = gate_cls
        return gate_cls

    def create(self, name: str, **kwargs: Any) -> Gate:
        try:
            return self._gates[name](**kwargs)
        except KeyError:
            raise NotSupported(f"unknown gate {name!r}", known=sorted(self._gates)) from None

    def names(self) -> list[str]:
        return sorted(self._gates)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": cls.name,
                "description": cls.description,
                "cacheable": cls.cacheable,
                "order": cls.order,
                "blocking": cls.blocking,
            }
            for cls in sorted(self._gates.values(), key=lambda c: c.order)
        ]


gate_registry = GateRegistry()


def register(gate_cls: type[Gate]) -> type[Gate]:
    """Decorator form of :meth:`GateRegistry.register`."""
    return gate_registry.register(gate_cls)
