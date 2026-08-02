"""The agent contract.

Everything an agent may touch arrives in an :class:`AgentContext`. Everything it
produces leaves in an :class:`AgentResult`. Agents never write to the ledger,
never mutate the task graph, and never commit -- they *describe* what should
happen and the orchestrator makes it so, inside a transaction, with a
checkpoint.

That indirection is what makes the system recoverable. An agent that crashed
halfway through has changed nothing; an agent that returned a result has changed
nothing yet either. There is exactly one place where work becomes durable, and
it is not inside a model-driven code path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..kernel.graph import Node, TaskGraph
from ..memory.context import (
    P_ACCEPTANCE,
    P_ARCHITECTURE,
    P_CONVENTIONS,
    P_GOAL,
    P_INTERFACES,
    P_LESSONS,
    P_MEMORY,
    ContextBuilder,
)
from ..memory.records import MemoryKind, MemoryRecord
from ..models.types import Message, TaskClass, TaskProfile
from ..obs.log import get_logger
from ..validation.types import ValidationReport

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config
    from ..memory.lessons import LessonLibrary
    from ..memory.store import MemoryStore
    from ..models.client import ModelClient
    from ..validation.runner import GateRunner
    from ..workspace.git import Repo
    from ..workspace.sandbox import Sandbox

log = get_logger("agents")


@dataclass(slots=True)
class AgentContext:
    """Everything an agent is given."""

    node: Node
    config: Config
    models: ModelClient
    memory: MemoryStore
    lessons: LessonLibrary
    repo: Repo
    sandbox: Sandbox
    gates: GateRunner
    graph: TaskGraph
    toolchain: dict[str, Any] = field(default_factory=dict)
    #: Project-level facts assembled once per run: goal, digest, tree.
    project: dict[str, Any] = field(default_factory=dict)
    artifacts_dir: Path = field(default_factory=Path)

    @property
    def root(self) -> Path:
        return self.repo.path

    @property
    def goal(self) -> str:
        return str(self.project.get("goal", ""))

    @property
    def spec(self) -> dict[str, Any]:
        return self.node.spec

    def logger(self):
        return log.bind(node=self.node.id, kind=self.node.kind)


@dataclass(slots=True)
class ProposedNode:
    """A node an agent wants created. The orchestrator decides whether to."""

    kind: str
    title: str
    spec: dict[str, Any] = field(default_factory=dict)
    #: Indices into the same proposal list, or existing node ids.
    deps: list[Any] = field(default_factory=list)
    priority: int = 100
    milestone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "spec": self.spec,
            "deps": self.deps,
            "priority": self.priority,
            "milestone": self.milestone,
        }


@dataclass(slots=True)
class AgentResult:
    """What an agent produced."""

    success: bool
    summary: str = ""
    #: Free-form output recorded on the node.
    data: dict[str, Any] = field(default_factory=dict)
    #: Records to persist to project memory.
    memory: list[MemoryRecord] = field(default_factory=list)
    #: Nodes to add to the graph.
    nodes: list[ProposedNode] = field(default_factory=list)
    #: Files changed, if the agent wrote any.
    changed_files: list[str] = field(default_factory=list)
    #: Commit message to use, if the agent produced code.
    commit_message: str = ""
    #: Validation outcome, when the agent ran gates itself.
    report: ValidationReport | None = None
    #: Set when the agent believes a stronger model is required.
    needs_escalation: bool = False
    #: Set when only a human can resolve this.
    needs_human: str = ""
    #: Artefacts produced (screenshots, reports).
    artifacts: list[str] = field(default_factory=list)
    #: Marks a milestone as reached, triggering a retrospective.
    milestone_reached: str = ""

    @classmethod
    def failure(cls, summary: str, **kwargs: Any) -> AgentResult:
        return cls(success=False, summary=summary, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "summary": self.summary,
            "data": self.data,
            "memory": [record.title for record in self.memory],
            "nodes": [node.to_dict() for node in self.nodes],
            "changed_files": self.changed_files,
            "artifacts": self.artifacts,
            "needs_escalation": self.needs_escalation,
            "needs_human": self.needs_human,
            "milestone_reached": self.milestone_reached,
            "report": self.report.to_dict() if self.report else None,
        }


class Agent(ABC):
    """Base class for every specialist."""

    #: Node kind this agent handles.
    kind: str = "agent"
    #: What sort of thinking this is, for routing.
    task_class: TaskClass = TaskClass.IMPLEMENTATION
    #: Baseline difficulty and stakes, refined per-node by ``profile()``.
    difficulty: float = 0.5
    stakes: float = 0.5
    #: Share of the context budget this agent should use. Planning needs
    #: breadth; a targeted fix needs depth on few files.
    context_fraction: float = 1.0
    #: Whether the orchestrator should commit the workspace after a success.
    commits: bool = False

    @abstractmethod
    def run(self, ctx: AgentContext) -> AgentResult: ...

    # -- routing ---------------------------------------------------------

    def profile(self, ctx: AgentContext, **overrides: Any) -> TaskProfile:
        """Describe the task so the router can price it.

        Difficulty rises with the node's attempt count: a node on its third try
        is empirically harder than the planner thought, whatever it thought.
        """
        attempt = max(0, ctx.node.attempts - 1)
        difficulty = min(1.0, self.difficulty + 0.12 * attempt)
        base = {
            "task_class": self.task_class,
            "difficulty": difficulty,
            "stakes": self.stakes,
            "attempt": attempt,
            "label": f"{self.kind}:{ctx.node.id[-8:]}",
        }
        base.update(overrides)
        return TaskProfile(**base)  # type: ignore[arg-type]

    # -- context ---------------------------------------------------------

    def context_budget(self, ctx: AgentContext) -> int:
        return int(ctx.config.memory.max_context_tokens * self.context_fraction)

    def builder(self, ctx: AgentContext) -> ContextBuilder:
        """A context builder pre-loaded with the sections nearly every agent wants.

        Assembling the common sections here rather than in each agent is what
        keeps prompts consistent across specialists -- the model sees the same
        layout whether it is planning or debugging, which measurably reduces
        format-following errors on smaller models.
        """
        builder = ContextBuilder(self.context_budget(ctx))
        node = ctx.node

        builder.add("Project goal", ctx.goal, priority=P_GOAL, stable=True, max_tokens=800)

        digest = ctx.project.get("digest", "")
        if digest:
            builder.add(
                "Architecture digest", digest, priority=P_ARCHITECTURE, stable=True, max_tokens=2500
            )

        if node.acceptance:
            builder.add(
                "Acceptance criteria for this task",
                "\n".join(f"- {item}" for item in node.acceptance),
                priority=P_ACCEPTANCE,
                max_tokens=700,
            )

        owned_paths = [str(path) for path in node.spec.get("paths", []) if path]
        if owned_paths:
            builder.add(
                "Files owned by this task",
                "Edit only these paths; inspect other files read-only for interfaces. "
                "A later dependency node owns any wiring or test file not listed here.\n"
                + "\n".join(f"- {path}" for path in owned_paths),
                priority=P_ACCEPTANCE,
                max_tokens=500,
            )

        query = " ".join(
            [node.title, str(node.spec.get("objective", "")), " ".join(node.spec.get("tags", []))]
        )
        records = ctx.memory.search(query, limit=ctx.config.memory.retrieval_limit, paths=node.spec.get("paths", []))
        interfaces = [r for r in records if r.kind == MemoryKind.INTERFACE]
        conventions = ctx.memory.by_kind(MemoryKind.CONVENTION, limit=8)
        other = [r for r in records if r.kind not in (MemoryKind.INTERFACE, MemoryKind.CONVENTION)]

        builder.add_records("Interfaces you must satisfy", interfaces, priority=P_INTERFACES, max_tokens=2000, stable=True)
        builder.add_records("Project conventions", conventions, priority=P_CONVENTIONS, max_tokens=1200, stable=True)
        builder.add_records("Relevant project memory", other, priority=P_MEMORY, max_tokens=2500)

        relevant_lessons = ctx.lessons.search(query or ctx.goal, limit=4, tags=[self.kind])
        if relevant_lessons:
            builder.add(
                "Lessons from previous work",
                "\n\n".join(lesson.render() for lesson in relevant_lessons),
                priority=P_LESSONS,
                max_tokens=900,
                stable=True,
            )

        return builder

    def system_prompt(self, ctx: AgentContext) -> str:
        """Role instructions. Subclasses override; this is the shared preamble."""
        return SHARED_PREAMBLE

    # -- model access ----------------------------------------------------

    def ask(
        self,
        ctx: AgentContext,
        builder: ContextBuilder,
        task: str,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Send an assembled prompt and return the parsed result."""
        messages: list[Message] = builder.build(system_prompt=self.system_prompt(ctx), task=task)
        report = builder.report()
        ctx.logger().debug(
            "context assembled",
            used_tokens=report["used"],
            budget=report["budget"],
            dropped=report["dropped"],
        )
        profile = kwargs.pop("profile", None) or self.profile(ctx)
        completion = ctx.models.complete(
            messages,
            profile,
            schema=schema,
            node_id=ctx.node.id,
            **kwargs,
        )
        return completion.parsed if schema else completion.text

    # -- shared helpers --------------------------------------------------

    @staticmethod
    def record_facts(ctx: AgentContext, records: list[MemoryRecord]) -> list[MemoryRecord]:
        return records

    def gate_names(self, ctx: AgentContext) -> list[str]:
        """Gates to run for this node: the node's own, or the project default."""
        return ctx.node.gates or list(ctx.config.validation.gates)

    def run_gates(
        self,
        ctx: AgentContext,
        *,
        changed_files: list[str] | None = None,
        fail_fast: bool | None = None,
        gate_names: list[str] | None = None,
    ) -> ValidationReport:
        gate_ctx = ctx.gates.build_context(
            root=ctx.root,
            sandbox=ctx.sandbox,
            toolchain=ctx.toolchain,
            node_id=ctx.node.id,
            changed_files=changed_files or [],
            settings=ctx.node.spec.get("gate_settings", {}),
            memory=ctx.memory,
        )
        return ctx.gates.run(
            gate_names if gate_names is not None else self.gate_names(ctx),
            gate_ctx,
            fail_fast=fail_fast,
        )


SHARED_PREAMBLE = """\
You are a component of Forge, an autonomous software engineering platform. You \
are working without human supervision on a real codebase, and your output is \
applied directly.

Operating rules:
- Prefer the smallest change that fully satisfies the task. Unrequested \
refactoring creates review burden and risk.
- Match the surrounding code: its naming, structure, error handling and comment \
density. Consistency matters more than your personal preference.
- Never invent APIs. If you need something that does not exist, create it \
explicitly as part of your change.
- Never write credentials, API keys or tokens into files. Read them from the \
environment.
- If the task cannot be completed correctly with the information given, say so \
plainly in your output rather than guessing. A clear "blocked, and here is \
why" is far more useful than a plausible wrong answer.
- Your output is parsed by a program. Follow the requested structure exactly.
"""
