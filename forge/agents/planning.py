"""Planning and architecture agents.

These two are where the platform earns or loses most of its quality, because
every downstream node inherits their output. They are also the only places where
Forge routinely spends frontier tokens on purpose: a bad plan costs a hundred
times more than the tokens saved by planning cheaply, and unlike implementation
there is no gate that catches a bad plan automatically.

The planner also owns *assumptions*. The brief says the human should provide
only a project description, which means Forge must decide a hundred small things
nobody told it. Rather than asking, it decides, writes down what it decided and
why, marks its confidence, and revisits when evidence arrives. That record is
the substitute for a requirements conversation, and it is what makes the run
auditable afterwards.
"""

from __future__ import annotations

import json
from typing import Any

from ..errors import ForgeError
from ..memory.context import P_HISTORY, P_TREE, file_tree, reference_images
from ..memory.records import MemoryKind, MemoryRecord
from ..memory.store import assumption, convention, decision, fact, interface, requirement
from ..models.provider import encode_image
from ..models.structured import array, boolean, enum, integer, object_schema, string
from ..models.types import TaskClass
from ..obs.log import get_logger
from .base import Agent, AgentContext, AgentResult, ProposedNode
from .registry import register

log = get_logger("agents.planning")


def _attach_reference_images(builder: Any, ctx: AgentContext) -> None:
    references = reference_images(ctx.root)
    for path in references:
        builder.add_image(encode_image(path, label=f"reference:{path.name}"))
    if references:
        labels: list[str] = []
        for path in references:
            try:
                labels.append(path.relative_to(ctx.root).as_posix())
            except ValueError:
                labels.append(f".forge/references/{path.name}")
        builder.add(
            "Human-supplied reference images",
            "\n".join(f"- {label}" for label in labels),
            priority=P_TREE,
            max_tokens=250,
        )


def _draft_source_evidence(ctx: AgentContext, draft: dict[str, Any]) -> dict[str, str]:
    """Read a bounded set of files named by a planning/architecture draft.

    Subscription CLI providers intentionally run without project tools, so a
    frontier coach cannot inspect the checkout on demand.  Supplying the files
    the local draft itself says are relevant gives the coach real evidence
    without exposing the whole repository or enabling autonomous tool use.
    """
    candidates: list[str] = []
    for task in draft.get("tasks", []):
        for raw in task.get("paths", []):
            path = str(raw).strip()
            if path and path not in candidates:
                candidates.append(path)
    for module in draft.get("modules", []):
        path = str(module.get("path", "")).strip()
        if path and path not in candidates:
            candidates.append(path)

    evidence: dict[str, str] = {}
    remaining_chars = 48_000
    for relative in candidates[:16]:
        path = (ctx.root / relative).resolve()
        try:
            path.relative_to(ctx.root.resolve())
        except ValueError:
            continue
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".mp4"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if remaining_chars <= 0:
            break
        content = content[:remaining_chars]
        evidence[relative] = content
        remaining_chars -= len(content)
    return evidence


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

_NODE_KINDS = [
    "architect", "scaffold", "implement", "test_author", "review",
    "validate", "browser_qa", "visual_review", "perf", "security",
    "document", "deploy",
]

PLAN_SCHEMA = object_schema(
    {
        "interpretation": string(
            "One paragraph restating what is being built, in concrete terms. "
            "Resolve vagueness in the goal here."
        ),
        "requirements": array(
            object_schema(
                {
                    "title": string("Short name for the requirement"),
                    "detail": string("What must be true for this to be satisfied"),
                    "must_have": boolean("False if this is desirable but not essential"),
                },
            ),
            "The concrete requirements implied by the goal, including ones the "
            "human did not state but clearly expects",
            minItems=3,
        ),
        "assumptions": array(
            object_schema(
                {
                    "title": string("The assumption, stated as a claim"),
                    "rationale": string("Why this is the reasonable default"),
                    "confidence": string("high, medium or low"),
                    "revisit_when": string("What evidence would change this"),
                },
            ),
            "Decisions you are making on the human's behalf because asking "
            "would stall the work",
        ),
        "milestones": array(
            object_schema(
                {
                    "name": string("Short milestone identifier, kebab-case"),
                    "objective": string("What is demonstrably true when this milestone is done"),
                    "acceptance": array(string(), "Checkable criteria", minItems=1),
                },
            ),
            "Ordered milestones. Each must end in something runnable and "
            "verifiable, not an intermediate state.",
            minItems=1,
        ),
        "tasks": array(
            object_schema(
                {
                    "title": string("Imperative, specific: 'Implement the collision solver'"),
                    "kind": enum(_NODE_KINDS, "Which specialist should do this"),
                    "milestone": string("Milestone name this belongs to"),
                    "objective": string("What this task must achieve, precisely"),
                    "acceptance": array(string(), "Criteria a reviewer can check", minItems=1),
                    "paths": array(string(), "Files this task is expected to touch"),
                    "deps": array(integer(), "Indices of tasks in this list that must finish first"),
                    "priority": integer("Lower runs first; use 10-200", minimum=1, maximum=1000),
                },
            ),
            "The task graph for the FIRST milestone only. Later milestones are "
            "planned when their turn comes, with what has been learned by then.",
            minItems=1,
        ),
        "risks": array(string(), "What is most likely to go wrong, most likely first"),
    },
    required=["interpretation", "requirements", "assumptions", "milestones", "tasks"],
)


ARCHITECTURE_SCHEMA = object_schema(
    {
        "overview": string("Two or three paragraphs describing the system's shape"),
        "stack": array(
            object_schema(
                {
                    "choice": string("The technology chosen"),
                    "role": string("What it is used for"),
                    "rationale": string("Why this over the alternatives"),
                    "alternatives": array(string(), "What was considered and rejected"),
                },
            ),
            "Technology choices with their reasoning",
        ),
        "modules": array(
            object_schema(
                {
                    "name": string("Module name"),
                    "path": string("Where it lives in the repository"),
                    "responsibility": string("One sentence; if it needs two, split the module"),
                    "interface": string("The public surface: functions, types, events"),
                    "depends_on": array(string(), "Other module names"),
                },
            ),
            "The module decomposition",
            minItems=1,
        ),
        "production_workflows": array(
            object_schema(
                {
                    "discipline": string(
                        "A user-visible or operational discipline such as graphics, "
                        "audio, content, data, deployment or observability"
                    ),
                    "target": string("The concrete quality bar or outcome"),
                    "tools": array(
                        string(),
                        "Specific authoring, inspection, conversion and validation tools",
                        minItems=1,
                    ),
                    "method": string(
                        "A repeatable step-by-step production method agents can execute"
                    ),
                    "artifacts": array(
                        string(),
                        "Intermediate and final files this workflow produces",
                        minItems=1,
                    ),
                    "validation": array(
                        string(),
                        "Objective checks and fresh-context review methods",
                        minItems=1,
                    ),
                    "reference_use": string(
                        "How supplied references define the bar without being copied blindly"
                    ),
                },
            ),
            "Executable production workflows for disciplines that materially affect "
            "the result. Runtime libraries alone are not a production workflow.",
            minItems=1,
        ),
        "conventions": array(
            object_schema(
                {"title": string("Convention name"), "detail": string("The rule, stated so it can be followed")},
            ),
            "How this codebase is written: naming, structure, error handling, testing",
        ),
        "data_flow": string("How data moves through the system, in prose"),
        "risks": array(string(), "Architectural risks and how the design mitigates them"),
    },
    required=["overview", "stack", "modules", "production_workflows", "conventions"],
)


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------


@register
class PlannerAgent(Agent):
    """Turns a goal, or a milestone, into a dependency graph of tasks.

    Plans one milestone at a time. Planning the whole project up front produces
    a confident, detailed and wrong graph for everything past the first
    milestone, because the decisions that constrain milestone three have not
    been made yet. Replanning at each boundary costs one extra planning call per
    milestone and produces a plan informed by the code that now exists.
    """

    kind = "plan"
    task_class = TaskClass.PLANNING
    difficulty = 0.75
    # A bad plan is not caught by any gate, so it is worth a strong model.
    stakes = 0.95
    context_fraction = 1.0

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You are the planner. You decompose a goal into a dependency graph of tasks that \
other agents will execute without further human input.

What a good plan looks like here:
- Every task is completable by one agent in one focused sitting. If a task \
would take a person more than a few hours, split it.
- Every task has acceptance criteria that a program or a reviewer can check. \
"Make it good" is not a criterion; "the player can fire and the projectile \
collides with walls" is.
- Dependencies are real. Two tasks that touch different files with no shared \
contract should not depend on each other, because parallel execution is free \
and serialisation is not.
- The first milestone must produce something that runs. A milestone that ends \
with untested scaffolding gives the validation layer nothing to check, and \
problems then surface much later and much more expensively.
- Include validation, review and QA tasks explicitly. They are work, they take \
time, and leaving them implicit means they get skipped.
- When the result has subjective qualities such as graphics, sound, motion, \
writing or interaction feel, schedule reference analysis and a small production \
spike before bulk implementation. The architecture must choose authoring tools, \
asset formats, repeatable methods and comparison/review techniques; do not leave \
those choices for implementation agents to improvise.

State assumptions rather than asking questions. You will not get an answer.
"""
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        milestone = ctx.spec.get("milestone")
        replanning = bool(ctx.spec.get("replan"))

        builder = self.builder(ctx)
        _attach_reference_images(builder, ctx)
        tree = file_tree(ctx.root, limit=250)
        if tree:
            builder.add("Existing files", tree, priority=P_TREE, max_tokens=1500)

        if replanning:
            history = self._recent_history(ctx)
            builder.add("What has happened so far", history, priority=P_HISTORY, max_tokens=2000)
            open_findings = ctx.memory.open_findings()
            builder.add_records("Unresolved findings", open_findings[:10], priority=P_HISTORY, max_tokens=1500)

        toolchain = ctx.toolchain.get("languages") or []
        if toolchain:
            builder.add(
                "Detected toolchain",
                f"Languages: {', '.join(toolchain)}\nCommands: {ctx.toolchain.get('commands', {})}",
                priority=P_TREE,
                max_tokens=400,
            )

        task = self._task_text(ctx, milestone, replanning)
        plan: dict[str, Any] = self.ask(
            ctx,
            builder,
            task,
            schema=PLAN_SCHEMA,
            profile=self.profile(
                ctx,
                attempt=0,
                min_tier="local_deep",
                max_tier="mid",
                label=f"local-plan:{ctx.node.id[-8:]}",
            ),
        )
        plan, coach_advice = _coach_and_revise(
            self, ctx, task, plan, PLAN_SCHEMA, label="plan"
        )

        records = self._records_from_plan(plan, ctx)
        if coach_advice:
            records.append(
                fact(
                    f"Coach advice for project plan {ctx.node.id}",
                    coach_advice,
                    source=f"coach:{ctx.node.id}",
                    tags=["coach", "planning"],
                )
            )
        nodes = self._nodes_from_plan(plan, ctx, milestone)

        if not nodes:
            return AgentResult.failure("the plan contained no executable tasks")

        milestones = plan.get("milestones", [])
        return AgentResult(
            success=True,
            summary=f"planned {len(nodes)} task(s) across {len(milestones)} milestone(s)",
            data={
                "interpretation": plan.get("interpretation", ""),
                "milestones": milestones,
                "risks": plan.get("risks", []),
            },
            memory=records,
            nodes=nodes,
        )

    def _task_text(self, ctx: AgentContext, milestone: str | None, replanning: bool) -> str:
        if replanning and milestone:
            return (
                f"Plan the tasks for milestone '{milestone}'.\n\n"
                "The previous milestones are complete. Use what the codebase now "
                "looks like and what went wrong previously to plan this one. "
                "Re-state the full milestone list with your current understanding, "
                "but emit tasks only for this milestone."
            )
        return (
            "Produce the initial plan for this project.\n\n"
            "Decide the milestones, then emit the task graph for the first "
            "milestone only. Be concrete about what exists at the end of each "
            "milestone."
        )

    def _recent_history(self, ctx: AgentContext) -> str:
        """A compact narrative of the run so far.

        Built from the node table rather than the event log: the planner needs
        to know what was done and what failed, not the sequence of lease
        renewals that got it there.
        """
        lines: list[str] = []
        for node in ctx.graph.all_nodes():
            if node.id == ctx.node.id:
                continue
            marker = {"succeeded": "done", "failed": "FAILED", "blocked": "BLOCKED"}.get(node.status)
            if marker:
                detail = ""
                if node.status != "succeeded" and node.result:
                    detail = f" -- {str(node.result.get('error', ''))[:160]}"
                lines.append(f"[{marker}] {node.title}{detail}")
        commits = ctx.repo.log(limit=15)
        if commits:
            lines.append("")
            lines.append("Recent commits:")
            lines += [f"- {c.subject}" for c in commits]
        return "\n".join(lines[-80:])

    def _records_from_plan(self, plan: dict[str, Any], ctx: AgentContext) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        confidence_map = {"high": 0.85, "medium": 0.6, "low": 0.35}

        if plan.get("interpretation"):
            records.append(
                requirement(
                    "Project interpretation",
                    plan["interpretation"],
                    source="planner",
                )
            )
        for item in plan.get("requirements", []):
            records.append(
                requirement(
                    item["title"],
                    item.get("detail", "")
                    + ("" if item.get("must_have", True) else "\n\n(Desirable, not essential.)"),
                    source="planner",
                    tags=["requirement", "must" if item.get("must_have", True) else "should"],
                )
            )
        for item in plan.get("assumptions", []):
            body = item.get("rationale", "")
            if item.get("revisit_when"):
                body += f"\n\nRevisit when: {item['revisit_when']}"
            records.append(
                assumption(
                    item["title"],
                    body,
                    confidence=confidence_map.get(str(item.get("confidence", "medium")).lower(), 0.6),
                    source="planner",
                    data={"revisit_when": item.get("revisit_when", "")},
                )
            )
        if plan.get("milestones"):
            records.append(
                MemoryRecord(
                    kind=MemoryKind.FACT,
                    title="Milestone plan",
                    body="\n".join(
                        f"{i + 1}. {m['name']}: {m.get('objective', '')}"
                        for i, m in enumerate(plan["milestones"])
                    ),
                    source="planner",
                    confidence=0.8,
                    tags=["milestones"],
                )
            )
        return records

    def _nodes_from_plan(
        self, plan: dict[str, Any], ctx: AgentContext, milestone: str | None
    ) -> list[ProposedNode]:
        tasks = plan.get("tasks", [])
        nodes: list[ProposedNode] = []
        for index, task in enumerate(tasks):
            nodes.append(
                ProposedNode(
                    kind=task.get("kind", "implement"),
                    title=task["title"],
                    spec={
                        "objective": task.get("objective", task["title"]),
                        "acceptance": task.get("acceptance", []),
                        "paths": task.get("paths", []),
                        "gates": ctx.config.validation.gates,
                    },
                    # Keep only edges that point at a real earlier task. Forward
                    # and self references are planner slips, not intent.
                    deps=[
                        d
                        for d in task.get("deps", [])
                        if isinstance(d, int) and 0 <= d < len(tasks) and d != index
                    ],
                    priority=int(task.get("priority", 100)),
                    milestone=task.get("milestone") or milestone or _first_milestone(plan),
                )
            )

        # Close every milestone with a retrospective. Self-improvement is not
        # optional bookkeeping; it is the mechanism by which the next milestone
        # goes better, so it is scheduled as work rather than hoped for.
        milestone_name = milestone or _first_milestone(plan)
        if milestone_name and ctx.config.improvement.retrospective_after_milestone:
            nodes.append(
                ProposedNode(
                    kind="retrospect",
                    title=f"Retrospective for milestone '{milestone_name}'",
                    spec={
                        "objective": "Analyse how this milestone went and record lessons",
                        "acceptance": ["Lessons recorded", "Routing and workflow observations captured"],
                        "milestone_name": milestone_name,
                    },
                    deps=list(range(len(tasks))),
                    priority=900,
                    milestone=milestone_name,
                )
            )
        return nodes


def _first_milestone(plan: dict[str, Any]) -> str:
    milestones = plan.get("milestones") or []
    return str(milestones[0]["name"]) if milestones else "milestone-1"


# --------------------------------------------------------------------------
# Architect
# --------------------------------------------------------------------------


@register
class ArchitectAgent(Agent):
    """Establishes module boundaries, interfaces and conventions.

    Runs before implementation and produces records rather than code. The
    interfaces it writes are what let later implementation nodes work on
    different modules concurrently without reading each other's source -- which
    is simultaneously the parallelism story and the biggest context-size saving
    in the platform.
    """

    kind = "architect"
    task_class = TaskClass.ARCHITECTURE
    difficulty = 0.8
    stakes = 0.95
    commits = True

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You are the architect. You decide the shape of the system before code is \
written, and you write down the contracts that let several agents build \
different parts of it in parallel without reading each other's code.

Judgement to apply:
- Choose boring, well-supported technology unless the goal specifically \
demands otherwise. This system will be maintained by agents that know common \
tools far better than novel ones.
- Module responsibilities must be stated in one sentence. If yours needs two, \
the module is doing two things.
- Interfaces are the deliverable. Write them precisely enough that someone can \
implement against them without seeing the implementation on the other side.
- State what you rejected and why. Without that, the next agent proposes it \
again.
- Architecture includes production, not only runtime modules. For every \
important user-visible or operational discipline, specify tools agents can \
actually run, the artifact pipeline, a repeatable method, and how outputs will \
be compared with supplied references or another concrete quality bar. A choice \
such as Canvas, Web Audio or a UI framework is only the start of that answer.
"""
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        builder = self.builder(ctx)
        _attach_reference_images(builder, ctx)
        tree = file_tree(ctx.root, limit=200)
        if tree:
            builder.add("Existing files", tree, priority=P_TREE, max_tokens=1200)

        objective = ctx.spec.get("objective") or "Design the architecture for this project."
        task = (
            f"{objective}\n\n"
            "Produce the architecture. Cover the technology stack, the module "
            "decomposition with explicit interfaces, the conventions this "
            "codebase will follow, how data flows through the system, and "
            "executable production workflows. For graphics, sound, motion, "
            "content or other subjective deliverables, name the authoring and "
            "inspection tools, intermediate artifacts, iteration method, and "
            "reference-based validation strategy."
        )
        result: dict[str, Any] = self.ask(
            ctx,
            builder,
            task,
            schema=ARCHITECTURE_SCHEMA,
            profile=self.profile(
                ctx,
                attempt=0,
                min_tier="local_deep",
                max_tier="mid",
                label=f"local-architecture:{ctx.node.id[-8:]}",
            ),
        )
        result, coach_advice = _coach_and_revise(
            self, ctx, task, result, ARCHITECTURE_SCHEMA, label="architecture"
        )

        records: list[MemoryRecord] = [
            decision(
                "System overview",
                result["overview"],
                source="architect",
            )
        ]
        if coach_advice:
            records.append(
                fact(
                    f"Coach advice for architecture {ctx.node.id}",
                    coach_advice,
                    source=f"coach:{ctx.node.id}",
                    tags=["coach", "architecture"],
                )
            )
        for item in result.get("stack", []):
            records.append(
                decision(
                    f"Use {item['choice']} for {item['role']}",
                    item.get("rationale", ""),
                    alternatives=item.get("alternatives", []),
                    source="architect",
                    tags=["stack"],
                )
            )
        for module in result.get("modules", []):
            body = f"{module.get('responsibility', '')}\n\nInterface:\n{module.get('interface', '')}"
            depends = module.get("depends_on") or []
            if depends:
                body += f"\n\nDepends on: {', '.join(depends)}"
            records.append(
                interface(
                    f"Module: {module['name']}",
                    body,
                    paths=[module["path"]] if module.get("path") else [],
                    tags=["module"],
                )
            )
        for item in result.get("conventions", []):
            records.append(convention(item["title"], item.get("detail", ""), source="architect"))
        if result.get("data_flow"):
            records.append(
                decision("Data flow", result["data_flow"], source="architect", tags=["dataflow"])
            )
        for risk in result.get("risks", []):
            records.append(
                MemoryRecord(
                    kind=MemoryKind.FACT,
                    title=f"Architectural risk: {risk[:70]}",
                    body=risk,
                    source="architect",
                    confidence=0.7,
                    tags=["risk"],
                )
            )

        # The architecture is also written into the repository as documentation,
        # because the humans who eventually read this project should not have to
        # query Forge's database to learn how it is built.
        doc = _architecture_markdown(result)
        docs_dir = ctx.root / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "architecture.md").write_text(doc, encoding="utf-8")

        return AgentResult(
            success=True,
            summary=f"architecture defined: {len(result.get('modules', []))} module(s)",
            data=result,
            memory=records,
            changed_files=["docs/architecture.md"],
            commit_message="docs: record system architecture",
        )


def _coach_and_revise(
    agent: Agent,
    ctx: AgentContext,
    task: str,
    draft: dict[str, Any],
    schema: dict[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    """Local draft, bounded cloud critique, then local-deep revision."""
    cloud = next(
        (
            name
            for name in ctx.config.models.ladder
            if ctx.config.models.models[name].hosted == "cloud"
        ),
        "",
    )
    if not cloud:
        return draft, ""
    try:
        coach_builder = agent.builder(ctx)
        _attach_reference_images(coach_builder, ctx)
        tree = file_tree(ctx.root, limit=250)
        if tree:
            coach_builder.add("Actual project file tree", tree, priority=P_TREE, max_tokens=1500)
        source_evidence = _draft_source_evidence(ctx, draft)
        if source_evidence:
            coach_builder.add_files(
                "Current source evidence selected by the local draft",
                source_evidence,
                priority=P_TREE + 1,
                max_tokens=8000,
            )
        coach_builder.add(
            f"Local {label} draft",
            json.dumps(draft, indent=2),
            priority=P_HISTORY,
            max_tokens=6000,
        )
        advice = str(
            agent.ask(
                ctx,
                coach_builder,
                f"Critique this {label} for the local model that will revise it. "
                "Identify concrete omissions, unsafe assumptions, bad boundaries, "
                "and validation gaps. Give ordered revision instructions. Use only "
                "the supplied tree and source evidence: this coaching call has no "
                "filesystem tools, so never claim to have inspected its working "
                "directory. Do not replace the draft or author a new one.",
                profile=agent.profile(
                    ctx,
                    attempt=0,
                    min_tier=cloud,
                    label=f"coach-{label}:{ctx.node.id[-8:]}",
                ),
                max_output_tokens=1500,
            )
            or ""
        ).strip()
    except ForgeError as exc:
        ctx.logger().warn(f"{label} coach unavailable; keeping local draft", error=str(exc))
        return draft, ""
    if not advice:
        return draft, ""

    try:
        revise_builder = agent.builder(ctx)
        _attach_reference_images(revise_builder, ctx)
        revise_builder.add(
            f"Your original {label} draft",
            json.dumps(draft, indent=2),
            priority=P_HISTORY,
            max_tokens=6000,
        )
        revise_builder.add(
            "Senior critique to apply",
            advice,
            priority=P_HISTORY - 1,
            max_tokens=1800,
        )
        revised = agent.ask(
            ctx,
            revise_builder,
            f"Revise the {label} draft using the critique. Preserve sound decisions, "
            "apply justified corrections, and return the complete revised result.",
            schema=schema,
            profile=agent.profile(
                ctx,
                attempt=0,
                min_tier="local_deep",
                max_tier="mid",
                label=f"revise-{label}:{ctx.node.id[-8:]}",
            ),
        )
        return revised, advice
    except ForgeError as exc:
        ctx.logger().warn(f"local {label} revision failed; keeping draft", error=str(exc))
        return draft, advice


def _architecture_markdown(result: dict[str, Any]) -> str:
    lines = ["# Architecture", "", "_Generated by Forge._", "", "## Overview", "", result.get("overview", ""), ""]
    if result.get("stack"):
        lines += ["## Technology choices", ""]
        for item in result["stack"]:
            lines += [f"### {item['choice']} -- {item.get('role', '')}", "", item.get("rationale", ""), ""]
            if item.get("alternatives"):
                lines += ["Rejected: " + ", ".join(item["alternatives"]), ""]
    if result.get("modules"):
        lines += ["## Modules", ""]
        for module in result["modules"]:
            lines += [
                f"### `{module['name']}` -- `{module.get('path', '')}`",
                "",
                module.get("responsibility", ""),
                "",
                "```",
                module.get("interface", ""),
                "```",
                "",
            ]
    if result.get("production_workflows"):
        lines += ["## Production workflows", ""]
        for workflow in result["production_workflows"]:
            lines += [
                f"### {workflow['discipline']} -- {workflow.get('target', '')}",
                "",
                "Tools: " + ", ".join(workflow.get("tools", [])),
                "",
                workflow.get("method", ""),
                "",
                "Artifacts:",
                *[f"- {item}" for item in workflow.get("artifacts", [])],
                "",
                "Validation:",
                *[f"- {item}" for item in workflow.get("validation", [])],
                "",
                "Reference use: " + workflow.get("reference_use", ""),
                "",
            ]
    if result.get("data_flow"):
        lines += ["## Data flow", "", result["data_flow"], ""]
    if result.get("conventions"):
        lines += ["## Conventions", ""]
        lines += [f"- **{c['title']}**: {c.get('detail', '')}" for c in result["conventions"]]
        lines += [""]
    if result.get("risks"):
        lines += ["## Risks", ""] + [f"- {r}" for r in result["risks"]]
    return "\n".join(lines)
