"""The self-improvement agents.

``RetrospectAgent`` runs at every milestone boundary. It reads the computed
metrics, the routing statistics and the detected promotion candidates, and turns
them into lessons that persist beyond this project plus concrete changes to
propose.

``ImproveAgent`` acts on one such proposal: adding a lint rule, tightening a
convention, adding a gate, adjusting a prompt.

The important restraint is that neither of them may silently change how the
platform behaves. A retrospective writes lessons and *proposes* configuration
changes; applying a change that alters routing or validation is a node like any
other, recorded in the ledger, reviewable and revertible. A system that rewrites
its own operating rules without leaving a trail is not one anybody can run for
five years.
"""

from __future__ import annotations

from typing import Any

from ..improve.metrics import compute_metrics
from ..improve.promotion import detect_promotions, gate_promotions, routing_promotions
from ..kernel.events import EventType
from ..memory.context import P_FAILURE, P_HISTORY
from ..memory.lessons import Lesson
from ..memory.records import MemoryKind, MemoryRecord
from ..memory.store import convention
from ..models.structured import array, enum, object_schema, string
from ..models.types import TaskClass
from ..obs.log import get_logger
from ..workspace.patch import EDIT_PLAN_SCHEMA, EditPlan, apply_edits
from .base import Agent, AgentContext, AgentResult, ProposedNode
from .registry import register

log = get_logger("agents.improving")


RETROSPECTIVE_SCHEMA = object_schema(
    {
        "assessment": string(
            "Two or three sentences: did this milestone go well, and what actually "
            "determined that? Be specific about causes, not effects."
        ),
        "lessons": array(
            object_schema(
                {
                    "title": string("A transferable claim, stated so it can be acted on"),
                    "detail": string("What to do differently, and why"),
                    "tags": array(string(), "Which agents or subsystems this applies to"),
                    "generalises": enum(
                        ["this_project", "any_project"],
                        "Whether this would help on an unrelated project too",
                    ),
                },
                required=["title", "detail", "generalises"],
            ),
            "What was learned. Fewer, sharper lessons beat many vague ones. "
            "An empty list is acceptable when nothing notable happened.",
        ),
        "workflow_changes": array(
            object_schema(
                {
                    "change": string("The concrete change to how work is done"),
                    "rationale": string("Which measurement supports this"),
                    "impact": enum(["low", "medium", "high"], "Expected benefit"),
                    "kind": enum(
                        ["planning", "context", "routing", "validation", "prompt", "tooling"],
                        "What part of the workflow this touches",
                    ),
                },
                required=["change", "rationale", "kind"],
            ),
            "Changes to how the platform works, each justified by a number above",
        ),
        "assumptions_to_revisit": array(
            string(), "Titles of recorded assumptions that the evidence now contradicts"
        ),
        "next_milestone_advice": string(
            "What the planner should do differently for the next milestone"
        ),
    },
    required=["assessment", "lessons", "workflow_changes"],
)


@register
class RetrospectAgent(Agent):
    """Analyses a completed milestone and records what should change."""

    kind = "retrospect"
    task_class = TaskClass.RETROSPECTIVE
    difficulty = 0.65
    # A wrong lesson persists across projects and quietly degrades every future
    # run, which makes this higher-stakes than its cost suggests.
    stakes = 0.8
    commits = False

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You are analysing how an autonomous build went, using measured data rather than \
impressions. The numbers you are given are facts from an event log; treat them \
as such and reason from them.

What a useful retrospective produces:
- Causes, not restatements. "Three debug nodes failed" is the data. "Debug \
nodes were given the failing output but not the file that produced it" is a \
finding.
- Lessons that would change a future decision. If a lesson would not alter what \
some agent does next time, do not record it.
- Changes justified by a specific number. "Reduce context" is not a change; \
"the implementer's context averaged 18k tokens of which the file tree was 4k \
and was never referenced" is.

Be honest about a milestone that went well. Manufacturing findings to seem \
thorough pollutes the lesson library, and every future run pays for it.

Never invent a numerical breakdown that is not present in the measured data. \
For example, task-class call totals do not imply calls or success rates by \
model. If two supplied measurements conflict, identify the telemetry problem \
instead of choosing the more convenient number.

Operator requirements outrank statistical optimisations. In particular, a \
routing observation may improve retries, coaching or scaffolding, but must not \
bypass a configured local-first ladder or cloud-usage limit.
"""
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        milestone = ctx.spec.get("milestone_name") or ctx.node.milestone
        ledger = ctx.models.ledger

        # Everything below is computed, not recalled. The model interprets; it
        # does not get to decide what happened.
        metrics = compute_metrics(ledger, ctx.graph, milestone=milestone)
        promotions = detect_promotions(
            ledger, ctx.memory, threshold=ctx.config.improvement.promote_findings_after
        )
        routing_notes = routing_promotions(
            ledger, min_samples=ctx.config.improvement.min_samples_for_routing_update
        )
        gate_notes = gate_promotions(ledger)
        policy_notes = ctx.models.policy.recommendations()

        builder = self.builder(ctx)
        builder.add("Measured outcomes", metrics.render(), priority=P_FAILURE, max_tokens=3000)
        if promotions:
            builder.add(
                "Repeated problems that tooling could prevent",
                "\n".join(
                    f"- ({c.occurrences}x, from {c.origin}) {c.examples[0] if c.examples else c.signature}"
                    for c in promotions
                ),
                priority=P_FAILURE + 1,
                max_tokens=1500,
            )
        observations = routing_notes + gate_notes + policy_notes
        if observations:
            builder.add(
                "Statistical observations",
                "\n".join(f"- {note}" for note in observations),
                priority=P_FAILURE + 2,
                max_tokens=1500,
            )
        failures = self._failure_narrative(ctx, milestone)
        if failures:
            builder.add("What failed and how it was resolved", failures, priority=P_HISTORY, max_tokens=2500)
        assumptions = ctx.memory.by_kind(MemoryKind.ASSUMPTION, limit=25)
        builder.add_records("Assumptions currently in force", assumptions, priority=P_HISTORY, max_tokens=1800)
        requirements = ctx.memory.by_kind(MemoryKind.REQUIREMENT, limit=30)
        builder.add_records(
            "Operator requirements that proposed changes must preserve",
            requirements,
            priority=P_FAILURE + 3,
            max_tokens=2400,
        )

        result: dict[str, Any] = self.ask(
            ctx,
            builder,
            f"Analyse milestone '{milestone or 'this run'}'. Use the measurements to "
            "identify what actually determined the outcome, record the lessons worth "
            "keeping, and propose changes each justified by a specific number.",
            schema=RETROSPECTIVE_SCHEMA,
        )

        records = self._record_lessons(ctx, result)
        records += self._revisit_assumptions(ctx, result)
        nodes = self._improvement_nodes(ctx, result, promotions)

        ledger.emit(
            EventType.RETROSPECTIVE_RECORDED,
            node_id=ctx.node.id,
            milestone=milestone,
            metrics=metrics.to_dict(),
            assessment=result.get("assessment", ""),
            lessons=len(result.get("lessons", [])),
            changes=len(result.get("workflow_changes", [])),
        )

        return AgentResult(
            success=True,
            summary=(
                f"retrospective for '{milestone}': {len(result.get('lessons', []))} lesson(s), "
                f"{len(nodes)} improvement(s) proposed"
            ),
            data={
                "metrics": metrics.to_dict(),
                "assessment": result.get("assessment", ""),
                "advice": result.get("next_milestone_advice", ""),
                "promotions": [c.to_dict() for c in promotions],
            },
            memory=records,
            nodes=nodes,
            milestone_reached=milestone or "",
        )

    def _failure_narrative(self, ctx: AgentContext, milestone: str | None) -> str:
        lines: list[str] = []
        for node in ctx.graph.all_nodes():
            if milestone and node.milestone != milestone:
                continue
            if node.attempts <= 1 and node.status == "succeeded":
                continue
            detail = ""
            if node.result and node.result.get("error"):
                detail = f" -- {str(node.result['error'])[:200]}"
            lines.append(
                f"[{node.status}, {node.attempts} attempt(s), tier {node.tier}] {node.title}{detail}"
            )
        return "\n".join(lines[:40])

    def _record_lessons(self, ctx: AgentContext, result: dict[str, Any]) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for item in result.get("lessons", []):
            body = item.get("detail", "")
            if item.get("generalises") == "any_project":
                # Only genuinely transferable lessons enter the shared library.
                # Project-specific ones stay in project memory, where they help
                # without polluting every future run's retrieval.
                ctx.lessons.add(
                    Lesson(
                        title=item["title"],
                        body=body,
                        context=f"Learned during {ctx.project.get('name', 'a project')}",
                        tags=list(item.get("tags", [])) or ["general"],
                        project=str(ctx.project.get("name", "")),
                    )
                )
                ctx.models.ledger.emit(
                    EventType.LESSON_LEARNED, node_id=ctx.node.id, title=item["title"]
                )
            records.append(
                MemoryRecord(
                    kind=MemoryKind.LESSON,
                    title=item["title"],
                    body=body,
                    source=f"retrospective:{ctx.node.id}",
                    confidence=0.75,
                    tags=list(item.get("tags", [])),
                )
            )
        return records

    def _revisit_assumptions(self, ctx: AgentContext, result: dict[str, Any]) -> list[MemoryRecord]:
        """Lower confidence in assumptions the evidence has undermined.

        Not deleted -- superseded with reduced confidence and a note about what
        contradicted them. The record of having believed something wrong is
        exactly what stops it being believed again.
        """
        records: list[MemoryRecord] = []
        titles = {t.lower() for t in result.get("assumptions_to_revisit", [])}
        if not titles:
            return records
        for record in ctx.memory.by_kind(MemoryKind.ASSUMPTION, limit=100):
            if record.title.lower() in titles:
                records.append(
                    MemoryRecord(
                        kind=MemoryKind.ASSUMPTION,
                        title=record.title,
                        body=record.body
                        + "\n\nContradicted by evidence during the retrospective; treat with caution.",
                        confidence=max(0.15, record.confidence - 0.35),
                        source=f"retrospective:{ctx.node.id}",
                        tags=[*record.tags, "revisited"],
                    )
                )
        return records

    def _improvement_nodes(
        self, ctx: AgentContext, result: dict[str, Any], promotions: list[Any]
    ) -> list[ProposedNode]:
        nodes: list[ProposedNode] = []
        for change in result.get("workflow_changes", []):
            if change.get("impact") == "low":
                continue  # recorded as a lesson; not worth a node
            nodes.append(
                ProposedNode(
                    kind="improve",
                    title=f"Improve {change['kind']}: {change['change'][:70]}",
                    spec={
                        "objective": change["change"],
                        "rationale": change.get("rationale", ""),
                        "change_kind": change["kind"],
                        "acceptance": ["The described improvement is in place and recorded"],
                    },
                    priority=850,
                    milestone=ctx.node.milestone,
                )
            )
        for candidate in promotions[:3]:
            nodes.append(
                ProposedNode(
                    kind="improve",
                    title=f"Prevent recurring problem: {candidate.signature[:60]}",
                    spec={
                        "objective": (
                            f"This problem has been found {candidate.occurrences} times. "
                            "Add deterministic tooling or a convention that prevents it, "
                            "so no model has to catch it again."
                        ),
                        "change_kind": "tooling",
                        "examples": candidate.examples,
                        "paths": candidate.paths,
                        "acceptance": ["A check or convention exists that would catch this automatically"],
                    },
                    priority=860,
                    milestone=ctx.node.milestone,
                )
            )
        return nodes


IMPROVEMENT_SCHEMA = object_schema(
    {
        "approach": enum(
            ["config_change", "convention", "gate", "test", "prompt_guidance", "no_action"],
            "How to implement this improvement",
        ),
        "rationale": string("Why this approach, and why not a heavier one"),
        "convention": string("If a convention: the rule, stated so it can be followed"),
        "edits": string("If files must change: describe what, briefly"),
        "gate_command": string("If a gate: the command that would detect this automatically"),
    },
    required=["approach", "rationale"],
)


@register
class ImproveAgent(Agent):
    """Implements one improvement proposed by a retrospective.

    Prefers the lightest intervention that works, in this order: a convention
    recorded in memory (free, applies to every future prompt), a configuration
    change, a new deterministic gate, then finally code. Reaching for code first
    is how a platform accumulates bespoke machinery for problems a sentence
    would have solved.
    """

    kind = "improve"
    task_class = TaskClass.REFACTORING
    difficulty = 0.5
    stakes = 0.6
    commits = True

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You are implementing an improvement to how this project is built.

Choose the lightest thing that actually works, in this order:
1. A recorded convention. Free, and it reaches every future prompt.
2. A configuration change to existing tooling -- a lint rule, a compiler flag.
3. A new automated check.
4. Changing code.

Prefer prevention over detection, and detection over instruction. A lint rule \
that makes the mistake impossible beats a check that catches it, which beats a \
sentence asking people not to make it.

If the improvement is not worth its cost, answer `no_action` and say why. \
Declining is a legitimate and often correct outcome.
"""
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        objective = ctx.spec.get("objective", ctx.node.title)
        rationale = ctx.spec.get("rationale", "")
        examples = ctx.spec.get("examples", [])

        builder = self.builder(ctx)
        builder.add(
            "The improvement to make",
            f"{objective}\n\nWhy: {rationale}",
            priority=P_FAILURE,
            max_tokens=1200,
        )
        if examples:
            builder.add(
                "Examples of the problem",
                "\n".join(f"- {e}" for e in examples[:8]),
                priority=P_FAILURE + 1,
                max_tokens=1200,
            )
        builder.add(
            "Existing conventions",
            "\n".join(f"- {r.title}: {r.body[:160]}" for r in ctx.memory.by_kind(MemoryKind.CONVENTION, limit=15)),
            priority=P_HISTORY,
            max_tokens=1200,
        )
        builder.add(
            "Current gate configuration",
            f"Enabled gates: {', '.join(ctx.config.validation.gates)}\n"
            f"Project commands: {ctx.toolchain.get('commands', {})}",
            priority=P_HISTORY,
            max_tokens=500,
        )

        decision: dict[str, Any] = self.ask(
            ctx, builder, "Decide how to implement this improvement.", schema=IMPROVEMENT_SCHEMA
        )
        approach = decision.get("approach", "no_action")

        if approach == "no_action":
            return AgentResult(
                success=True,
                summary=f"no action taken: {decision.get('rationale', '')[:120]}",
                data=decision,
            )

        if approach in ("convention", "prompt_guidance"):
            text = decision.get("convention") or decision.get("rationale", "")
            return AgentResult(
                success=True,
                summary=f"recorded a convention: {objective[:80]}",
                data=decision,
                memory=[
                    convention(
                        objective[:90],
                        text,
                        source=f"improve:{ctx.node.id}",
                        tags=["improvement"],
                    )
                ],
            )

        if approach == "gate" and decision.get("gate_command"):
            command = decision["gate_command"]
            ok = ctx.sandbox.exec(command, shell=True, timeout=300)
            # A proposed gate that does not run is not a gate. Verifying before
            # adopting prevents the platform from configuring itself into a
            # permanently red state.
            if not ok.ok and "not found" in ok.combined.lower():
                return AgentResult(
                    success=True,
                    summary="proposed gate command is not available on this host; recorded as a convention instead",
                    memory=[
                        convention(
                            objective[:90],
                            f"{decision.get('rationale', '')}\n\nProposed check: `{command}` "
                            f"(not installed on this host).",
                            source=f"improve:{ctx.node.id}",
                        )
                    ],
                    data=decision,
                )
            from ..validation.gates.command import custom_command_gate

            gate_name = f"custom_{ctx.node.id[-6:].lower()}"
            custom_command_gate(gate_name, command, description=objective[:100])
            return AgentResult(
                success=True,
                summary=f"registered gate '{gate_name}' running `{command}`",
                data={**decision, "gate_name": gate_name},
                memory=[
                    convention(
                        f"Gate: {gate_name}",
                        f"`{command}` runs as a validation gate. {decision.get('rationale', '')}",
                        source=f"improve:{ctx.node.id}",
                        tags=["gate"],
                    )
                ],
            )

        # config_change / test / code: produce an edit plan.
        payload = self.ask(
            ctx,
            builder,
            f"Implement this improvement using the '{approach}' approach. "
            f"{decision.get('edits', '')}\n\nReturn an edit plan.",
            schema=EDIT_PLAN_SCHEMA,
        )
        plan = EditPlan.from_payload(payload)
        applied = apply_edits(ctx.root, plan)
        report = self.run_gates(ctx, changed_files=applied.written)
        if not report.passed:
            return AgentResult.failure(
                f"the improvement broke validation: {report.summary_line()}", report=report
            )
        return AgentResult(
            success=True,
            summary=plan.summary or f"applied improvement to {len(applied.written)} file(s)",
            changed_files=applied.written,
            commit_message=f"chore: {plan.summary or objective[:80]}",
            data=decision,
            report=report,
        )
