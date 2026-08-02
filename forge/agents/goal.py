"""The goal agent: deciding whether the project is actually done.

Every other agent answers a local question -- did this task work. Something has
to answer the global one, and "all the nodes finished" is not the same as "the
thing the human asked for exists". A plan can be completed faithfully and still
miss the goal, because the plan was written before the system knew what it was
building.

So the goal node is a **barrier**: it becomes runnable only when nothing else
can make progress. At that point it re-reads the original goal and the recorded
requirements, runs the full validation suite, looks at what the project actually
produces, and judges. If it finds gaps it creates work to close them *and* a
fresh barrier node to judge again afterwards -- which is how the platform
converges on the goal rather than on its first plan.

The recursion is bounded. A goal check that keeps finding gaps after several
rounds is not converging, and the honest response is to stop and tell the human
what is missing rather than to keep spending.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..memory.context import P_ACCEPTANCE, P_FAILURE, P_TREE, file_tree, reference_images
from ..memory.records import MemoryKind
from ..models.provider import encode_image
from ..models.structured import array, boolean, enum, object_schema, string
from ..models.types import TaskClass
from ..obs.log import get_logger
from .base import Agent, AgentContext, AgentResult, ProposedNode
from .registry import register

log = get_logger("agents.goal")

#: Judging rounds before the platform concedes it is not converging.
# A high safety bound. Repeated identical gaps stop earlier via
# MAX_NO_PROGRESS_ROUNDS; changing gaps are evidence that work is converging.
MAX_ROUNDS = 12
MAX_NO_PROGRESS_ROUNDS = 3

COMPLETION_SCHEMA = object_schema(
    {
        "complete": boolean("True only if the stated goal is genuinely delivered"),
        "assessment": string("Two or three sentences on what exists and what does not"),
        "satisfied": array(string(), "Requirements that are demonstrably met"),
        "gaps": array(
            object_schema(
                {
                    "what": string("What is missing or wrong, specifically"),
                    "why_it_matters": string("Why the goal is not met without it"),
                    # No `document` and no `review`. A gap is by definition
                    # something standing between the current state and the
                    # goal, so closing it has to change the product. Offered
                    # the choice, a model routed "Missing Modern Presentation &
                    # Visual Effects" and "Broken Game Flow & HUD" to the
                    # documentation agent, which edited README.md and marked
                    # both gaps succeeded with the game untouched. Closing
                    # documentation is queued separately in `_finish`.
                    "kind": enum(
                        ["implement", "debug", "test_author", "browser_qa"],
                        "Which specialist should close this gap",
                    ),
                    "essential": boolean("False if this is polish rather than a requirement"),
                },
                required=["what", "kind", "essential"],
            ),
            "What stands between the current state and the goal. Empty if none.",
        ),
    },
    required=["complete", "assessment", "gaps"],
)


@register
class GoalAgent(Agent):
    """Judges the project against the goal the human actually stated."""

    kind = "goal"
    task_class = TaskClass.CODE_REVIEW
    difficulty = 0.7
    # Declaring a project finished when it is not is the single worst error the
    # platform can make: it is the one failure a human will not be warned about.
    stakes = 0.95
    commits = False

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You are deciding whether an autonomously built project actually delivers what \
was asked for. You are the last check before a human is told it is finished.

Judge against the original goal statement, not against the plan that was \
derived from it. A plan that was executed perfectly can still miss the point.

Be strict about these, which are the ways autonomous builds usually fall short:
- Something that compiles and passes tests but does not run end to end.
- A feature that exists in the code but is unreachable from the interface.
- Placeholder content, stub functions, or a "coming soon" where the substance \
should be.
- The literal requirement met while the evident intent is not. If the goal says \
"one polished level", one empty room satisfies the words and not the request.

Be equally strict in the other direction: do not invent gaps to seem rigorous. \
If the goal is met, say so. Padding the gap list costs real time and money and \
delays a finished project.
"""
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        round_index = int(ctx.spec.get("check_round", 0))

        report = self.run_gates(ctx)
        builder = self.builder(ctx)
        builder.add(
            "The goal as stated by the human",
            ctx.goal,
            priority=P_ACCEPTANCE - 1,
            max_tokens=800,
        )
        builder.add_records(
            "Recorded requirements",
            ctx.memory.by_kind(MemoryKind.REQUIREMENT, limit=30),
            priority=P_ACCEPTANCE,
            max_tokens=2500,
        )
        builder.add("Validation results", report.render(failures_only=False), priority=P_FAILURE, max_tokens=3500)
        builder.add("What was built", file_tree(ctx.root, limit=300), priority=P_TREE, max_tokens=2000)
        builder.add("Work completed", self._work_summary(ctx), priority=P_TREE + 1, max_tokens=2500)

        references, candidates = _goal_images(ctx.root, ctx.artifacts_dir)
        for path in [*references, *candidates]:
            try:
                label = "reference" if path in references else "candidate"
                builder.add_image(encode_image(path, label=f"{label}:{path.name}"))
            except OSError as exc:  # pragma: no cover - artifact disappeared
                ctx.logger().warn("could not read goal image", path=str(path), error=str(exc))
        if references or candidates:
            builder.add(
                "Visual evidence",
                "\n".join(
                    [
                        *(f"REFERENCE: {path.name}" for path in references),
                        *(f"CANDIDATE: {path.name}" for path in candidates),
                    ]
                ),
                priority=P_FAILURE,
                max_tokens=200,
            )

        open_findings = ctx.memory.open_findings()
        if open_findings:
            builder.add_records(
                "Unresolved findings from reviews",
                open_findings[:15],
                priority=P_FAILURE + 1,
                max_tokens=2000,
            )

        result: dict[str, Any] = self.ask(
            ctx,
            builder,
            "Does this project deliver the goal as stated? Judge what exists, not "
            "what was planned. List only gaps that genuinely stand between the "
            "current state and the goal.",
            schema=COMPLETION_SCHEMA,
            profile=self.profile(ctx, needs_vision=bool(references or candidates)),
        )

        essential = [g for g in result.get("gaps", []) if g.get("essential", True)]

        # A reviewer that returned `request_changes` and a goal check that
        # returns `complete` cannot both be right, and the reviewer is the one
        # that looked at a specific artifact and named a specific defect.
        #
        # Appearance and behaviour have no deterministic gate behind them, so
        # `report.passed` above says nothing about either: it means the code
        # compiled, linted and booted. Without this, the model's boolean was
        # the entire verdict, and it returned true with 36 findings still open
        # -- including "missing plunger lane, outlanes, drop-target bank,
        # ramps, multiplier lane and drain" on a pinball table, raised 26 hours
        # earlier. Treat those as gaps so the existing gap-closing machinery
        # turns them into work instead of stopping on top of them.
        blocking = _blocking_findings(open_findings)
        essential = _merge_finding_gaps(essential, blocking)
        complete = bool(result.get("complete")) and not essential and report.passed

        if complete:
            return self._finish(ctx, result, report)

        signatures = list(ctx.spec.get("gap_signatures", []))
        signature = "|".join(
            sorted(str(g.get("what", "")).strip().lower() for g in essential)
        )
        signatures.append(signature)
        unchanged = 0
        for value in reversed(signatures):
            if value != signature:
                break
            unchanged += 1

        if unchanged >= MAX_NO_PROGRESS_ROUNDS or round_index + 1 >= MAX_ROUNDS:
            # Not converging. Stop spending and hand a specific list to a human.
            missing = "\n".join(f"- {g['what']}" for g in essential) or "(see assessment)"
            return AgentResult(
                success=True,
                summary=f"goal gap-closing did not converge; {len(essential)} gap(s) remain",
                data={"complete": False, "assessment": result.get("assessment", ""), "gaps": essential},
                report=report,
                needs_human=(
                    "The project does not yet deliver the stated goal, and automatic "
                    f"gap-closing repeated the same gaps {unchanged} time(s).\n\n"
                    f"Assessment: {result.get('assessment', '')}\n\n"
                    f"Outstanding:\n{missing}\n\n"
                    "Either clarify the goal, relax a requirement, or fix the environment, "
                    "then `forge unblock <node> <guidance>`."
                ),
            )

        nodes = self._gap_nodes(ctx, essential, report)
        nodes.append(
            ProposedNode(
                kind="goal",
                title=f"Re-check the goal (round {round_index + 2})",
                spec={
                    "objective": "Judge whether the project now delivers the stated goal",
                    "acceptance": ["The stated goal is delivered and verified"],
                    "barrier": True,
                    "check_round": round_index + 1,
                    "gap_signatures": signatures[-MAX_NO_PROGRESS_ROUNDS:],
                },
                priority=990,
            )
        )
        return AgentResult(
            success=True,
            summary=f"goal not yet met: {len(essential)} gap(s); queued work to close them",
            data={"complete": False, "assessment": result.get("assessment", ""), "gaps": result.get("gaps", [])},
            report=report,
            nodes=nodes,
        )

    # -- helpers ---------------------------------------------------------

    def _finish(self, ctx: AgentContext, result: dict[str, Any], report: Any) -> AgentResult:
        """Queue the closing work: documentation, then deployment if enabled."""
        nodes: list[ProposedNode] = []
        existing = {n.kind for n in ctx.graph.all_nodes()}

        if "document" not in existing:
            nodes.append(
                ProposedNode(
                    kind="document",
                    title="Document the finished project",
                    spec={
                        "objective": "Write the documentation a person needs to run and maintain this",
                        "acceptance": ["README explains install, run and test", "Project memory exported"],
                    },
                    priority=980,
                )
            )
        if ctx.config.deploy.enabled and "deploy" not in existing:
            nodes.append(
                ProposedNode(
                    kind="deploy",
                    title="Deploy the project",
                    spec={
                        "objective": "Ship the built artefact behind a health check",
                        "acceptance": ["Deployment succeeded and the health check passed"],
                    },
                    deps=[0] if nodes else [],
                    priority=985,
                )
            )
        if nodes:
            # One more barrier after the closing work, so documentation and
            # deployment are themselves verified rather than assumed.
            nodes.append(
                ProposedNode(
                    kind="goal",
                    title="Final verification",
                    spec={
                        "objective": "Confirm the delivered project after documentation and deployment",
                        "acceptance": ["The stated goal is delivered and verified"],
                        "barrier": True,
                        "check_round": MAX_ROUNDS - 1,
                    },
                    priority=999,
                )
            )

        from ..memory.store import fact

        log.info("goal met", assessment=result.get("assessment", "")[:120])
        return AgentResult(
            success=True,
            summary="the stated goal is delivered" + (f"; queued {len(nodes)} closing task(s)" if nodes else ""),
            data={"complete": True, "assessment": result.get("assessment", ""),
                  "satisfied": result.get("satisfied", [])},
            report=report,
            nodes=nodes,
            memory=[
                fact(
                    "Goal completion assessment",
                    result.get("assessment", ""),
                    source=f"goal:{ctx.node.id}",
                    tags=["completion"],
                )
            ],
            milestone_reached=ctx.node.milestone or "",
        )

    def _gap_nodes(self, ctx: AgentContext, gaps: list[dict[str, Any]], report: Any) -> list[ProposedNode]:
        nodes: list[ProposedNode] = []
        for gap in gaps[:8]:
            nodes.append(
                ProposedNode(
                    kind=_gap_kind(gap),
                    title=f"Close gap: {gap['what'][:70]}",
                    spec={
                        "objective": f"{gap['what']}\n\nWhy this matters: {gap.get('why_it_matters', '')}",
                        "acceptance": [f"The gap is closed: {gap['what']}"],
                        "failure": {"summary": gap["what"], "evidence": report.render()},
                        # Set when this gap came from an unresolved review
                        # finding, so closing the work also closes the finding.
                        **({"resolves_finding": gap["finding_id"]} if gap.get("finding_id") else {}),
                    },
                    priority=40,
                    milestone=ctx.node.milestone,
                )
            )
        return nodes

    def _work_summary(self, ctx: AgentContext) -> str:
        lines: list[str] = []
        for node in ctx.graph.all_nodes():
            if node.status != "succeeded" or node.kind in ("goal", "plan", "retrospect"):
                continue
            summary = (node.result or {}).get("summary", "")
            lines.append(f"- [{node.kind}] {node.title}: {str(summary)[:120]}")
        commits = ctx.repo.log(limit=25) if ctx.repo.has_commits() else []
        if commits:
            lines.append("")
            lines.append("Commits:")
            lines += [f"  {c.subject}" for c in commits]
        return "\n".join(lines[:120])


#: Severities a reviewer used to mean "this is not finished". Anything below
#: stays advisory memory for the next planning pass, which is the existing
#: contract in `reviewing.py`.
_BLOCKING_SEVERITIES = frozenset({"critical", "high"})

#: Sources whose findings a *judgement* produced, as opposed to a gate.
_REVIEW_SOURCES = ("review:", "visual:", "security:", "perf:", "browser_qa")


def _blocking_findings(records: list[Any]) -> list[Any]:
    """Unresolved review findings severe enough to contradict "delivered".

    Gate-sourced findings are excluded on purpose, and the distinction is the
    whole reason this is not a one-liner. A gate re-runs on every validation:
    if its complaint is still true, `report.passed` is already False and says
    so with fresh evidence. Its old records are a log, not a backlog, and the
    project this was written against had 30 stale `gate:types` findings such as
    "'add' is declared but its value is never read" against 4 real review
    findings. Blocking on all of them would have buried the four that mattered
    under two-day-old compile errors that no longer reproduce.

    A review finding is different: nothing re-derives it, so it stands until
    something closes it.
    """
    blocking = []
    for record in records:
        severity = str((getattr(record, "data", None) or {}).get("severity", "")).lower()
        source = str(getattr(record, "source", ""))
        if severity in _BLOCKING_SEVERITIES and source.startswith(_REVIEW_SOURCES):
            blocking.append(record)
    return blocking


def _merge_finding_gaps(gaps: list[dict[str, Any]], findings: list[Any]) -> list[dict[str, Any]]:
    """Add each unresolved finding as a gap, without duplicating one the model named.

    The model is asked for gaps in its own words, so it often restates a
    finding it was shown. Matching on the title keeps the gap list from
    doubling every round, which would defeat the no-progress detector.
    """
    merged = list(gaps)
    seen = {str(gap.get("what", "")).strip().lower() for gap in merged}
    for record in findings:
        title = str(getattr(record, "title", "")).strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        merged.append(
            {
                "what": title,
                "why_it_matters": str(getattr(record, "body", ""))[:600],
                "essential": True,
                "kind": "implement",
                "source": "review-finding",
                "finding_id": getattr(record, "id", ""),
            }
        )
    return merged


#: Node kinds that can actually close a gap, because they change the product.
_GAP_KINDS = frozenset({"implement", "debug", "test_author", "browser_qa"})


def _gap_kind(gap: dict[str, Any]) -> str:
    """Coerce a gap to a specialist that changes the product.

    Enforced here as well as in the schema because the schema is advice to a
    model and this is not negotiable: a documentation node closing "Missing
    Modern Presentation & Visual Effects" reports success having touched
    nothing but README.md, and the gap is then considered closed.
    """
    kind = str(gap.get("kind", "implement"))
    return kind if kind in _GAP_KINDS else "implement"


def _goal_images(root: Path, artifacts_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return the human reference and newest normal/screenshot-mode captures."""
    # Deciding "does this look like the thing that was asked for" is the one
    # judgement with no deterministic gate behind it, so it gets the whole
    # reference set rather than whichever single file happened to sort first.
    references = reference_images(root, limit=4)
    def newest_capture(path: Path) -> int:
        try:
            return max(
                (candidate.stat().st_mtime_ns for candidate in path.glob("screenshot*.png")),
                default=path.stat().st_mtime_ns,
            )
        except OSError:
            return 0

    directories = sorted(
        (path for path in artifacts_dir.iterdir() if path.is_dir()),
        key=newest_capture,
        reverse=True,
    ) if artifacts_dir.is_dir() else []
    for directory in directories:
        candidates = sorted(
            path
            for path in directory.glob("screenshot*.png")
            if not path.name.startswith(("diff_", "smoke_"))
        )
        if candidates:
            return references, candidates[:2]
    return references, []
