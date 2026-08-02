"""Agents whose job is to check, not to build.

``ValidateAgent`` is the plain one: run the gates, report. It exists as a node
kind so a plan can place validation explicitly on the critical path -- "the
renderer and the physics must both pass integration before the level loads" is
a dependency, and dependencies belong in the graph.

``BrowserQAAgent`` is the interesting one. It is the bridge between "the code
compiles" and "a person could use this", and it works by having a model design
the interaction flow while a deterministic harness executes it. The model never
writes browser code; it fills in a fixed step vocabulary that either runs or
fails schema validation. That split keeps the flakiness of generated automation
out of the loop while keeping the creativity of deciding *what to try* in it.
"""

from __future__ import annotations

from typing import Any

from ..memory.context import P_FAILURE, P_TREE, file_tree
from ..memory.store import fact
from ..models.types import TaskClass
from ..obs.log import get_logger
from ..validation.gates.browser import FLOW_SCHEMA, playwright_available
from ..validation.types import Severity
from .base import Agent, AgentContext, AgentResult, ProposedNode
from .registry import register

log = get_logger("agents.validating")


@register
class ValidateAgent(Agent):
    """Runs the configured gates and reports the verdict.

    Produces repair nodes rather than fixing anything itself. Keeping detection
    and repair in separate nodes means a validation run that finds four
    unrelated failures produces four independently schedulable repairs, which
    can then proceed in parallel and be retried or escalated separately.
    """

    kind = "validate"
    task_class = TaskClass.CLASSIFICATION
    difficulty = 0.1
    stakes = 0.3
    commits = False

    def run(self, ctx: AgentContext) -> AgentResult:
        report = self.run_gates(ctx)
        records = []
        nodes: list[ProposedNode] = []

        if report.passed:
            return AgentResult(
                success=True,
                summary=report.summary_line(),
                report=report,
                data=report.to_dict(),
            )

        blocking = [v for v in report.failures if v.errored or self._blocking(v.gate)]
        for verdict in blocking:
            nodes.append(
                ProposedNode(
                    kind="debug",
                    title=f"Fix {verdict.gate} failures",
                    spec={
                        "objective": f"The {verdict.gate} gate is failing. Fix it.",
                        "acceptance": [f"The {verdict.gate} gate passes"],
                        "failure": verdict.to_dict() | {"evidence": verdict.render()},
                        "paths": sorted({i.path for i in verdict.issues if i.path}),
                        "gates": [verdict.gate],
                    },
                    priority=20,
                    milestone=ctx.node.milestone,
                )
            )
        records.append(
            fact(
                f"Validation failed for {ctx.node.title}",
                report.render(),
                source=f"node:{ctx.node.id}",
                tags=["validation"],
            )
        )

        # The node itself succeeds: it validated, and validation found problems.
        # Reporting a failure here would retry the *measurement*, which would
        # produce the same answer at the same cost.
        return AgentResult(
            success=True,
            summary=f"{report.summary_line()}; {len(nodes)} repair task(s) created",
            report=report,
            data=report.to_dict(),
            memory=records,
            nodes=nodes,
        )

    @staticmethod
    def _blocking(gate_name: str) -> bool:
        from ..validation.gate import gate_registry

        try:
            return gate_registry.create(gate_name).blocking
        except Exception:  # pragma: no cover
            return True


@register
class BrowserQAAgent(Agent):
    """Exercises the running application as a user would.

    Designs a flow from the acceptance criteria, runs it in a real browser,
    captures screenshots, and files what broke. When the flow passes it queues a
    visual review of the captures, which is how "it works" gets upgraded to "it
    works and looks right" without a human in the loop.
    """

    kind = "browser_qa"
    task_class = TaskClass.TEST_AUTHORING
    difficulty = 0.55
    stakes = 0.6
    commits = False

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You design a user flow to be executed against the running application by an \
automated browser.

You do not write code. You produce a list of steps from a fixed vocabulary: \
click, fill, press, key_down, key_up, wait, wait_for, goto, reload, resize, \
tab_away, expect_fps, expect_text, expect_selector.

Design it like a sceptical tester:
- Follow the path that proves the acceptance criteria, not a tour of the UI.
- Assert something after every action that should change the page. A flow with \
no expectations passes against a broken application.
- Prefer selectors that will survive a restyle: data attributes, roles, stable \
ids. Avoid selectors built from generated class names.
- Put CSS selectors in `selector`. `goto` values may be relative routes. `wait` \
and `tab_away` values are seconds; `resize` is WIDTHxHEIGHT. For `expect_fps`, \
use value `minimum_fps,maximum_frame_ms` and duration in milliseconds.
- Use key_down, wait, key_up for controls whose hold duration matters, such as \
flippers and a pinball plunger. `press` is only a momentary tap.
- Keep it under a dozen steps. Long flows fail for uninteresting reasons.
"""
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        if not playwright_available():
            return AgentResult(
                success=True,
                summary="browser QA skipped: playwright is not installed",
                memory=[
                    fact(
                        "Playwright unavailable",
                        "Browser QA cannot run on this host. Install with "
                        "`pip install playwright && playwright install chromium`.",
                        source="browser_qa",
                        tags=["environment", "blocked"],
                    )
                ],
            )

        flow = self._design_flow(ctx)
        gate_settings = dict(ctx.spec.get("gate_settings", {}))
        gate_settings.setdefault("smoke", {})["steps"] = flow.get("steps", [])
        gate_settings.setdefault("browser", {})["routes"] = ctx.spec.get("routes", ["/"])


        gate_ctx = ctx.gates.build_context(
            root=ctx.root,
            sandbox=ctx.sandbox,
            toolchain=ctx.toolchain,
            node_id=ctx.node.id,
            settings=gate_settings,
            memory=ctx.memory,
        )
        report = ctx.gates.run(["browser", "smoke", "visual"], gate_ctx)

        artifacts = [path for verdict in report.verdicts for path in verdict.artifacts]
        nodes: list[ProposedNode] = []
        records = []

        for verdict in report.failures:
            critical = [i for i in verdict.issues if i.severity == Severity.CRITICAL]
            if not critical and verdict.ok:
                continue
            nodes.append(
                ProposedNode(
                    kind="debug",
                    title=f"Fix browser failures in {ctx.node.title[:60]}",
                    spec={
                        "objective": "The application fails when driven in a real browser. Fix it.",
                        "acceptance": ["The browser and smoke gates pass"],
                        "failure": verdict.to_dict() | {"evidence": verdict.render()},
                        "gates": ["browser", "smoke"],
                        "gate_settings": gate_settings,
                    },
                    priority=25,
                    milestone=ctx.node.milestone,
                )
            )

        visual = next((v for v in report.verdicts if v.gate == "visual"), None)
        changed = bool(visual and visual.detail.get("changed"))
        if artifacts and (report.passed or changed):
            # Only look at pixels with a model when something actually changed
            # or when the flow passed and there is something new to judge.
            nodes.append(
                ProposedNode(
                    kind="visual_review",
                    title=f"Review the appearance of {ctx.node.title[:60]}",
                    spec={
                        "objective": "Judge whether the captured screens look finished",
                        "acceptance": ["Screens reviewed and findings recorded"],
                        "images": artifacts,
                    },
                    priority=200,
                    milestone=ctx.node.milestone,
                )
            )

        if flow.get("steps"):
            records.append(
                fact(
                    f"Browser flow for {ctx.node.title[:60]}",
                    "\n".join(
                        f"{i + 1}. {s.get('action')} {s.get('selector', '')} {s.get('value', '')}".strip()
                        for i, s in enumerate(flow["steps"])
                    ),
                    source="browser_qa",
                    tags=["flow", "qa"],
                )
            )

        return AgentResult(
            success=True,
            summary=f"browser QA: {report.summary_line()}",
            report=report,
            data={"flow": flow, "artifacts": artifacts},
            nodes=nodes,
            memory=records,
            artifacts=artifacts,
        )

    def _design_flow(self, ctx: AgentContext) -> dict[str, Any]:
        provided = ctx.spec.get("steps")
        if provided:
            return {"steps": provided}

        builder = self.builder(ctx)
        tree = file_tree(ctx.root, limit=150)
        builder.add("Repository layout", tree, priority=P_TREE, max_tokens=1000)
        markup = self._markup_hints(ctx)
        if markup:
            builder.add("Selectors present in the source", markup, priority=P_FAILURE, max_tokens=2500)

        acceptance = ctx.node.acceptance or ["The application loads and its main interaction works"]
        task = (
            "Design a browser flow that proves these acceptance criteria:\n"
            + "\n".join(f"- {item}" for item in acceptance)
            + "\n\nUse only selectors you can see in the source above."
        )
        try:
            return self.ask(ctx, builder, task, schema=FLOW_SCHEMA)
        except Exception as exc:  # a flow we cannot design is not a fatal error
            ctx.logger().warn("could not design a browser flow", error=str(exc))
            return {"steps": []}

    def _markup_hints(self, ctx: AgentContext) -> str:
        """Extract candidate selectors from source, so the model does not invent them.

        Cheap grep-style extraction beats showing whole component files: it is a
        fraction of the tokens and gives the model exactly the vocabulary it
        needs to write a flow that will actually match something.
        """
        import re

        pattern = re.compile(r"""(?:id|data-testid|data-test|role|aria-label)\s*=\s*["']([^"']{2,60})["']""")
        found: list[str] = []
        for path in sorted(ctx.root.rglob("*")):
            if path.suffix not in {".html", ".tsx", ".jsx", ".vue", ".svelte", ".ts", ".js"}:
                continue
            rel = path.relative_to(ctx.root)
            if any(part in {"node_modules", ".git", "dist", "build", ".forge"} for part in rel.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover
                continue
            for match in pattern.finditer(text):
                found.append(f"{rel.as_posix()}: {match.group(0)}")
            if len(found) > 120:
                break
        return "\n".join(found[:120])
