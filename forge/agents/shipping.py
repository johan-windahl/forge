"""Documentation and deployment.

Deployment is the step where autonomy is most dangerous and most valuable, so it
is the most conservative agent in the platform: it validates first, deploys
behind a health check, and rolls back automatically when the check fails. There
is no path where a failing health check leaves the deployed artefact in place
and merely logs a warning.

Deployment strategies are pluggable because "deploy" means something different
for a static site, a container and a script, and pretending otherwise produces a
single strategy that serves none of them well.
"""

from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..errors import PatchError
from ..kernel.events import EventType
from ..memory.context import P_TASK_FILES, P_TREE, file_tree, read_files
from ..memory.store import fact
from ..models.structured import array, object_schema, string
from ..models.types import TaskClass
from ..obs.log import get_logger
from ..workspace.patch import EDIT_PLAN_SCHEMA, EditPlan, apply_edits
from .base import Agent, AgentContext, AgentResult
from .registry import register

log = get_logger("agents.shipping")


def _documentation_files(root: Path) -> list[str]:
    """Files a README author must see before replacing or documenting them."""
    return [
        name
        for name in (
            "README.md",
            "package.json",
            "pyproject.toml",
            "Makefile",
            "Cargo.toml",
            "docker-compose.yml",
        )
        if (root / name).is_file()
    ]


@register
class DocumentAgent(Agent):
    """Writes the documentation a human needs to take the project over.

    Scoped to what only a human reader needs -- how to run it, how it is shaped,
    what was assumed. The architecture document and the memory export are
    generated elsewhere; duplicating them here would create two descriptions
    that drift.
    """

    kind = "document"
    task_class = TaskClass.DOCUMENTATION
    difficulty = 0.3
    stakes = 0.4
    commits = True

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You write documentation for a person who has just been handed this repository \
and has to run it.

Cover, in this order: what it is, how to install and run it, how to run the \
tests, how the code is laid out, and anything surprising about the environment.

Be concrete. Every command you write must be one that actually works in this \
repository -- you have its contents, so check rather than assume. Do not \
describe features that do not exist, and do not pad.
"""
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        builder = self.builder(ctx)
        builder.add("Repository layout", file_tree(ctx.root, limit=250), priority=P_TREE, max_tokens=1800)
        key_files = _documentation_files(ctx.root)
        builder.add_files("Project manifests", read_files(ctx.root, key_files), priority=P_TASK_FILES, max_tokens=3000)
        builder.add(
            "Known commands",
            "\n".join(f"{k}: {v}" for k, v in ctx.toolchain.get("commands", {}).items()),
            priority=P_TASK_FILES,
            max_tokens=400,
        )

        rejection = ""
        applied = None
        for round_index in range(3):
            task = "Write or update README.md for this project. Return it as an edit plan."
            if rejection:
                task += (
                    "\n\nYour previous edit plan was rejected before any file changed: "
                    f"{rejection}. Correct the operation and return a valid plan."
                )
            payload = self.ask(ctx, builder, task, schema=EDIT_PLAN_SCHEMA)
            plan = EditPlan.from_payload(payload)
            try:
                applied = apply_edits(ctx.root, plan)
                break
            except PatchError as exc:
                rejection = str(exc)
                ctx.logger().warn(
                    "documentation edit plan rejected; retrying locally",
                    error=rejection,
                    round=round_index + 1,
                )
        if applied is None:
            return AgentResult.failure(
                f"could not apply documentation edit plan after 3 rounds: {rejection}",
                needs_escalation=True,
            )

        # Project memory is also exported as documentation, so the assumptions
        # and decisions Forge made are visible to a human without any tooling.
        docs = ctx.root / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "project-memory.md").write_text(ctx.memory.export_markdown(), encoding="utf-8")

        changed = sorted({*applied.written, "docs/project-memory.md"})
        return AgentResult(
            success=True,
            summary=f"documentation updated ({len(changed)} file(s))",
            changed_files=changed,
            commit_message="docs: update project documentation",
        )


DEPLOY_PLAN_SCHEMA = object_schema(
    {
        "strategy": string("static, docker or script"),
        "steps": array(string(), "Shell commands to run, in order", minItems=1),
        "artifact_path": string("Directory or file that constitutes the deployable output"),
        "healthcheck_url": string("URL to poll after deploying, if applicable"),
        "rollback_steps": array(string(), "Commands that undo this deployment"),
    },
    required=["strategy", "steps"],
)


@register
class DeployAgent(Agent):
    """Ships the built artefact, with an automatic rollback on failure."""

    kind = "deploy"
    task_class = TaskClass.IMPLEMENTATION
    difficulty = 0.5
    stakes = 0.9
    commits = False

    def run(self, ctx: AgentContext) -> AgentResult:
        config = ctx.config.deploy
        if not config.enabled and not ctx.spec.get("force"):
            return AgentResult(
                success=True,
                summary="deployment is disabled in configuration",
                data={"skipped": True},
            )

        # Never deploy something that does not pass its gates. This is the one
        # place where the platform refuses to proceed on a failing verdict
        # regardless of severity settings.
        report = self.run_gates(ctx)
        if not report.passed:
            return AgentResult.failure(
                f"refusing to deploy: {report.summary_line()}",
                report=report,
                data={"blocked_by": [v.gate for v in report.failures]},
            )

        plan = self._plan(ctx, config)
        ctx.models.ledger.emit(EventType.DEPLOY_STARTED, node_id=ctx.node.id, strategy=plan["strategy"])

        previous = self._snapshot(ctx, plan)
        outputs: list[str] = []
        try:
            for step in plan["steps"]:
                result = ctx.sandbox.exec(step, shell=True, timeout=ctx.config.sandbox.command_timeout)
                outputs.append(f"$ {step}\n{result.tail(30)}")
                if not result.ok:
                    raise RuntimeError(f"deploy step failed: {step}\n{result.tail(40)}")

            healthy, detail = self._healthcheck(plan, config)
            if not healthy:
                raise RuntimeError(f"health check failed: {detail}")

        except (RuntimeError, OSError) as exc:
            rolled_back = self._rollback(ctx, plan, previous) if config.rollback_on_failure else False
            ctx.models.ledger.emit(
                EventType.DEPLOY_FAILED, node_id=ctx.node.id, error=str(exc), rolled_back=rolled_back
            )
            return AgentResult.failure(
                f"deployment failed and was {'rolled back' if rolled_back else 'NOT rolled back'}: {exc}",
                data={"output": "\n\n".join(outputs)[-4000:], "rolled_back": rolled_back},
            )

        ctx.models.ledger.emit(
            EventType.DEPLOY_SUCCEEDED, node_id=ctx.node.id, strategy=plan["strategy"]
        )
        return AgentResult(
            success=True,
            summary=f"deployed via {plan['strategy']}",
            data={"plan": plan, "output": "\n\n".join(outputs)[-3000:]},
            memory=[
                fact(
                    "Deployment procedure",
                    "\n".join(plan["steps"]),
                    source="deploy",
                    tags=["deploy", plan["strategy"]],
                )
            ],
        )

    def _plan(self, ctx: AgentContext, config: Any) -> dict[str, Any]:
        """Use the configured procedure if there is one; otherwise design it.

        Operator configuration always wins. Asking a model to invent a
        deployment procedure for infrastructure it cannot see is exactly the
        kind of confident guess that should not reach production.
        """
        if config.command:
            return {
                "strategy": config.strategy,
                "steps": [config.command],
                "artifact_path": ctx.spec.get("artifact_path", "dist"),
                "healthcheck_url": config.healthcheck_url,
                "rollback_steps": [],
            }
        builder = self.builder(ctx)
        builder.add("Repository layout", file_tree(ctx.root, limit=150), priority=P_TREE, max_tokens=1200)
        builder.add(
            "Known commands",
            "\n".join(f"{k}: {v}" for k, v in ctx.toolchain.get("commands", {}).items()),
            priority=P_TASK_FILES,
            max_tokens=400,
        )
        plan: dict[str, Any] = self.ask(
            ctx,
            builder,
            f"Produce a deployment procedure using the '{config.strategy}' strategy. "
            "Use only commands that will work in this repository with the tools it "
            "already declares. Do not invent infrastructure.",
            schema=DEPLOY_PLAN_SCHEMA,
        )
        plan.setdefault("healthcheck_url", config.healthcheck_url)
        return plan

    def _snapshot(self, ctx: AgentContext, plan: dict[str, Any]) -> Path | None:
        """Copy the current deployed artefact so rollback has something to restore."""
        artifact = plan.get("artifact_path")
        if not artifact:
            return None
        source = ctx.root / artifact
        if not source.exists():
            return None
        backup = ctx.config.artifacts_dir / "deploy-backup" / ctx.node.id
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, backup)
        else:
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup / source.name)
        return backup

    def _rollback(self, ctx: AgentContext, plan: dict[str, Any], backup: Path | None) -> bool:
        ok = True
        for step in plan.get("rollback_steps", []):
            result = ctx.sandbox.exec(step, shell=True, timeout=300)
            ok = ok and result.ok
        if backup and backup.exists() and plan.get("artifact_path"):
            target = ctx.root / plan["artifact_path"]
            try:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                    shutil.copytree(backup, target)
                else:
                    shutil.copy2(next(backup.iterdir()), target)
            except (OSError, StopIteration) as exc:  # pragma: no cover
                log.error("rollback restore failed", error=str(exc))
                ok = False
        ctx.models.ledger.emit(EventType.DEPLOY_ROLLED_BACK, node_id=ctx.node.id, ok=ok)
        return ok

    def _healthcheck(self, plan: dict[str, Any], config: Any) -> tuple[bool, str]:
        url = plan.get("healthcheck_url") or config.healthcheck_url
        if not url:
            return True, "no health check configured"
        deadline = time.monotonic() + config.healthcheck_timeout
        last = "no attempt made"
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    if 200 <= response.status < 400:
                        return True, f"HTTP {response.status}"
                    last = f"HTTP {response.status}"
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}"
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                last = str(exc)
            time.sleep(2.0)
        return False, last
