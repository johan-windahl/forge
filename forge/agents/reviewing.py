"""Review agents: model judgement, applied where measurement cannot reach.

The platform's stance is that gates decide correctness and reviews decide
*quality* -- the things no compiler has an opinion about: whether the code says
what it means, whether the abstraction is the right one, whether a screenshot
looks like a finished product or a wireframe.

Two design commitments keep reviews from becoming expensive noise:

**Reviews run on diffs, not repositories.** A review that reads the whole
codebase costs a fortune and produces generic advice. A review of the change
produces specific advice about the change.

**Findings must be actionable and severity-rated.** A finding that cannot be
turned into a task is not recorded. Severity determines whether a repair node is
created now, filed for the milestone, or simply noted -- so a low-severity
opinion never blocks a run, and a high-severity one never gets lost.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..memory.context import P_FAILURE, P_TASK_FILES, read_files, reference_images
from ..memory.records import MemoryRecord
from ..memory.store import finding
from ..models.provider import encode_image
from ..models.structured import array, boolean, enum, object_schema, string
from ..models.types import TaskClass
from ..obs.log import get_logger
from ..validation.types import Severity
from .base import Agent, AgentContext, AgentResult, ProposedNode
from .registry import register

log = get_logger("agents.reviewing")

_SEVERITIES = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
_WHOLE_CODE_REVIEW = re.compile(
    r"\b(?:whole|entire|full)\s+(?:milestone|codebase|project|repository)|"
    r"\barchitecture\s+contract\b",
    re.IGNORECASE,
)

FINDINGS_SCHEMA = object_schema(
    {
        "verdict": enum(
            ["approve", "approve_with_findings", "request_changes"],
            "Overall judgement of the change",
        ),
        "summary": string("Two sentences: what the change does and whether it is sound"),
        "findings": array(
            object_schema(
                {
                    "title": string("Short, specific: 'Collision check skips the Y axis'"),
                    "detail": string("What is wrong and why it matters. Be concrete."),
                    "severity": enum(_SEVERITIES, "How much this matters"),
                    "path": string("File it applies to, if a specific one"),
                    "suggestion": string("What to do instead"),
                    "certain": boolean("False if you are inferring rather than seeing the problem"),
                },
                required=["title", "detail", "severity"],
            ),
            "Problems worth someone's time. An empty list is a valid and common answer.",
        ),
    },
    required=["verdict", "summary", "findings"],
)


class ReviewingAgent(Agent):
    """Shared behaviour: turn findings into memory records and repair nodes."""

    task_class = TaskClass.CODE_REVIEW
    commits = False

    def _process_findings(
        self, ctx: AgentContext, result: dict[str, Any], *, source: str
    ) -> tuple[list[MemoryRecord], list[ProposedNode]]:
        records: list[MemoryRecord] = []
        nodes: list[ProposedNode] = []

        for item in result.get("findings", []):
            severity = str(item.get("severity", Severity.MEDIUM))
            body = item.get("detail", "")
            if item.get("suggestion"):
                body += f"\n\nSuggested fix: {item['suggestion']}"
            if not item.get("certain", True):
                body += "\n\n(Reviewer was not certain; verify before acting.)"
            record = finding(
                item["title"],
                body,
                severity=severity,
                paths=[item["path"]] if item.get("path") else [],
                source=source,
            )
            records.append(record)
            # Only high-severity findings become work immediately. Everything
            # else is memory the next planning pass can weigh -- otherwise a
            # thorough reviewer can generate more tasks than the project has.
            if severity in (Severity.HIGH, Severity.CRITICAL):
                nodes.append(
                    ProposedNode(
                        kind="debug" if severity == Severity.CRITICAL else "implement",
                        title=f"Address: {item['title'][:80]}",
                        spec={
                            "objective": body,
                            "acceptance": [f"The reviewed problem is resolved: {item['title']}"],
                            "paths": [item["path"]] if item.get("path") else [],
                            "failure": {"summary": item["title"], "evidence": body},
                            # Closes the loop. Without this the node fixes the
                            # problem and the finding stays `active` forever:
                            # `resolve_finding` existed and had no callers, so
                            # a project accumulated open findings it had
                            # already fixed and nothing could tell the two
                            # apart.
                            "resolves_finding": record.id,
                        },
                        priority=30 if severity == Severity.CRITICAL else 80,
                        milestone=ctx.node.milestone,
                    )
                )
        return records, nodes


@register
class CodeReviewAgent(ReviewingAgent):
    """Reviews the change produced by this node's dependencies."""

    kind = "review"
    difficulty = 0.6
    stakes = 0.7

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You are a code reviewer. Automated gates have already run -- compilation, \
tests, linting, type checking. Do not repeat them; report what they cannot see.

Look for:
- Logic that is wrong in a case the tests do not cover.
- State that can be mutated from two places, or lifecycle that can leak.
- Error paths that swallow failures or leave the system half-updated.
- Abstractions that will not survive the next requirement, where you can name \
that requirement.
- Code that contradicts a recorded interface, convention or decision.

Do not report:
- Style the formatter already governs.
- Speculative performance concerns without a reason to think this is hot.
- Preferences. If you would have written it differently but this way is \
correct and clear, say nothing.

An empty findings list is a good review of good code. Say so and move on.
"""
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        objective = str(ctx.spec.get("objective", ""))
        whole_review = bool(_WHOLE_CODE_REVIEW.search(f"{ctx.node.title} {objective}"))
        diff = self._diff(ctx)
        if not diff.strip() and not whole_review:
            return AgentResult(success=True, summary="no changes to review")

        builder = self.builder(ctx)
        if whole_review:
            paths = ctx.spec.get("paths", []) or _whole_review_paths(ctx.root)
            builder.add_files(
                "Milestone codebase under review",
                read_files(ctx.root, paths, max_bytes_each=80_000),
                priority=P_TASK_FILES,
                max_tokens=16_000,
            )
            if diff.strip():
                builder.add(
                    "Most recent dependency diff",
                    diff,
                    priority=P_TASK_FILES + 5,
                    max_tokens=2_000,
                    tail_lines=100,
                )
            review_task = (
                "Review the milestone codebase against the project's requirements, "
                "interfaces and conventions, with special attention to the objective "
                f"below. Report only concrete issues.\n\n{objective}"
            )
        else:
            builder.add(
                "The change under review",
                diff,
                priority=P_TASK_FILES,
                max_tokens=12_000,
                tail_lines=400,
            )
            paths = ctx.spec.get("paths", []) or _paths_from_diff(diff)
            context_files = read_files(ctx.root, [p for p in paths if p not in diff][:8])
            builder.add_files(
                "Surrounding code",
                context_files,
                priority=P_TASK_FILES + 5,
                max_tokens=6_000,
            )
            review_task = (
                "Review this change against the project's requirements, interfaces "
                "and conventions. Report only what matters."
            )

        result: dict[str, Any] = self.ask(
            ctx,
            builder,
            review_task,
            schema=FINDINGS_SCHEMA,
        )
        records, nodes = self._process_findings(ctx, result, source=f"review:{ctx.node.id}")
        blocking = [f for f in result.get("findings", []) if f.get("severity") == Severity.CRITICAL]

        return AgentResult(
            # A review that finds problems has succeeded at reviewing. The
            # problems become work; they do not fail the review node.
            success=True,
            summary=f"{result.get('verdict', 'reviewed')}: {len(result.get('findings', []))} finding(s)",
            data={"verdict": result.get("verdict"), "summary": result.get("summary"),
                  "critical": len(blocking)},
            memory=records,
            nodes=nodes,
        )

    def _diff(self, ctx: AgentContext) -> str:
        base = ctx.spec.get("base_ref")
        if base:
            return ctx.repo.diff(f"{base}..HEAD")
        # Default: everything this node's dependencies committed.
        dep_commits = []
        for dep_id in ctx.node.deps:
            dep = ctx.graph.try_get(dep_id)
            if dep and dep.result and dep.result.get("commit"):
                dep_commits.append(dep.result["commit"])
        if dep_commits:
            return ctx.repo.diff(f"{dep_commits[0]}~1..HEAD")
        return ctx.repo.diff("HEAD~1..HEAD") if ctx.repo.has_commits() else ""


def _paths_from_diff(diff: str) -> list[str]:
    paths: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[6:].strip())
    return paths


def _whole_review_paths(root: Path, *, limit: int = 80) -> list[str]:
    """Select reviewable project files, with contracts before implementations."""
    preferred: list[Path] = []
    for relative in ("docs/architecture.md", "docs/ARCHITECTURE.md", "README.md"):
        path = root / relative
        if path.is_file():
            preferred.append(path)
    for directory in (root / "src", root / "tests"):
        if directory.is_dir():
            preferred.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    for pattern in ("*.json", "*.toml", "*.yaml", "*.yml", "*.config.ts", "*.config.js"):
        preferred.extend(sorted(path for path in root.glob(pattern) if path.is_file()))

    allowed = {".md", ".ts", ".tsx", ".js", ".jsx", ".py", ".json", ".toml", ".yaml", ".yml"}
    seen: set[str] = set()
    selected: list[str] = []
    for path in preferred:
        if path.suffix.lower() not in allowed:
            continue
        relative_path = path.relative_to(root)
        if any(
            part in {"node_modules", ".git", "dist", "build", ".forge"}
            for part in relative_path.parts
        ):
            continue
        value = relative_path.as_posix()
        if value in seen:
            continue
        seen.add(value)
        selected.append(value)
        if len(selected) >= limit:
            break
    return selected


@register
class SecurityReviewAgent(ReviewingAgent):
    """Reads the code for security problems the pattern scanner cannot see.

    Runs *after* the deterministic security gates, and is told what they already
    found so it does not spend its context rediscovering a hard-coded key that a
    regex caught for free.
    """

    kind = "security"
    difficulty = 0.7
    stakes = 0.9

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You are reviewing for security. Automated scanners have already checked for \
committed credentials, known-vulnerable dependencies and a list of dangerous \
constructs; their results are given to you. Find what they cannot.

Focus on:
- Trust boundaries: what crosses from user input into a sink, unvalidated.
- Authorisation applied at the wrong layer, or assumed by the caller.
- Secrets or tokens reaching logs, error messages, URLs or the client bundle.
- Resource exhaustion reachable by an unauthenticated request.
- Cryptography used in a way that does not do what the code assumes.

Rate honestly. A theoretical issue in code with no attacker-reachable path is \
low severity, and saying so is more useful than inflating it.
"""
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        report = self.run_gates(ctx)
        builder = self.builder(ctx)
        builder.add("Automated scanner results", report.render(failures_only=False), priority=P_FAILURE, max_tokens=3000)

        paths = ctx.spec.get("paths", []) or _security_relevant_paths(ctx.root)
        builder.add_files("Code under review", read_files(ctx.root, paths), priority=P_TASK_FILES, max_tokens=12000)

        result: dict[str, Any] = self.ask(
            ctx,
            builder,
            "Review this codebase for security problems the automated scanners "
            "cannot detect. Report only issues with a plausible path to exploitation.",
            schema=FINDINGS_SCHEMA,
        )
        records, nodes = self._process_findings(ctx, result, source=f"security:{ctx.node.id}")
        return AgentResult(
            success=True,
            summary=f"security review: {len(result.get('findings', []))} finding(s)",
            data={"verdict": result.get("verdict"), "summary": result.get("summary")},
            memory=records,
            nodes=nodes,
            report=report,
        )


def _security_relevant_paths(root: Path, limit: int = 20) -> list[str]:
    """Prefer files whose names suggest they handle input, auth or data."""
    interesting = ("auth", "login", "session", "api", "server", "route", "handler",
                   "db", "query", "user", "admin", "upload", "crypto", "token")
    scored: list[tuple[int, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".ts", ".tsx", ".js", ".py", ".go", ".rs"}:
            continue
        rel = path.relative_to(root)
        if any(part in {"node_modules", ".git", "dist", "build", ".forge"} for part in rel.parts):
            continue
        name = rel.as_posix().lower()
        score = sum(1 for keyword in interesting if keyword in name)
        scored.append((score, rel.as_posix()))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [path for _, path in scored[:limit]]


@register
class VisualReviewAgent(ReviewingAgent):
    """Judges captured screenshots.

    The one place a vision model is genuinely necessary, and deliberately the
    narrowest possible use of one: it runs only when the visual gate reports a
    change or when a node explicitly asks for a look, and it is given at most a
    handful of images. Sending every screenshot from every run to a vision model
    would dominate the cloud budget on its own.
    """

    kind = "visual_review"
    task_class = TaskClass.VISUAL_JUDGEMENT
    difficulty = 0.6
    stakes = 0.6

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You are looking at screenshots of the application being built and judging \
whether they show finished work.

Judge against what the project is trying to be. A tool can look plain and be \
finished; a game that looks plain is not.

Report specifically. "The UI needs polish" is useless. "The health bar overlaps \
the crosshair at this resolution" is actionable. If a screenshot shows an error \
state, a blank region, unstyled default text, or obviously placeholder content, \
that is a critical finding regardless of how the rest looks.
"""
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        images = self._images(ctx)
        if not images:
            return AgentResult(success=True, summary="no screenshots available to review")

        builder = self.builder(ctx)
        references = reference_images(ctx.root, limit=2)
        candidates = [Path(path) for path in images[: max(1, 4 - len(references))]]
        for path in [*references, *candidates]:
            try:
                label = "reference" if path in references else "candidate"
                builder.add_image(encode_image(path, label=f"{label}:{path.name}"))
            except OSError as exc:  # pragma: no cover
                ctx.logger().warn("could not read screenshot", path=str(path), error=str(exc))
        builder.add(
            "Images provided",
            "\n".join(
                [
                    *(f"REFERENCE: {path.name}" for path in references),
                    *(f"CANDIDATE: {path.name}" for path in candidates),
                ]
            ),
            priority=P_TASK_FILES,
            max_tokens=200,
        )

        result: dict[str, Any] = self.ask(
            ctx,
            builder,
            "Compare each CANDIDATE screenshot with the project goal and any "
            "REFERENCE images. Preserve originality: judge composition, hierarchy, "
            "density, palette, readability and finish rather than demanding copied "
            "pixels or branding. Report the largest concrete gaps.",
            schema=FINDINGS_SCHEMA,
            profile=self.profile(ctx, needs_vision=True),
        )
        records, nodes = self._process_findings(ctx, result, source=f"visual:{ctx.node.id}")

        # A clean visual review is the approval signal for the baselines the
        # visual gate captured, which is what makes the *next* run's pixel
        # comparison meaningful.
        approved = 0
        if result.get("verdict") == "approve":
            from ..validation.gates.visual import approve_baselines

            gate_ctx = ctx.gates.build_context(
                root=ctx.root, sandbox=ctx.sandbox, toolchain=ctx.toolchain, node_id=ctx.node.id
            )
            approved = approve_baselines(gate_ctx)

        return AgentResult(
            success=True,
            summary=f"visual review: {result.get('verdict')}, {len(result.get('findings', []))} finding(s)",
            data={"verdict": result.get("verdict"), "summary": result.get("summary"),
                  "baselines_approved": approved},
            memory=records,
            nodes=nodes,
            artifacts=[str(p) for p in images[:4]],
        )

    def _images(self, ctx: AgentContext) -> list[str]:
        explicit = ctx.spec.get("images", [])
        if explicit:
            return [p for p in explicit if Path(p).is_file()]
        candidates: list[Path] = []
        for node_id in [ctx.node.id, *ctx.node.deps]:
            directory = ctx.artifacts_dir / node_id
            if directory.is_dir():
                candidates.extend(sorted(directory.glob("*.png")))
        return [str(p) for p in candidates if not p.name.startswith("diff_")]


@register
class PerfReviewAgent(ReviewingAgent):
    """Reads performance measurements and proposes what to do about them."""

    kind = "perf"
    difficulty = 0.6
    stakes = 0.5

    def run(self, ctx: AgentContext) -> AgentResult:
        report = self.run_gates(ctx)
        measurements = {
            verdict.gate: {"score": verdict.score, "detail": verdict.detail}
            for verdict in report.verdicts
            if verdict.score is not None
        }
        if not measurements:
            return AgentResult(success=True, summary="no performance measurements available", report=report)

        builder = self.builder(ctx)
        builder.add(
            "Measurements",
            "\n".join(f"{gate}: {data}" for gate, data in measurements.items()),
            priority=P_FAILURE,
            max_tokens=2500,
        )
        result: dict[str, Any] = self.ask(
            ctx,
            builder,
            "These are the measured performance characteristics. Identify anything "
            "that will be a problem for this project's actual usage, and say what "
            "to change. Ignore numbers that are fine.",
            schema=FINDINGS_SCHEMA,
        )
        records, nodes = self._process_findings(ctx, result, source=f"perf:{ctx.node.id}")
        return AgentResult(
            success=True,
            summary=f"performance review: {len(result.get('findings', []))} finding(s)",
            data={"measurements": measurements, "verdict": result.get("verdict")},
            memory=records,
            nodes=nodes,
            report=report,
        )
