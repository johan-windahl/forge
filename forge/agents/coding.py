"""Agents that write code.

All of them share one loop: assemble focused context, get an edit plan, apply it
atomically, run gates, and iterate on failure within the node's own budget
before handing the failure back to the scheduler.

The inner loop is the important design decision. An implementation node does not
return "here is my code, hope it works" -- it validates its own work and fixes
what it can, because the alternative is a round trip through the scheduler for
every missing semicolon, each one paying full context-assembly cost. The loop is
bounded (``max_fix_rounds``) so a node that cannot converge escalates rather than
grinding.

Debugging is separated from implementation because the two need different
context and different models. Implementation needs interfaces and conventions;
debugging needs the failure output, the code around it, and usually a stronger
model -- which the router discovers on its own from the outcome statistics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..errors import ForgeError, PatchError
from ..execution.opencode import OpenCodeExecutor, OpenCodeResult
from ..kernel.events import Event, EventType
from ..memory.context import (
    P_ACCEPTANCE,
    P_FAILURE,
    P_INTERFACES,
    P_TASK_FILES,
    P_TREE,
    ContextBuilder,
    file_tree,
    read_files,
)
from ..memory.records import MemoryRecord
from ..memory.store import fact, interface
from ..models.structured import array, enum, integer, object_schema, string
from ..models.types import TaskClass, estimate_tokens
from ..obs.log import get_logger
from ..validation.gates.security import scan_text_for_secrets
from ..validation.types import ValidationReport
from ..workspace.git import Repo
from ..workspace.patch import EDIT_PLAN_SCHEMA, EditPlan, apply_edits
from .base import Agent, AgentContext, AgentResult, ProposedNode
from .registry import register

log = get_logger("agents.coding")

_EXPORTED_DECLARATION = re.compile(
    r"^[+-]\s*export\s+(?:default\s+)?(?:declare\s+)?"
    r"(?:abstract\s+)?(?:interface|type|class|function|const|let|var|enum|namespace)\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
_REMOVAL_INTENT = re.compile(
    r"\b(remove|delete|drop|rename|deprecat|replace\s+the\s+(?:public\s+)?api)\w*\b",
    re.IGNORECASE,
)
_DIAGNOSTIC_REPAIR = re.compile(
    r"(?:\bTS\d{3,5}\b|"
    r"\b(?:lint|typecheck|compilation|compiler|build)\s+(?:error|failure|violation)s?\b|"
    r"\b(?:browser|smoke)\s+(?:error|failure)s?\b|"
    r"\bsyntax\s+errors?\b|"
    r"\b(?:unused|missing)\s+(?:import|export|variable|symbol|type)s?\b|"
    r"\bimport\s+(?:path|extension)\s+errors?\b|"
    r"\b(?:eslint|parser)\s+errors?\b|"
    r"\btest\s+startup\b|"
    r"\bvitest\s+globals?\b)",
    re.IGNORECASE,
)
_AUDIT_TASK = re.compile(r"^\s*(?:audit|verify)\b", re.IGNORECASE)
_RUNTIME_DIAGNOSTIC = re.compile(
    r"\b(?:browser|smoke)\s+(?:error|failure)s?\b", re.IGNORECASE
)
_TRANSIENT_OPENCODE_ERROR = re.compile(
    r"(?:error\s*device\s*lost|device\s*lost|vk::queue::submit|"
    r"connection\s*(?:reset|closed|refused)|temporarily\s+unavailable|"
    r"service\s+unavailable)",
    re.IGNORECASE,
)
# Runtime and quality gates validate the integrated commit once. Running them
# inside every local repair round duplicates the expensive work and attributes
# whole-application failures to a narrow leaf before its dependencies exist.
_INTEGRATION_ONLY_GATES = frozenset(
    {
        "integration",
        "browser",
        "smoke",
        "visual",
        "bundle_size",
        "load_perf",
        "benchmark",
        "coverage",
        "deps",
        "dangerous_patterns",
    }
)

#: How many validate-and-fix rounds a coding node runs before giving up and
#: letting the scheduler decide (retry, escalate, or park).
#:
#: Five, not three, and the difference matters more than it looks. Rounds are
#: the *only* place incremental repair happens: they share one workspace, so
#: round three edits what round two wrote. An attempt is not incremental --
#: `restore_for_attempt` resets the tree, by design, so a new attempt rewrites
#: the file from scratch and meets a fresh set of errors.
#:
#: So a node that cannot converge within its rounds does not slowly improve
#: over ten attempts; it starts over ten times. Measured on the pinball run:
#: one node, 10 attempts, 19 gate failures, 12 cost units on sonnet and opus,
#: still failing on `src/main.ts(56,5): error TS18047: 'root' is possibly
#: 'null'` -- a two-line fix it had got within reach of and had thrown away
#: repeatedly. Two more rounds cost two model calls; an extra attempt costs
#: the whole file again.
MAX_FIX_ROUNDS = 5

#: How many times a round may ask to see more files before it has to commit to an
#: edit. Multiple rounds matter because a model reliably discovers another tier
#: of dependencies only after reading the first. Observed live -- opus asked for vec2/loop/
#: architecture, and having read them asked for types/tuning/tsconfig, which were
#: grantable and were refused. Each refused round still costs a full-context call
#: on a frontier rung, so the limit buys nothing by being tight; what stops a
#: loop is that a spent budget ends the asking, not that it ends it sooner.
#:
#: Context selection is a guess made before the model has read anything. When the
#: guess is wrong the model has no way to say so: it writes against an interface
#: it is inferring, and the result compiles against an imagined API. Observed on
#: the pinball run, in the `summary` field of an edit plan, because there was
#: nowhere else to put it -- "Implementing src/engine/collide.ts ... Need to see
#: tuning.ts first for physics constants." It then wrote 500 lines blind, calling
#: a `v()` helper that did not exist and referencing types that were never
#: defined. Three fix rounds could not recover, and the node blocked.
#:
#: A granted request does not consume a fix round: asking for the right file is
#: what makes the following round useful, so charging for it would defeat it.
# Six accommodates integration work that discovers dependencies in layers. The
# live screenshot task needed a fifth request for the high-score interface only
# after applying a first draft and seeing type errors; refusing that one file
# threw away the value of the preceding local work. This is a request-round
# limit, while ``MAX_REQUESTED_FILES`` separately caps prompt growth per round.
MAX_FILE_REQUESTS = 6

#: Ceiling on one request, so a model cannot ask for the whole repository and
#: push the actual task out of the context window.
MAX_REQUESTED_FILES = 8

# OpenCode already supplies repository tools and their schemas, and its agents
# can discover source on demand. Feeding it the full native-model context budget
# duplicates that source, leaves too little room for tool results and output,
# and can overflow a local server whose usable window is smaller than the model
# metadata claims. Keep the durable project/acceptance context, but let the
# scout fetch implementation detail with tools.
OPENCODE_PROMPT_BUDGET = 12_000

#: Manifests and build configuration, pinned into every coding node's context.
#: Small, stable, and the answer to a whole class of file request that would
#: otherwise cost a full round-trip each. Only those that exist are included.
_PROJECT_FILES = (
    "package.json",
    "tsconfig.json",
    "vite.config.ts",
    "vitest.config.ts",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
)

#: How many times a model may ask to re-read a file this attempt wrote itself.
#: Free of the request budget, because re-reading its own output is continuity
#: rather than discovery -- but not free of a limit, because serving one adds
#: nothing to the prompt that was not already there, so the ask can repeat
#: forever without a counter.
MAX_REREADS = 2

#: Budget for a digest. Generous, because the whole point is that the raw
#: compiler output goes *here* rather than into the repair prompt.
DIGEST_BUDGET_TOKENS = 20_000

#: Budget for a consult: the failing evidence plus the file it points at, and
#: nothing else. A tenth of a normal request, because the whole point is that the
#: expensive rung reads a little and writes a little.
CONSULT_BUDGET_TOKENS = 6_000

#: Which rung answers a consult. The cheapest cloud rung, not the strongest: the
#: task is "read this compiler error and say what is wrong", which haiku does
#: perfectly well, and pinning it here is what stops a diagnosis from costing
#: opus prices.
CONSULT_RUNG = "haiku"

MAX_DECOMPOSITION_DEPTH = 3

DECOMPOSITION_SCHEMA = object_schema(
    {
        "reason": string("Why the original task is too broad for one reliable edit"),
        "tasks": array(
            object_schema(
                {
                    "title": string("Short imperative child-task title"),
                    "objective": string("One independently completable objective"),
                    "acceptance": array(string(), "Deterministic completion criteria", minItems=1),
                    "paths": array(string(), "Files this child should own"),
                    "deps": array(integer(), "Earlier child indices this child depends on"),
                },
                required=["title", "objective", "acceptance", "paths"],
            ),
            "Smaller implementation leaves",
            minItems=2,
            maxItems=8,
        ),
    },
    required=["reason", "tasks"],
)

ADVICE_SCHEMA = object_schema(
    {
        "diagnosis": string("The root cause, not a restatement of the error"),
        "evidence": array(string(), "Concrete evidence supporting the diagnosis"),
        "instructions": array(
            string(), "Ordered changes the local engineer should make", minItems=1
        ),
        "expected_result": string("What the failing check should report after the repair"),
        "alternative": string("Different approach if the primary repair fails"),
        "confidence": enum(["low", "medium", "high"], "Confidence in the diagnosis"),
    },
    required=["diagnosis", "instructions", "expected_result", "confidence"],
)

#: An acceptance criterion that names a test, and what a test file looks like.
#: Deliberately narrow: this only fires when the criteria say so outright, so it
#: cannot argue with a node that was never asked to write tests.
_MENTIONS_TEST = re.compile(r"\b(test|asserted by test|unit test|soak test)\b", re.I)
_IS_TEST_FILE = re.compile(r"(^|/)(tests?|spec)/|\.(test|spec)\.[jt]sx?$|_test\.py$|test_.*\.py$")


class CodingAgent(Agent):
    """Shared machinery for anything that produces an edit plan."""

    commits = True
    task_class = TaskClass.IMPLEMENTATION

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE

        return (
            SHARED_PREAMBLE
            + """
You write code by returning an edit plan: a list of file operations.

Choosing an operation:
- `write` replaces a file entirely. Use it for new files, and for existing \
files under roughly 150 lines where you are rewriting most of it.
- `replace` swaps an exact snippet for new text. Use it for targeted changes in \
larger files. The `anchor` must be copied character-for-character from the file \
content you were shown, and must be long enough to appear exactly once.
- `insert_after` adds text following an anchor.
- `delete` removes a file.

Get this right, because it is the most common failure:
- Anchors must be unique. If a short anchor appears several times, include \
enough surrounding lines to disambiguate it.
- Never use `replace` on a file you have not been shown. Ask for it by \
including it in the plan as a `write` only if you are creating it.
- Write complete, working code. No `TODO`, no `...`, no "implementation left as \
an exercise". Code that does not run fails the gates and costs another round.

If you have not been shown a file whose exact contents you need -- the constants, \
the type definitions, the helper signatures you are about to call -- put its path \
in `need_files` and send an **empty** `edits` list. You are asked again with those \
files included, at no cost to your retry budget. Do not write a placeholder: an \
empty edit list is the correct and expected answer when you are asking. Ask for \
everything you need at once -- you get two requests, and each one costs a full \
round trip. Inferring \
an interface and being wrong costs far more -- the code compiles against an API \
that does not exist, and the gates report it as a dozen unrelated errors.
"""
        )

    # -- the core loop ---------------------------------------------------

    def implement(
        self,
        ctx: AgentContext,
        task: str,
        *,
        extra_sections: dict[str, tuple[str, int]] | None = None,
        include_paths: list[str] | None = None,
    ) -> AgentResult:
        """Produce, apply and validate an edit plan, fixing what it can."""
        satisfied = self._already_satisfied_decomposition(ctx)
        if satisfied is not None:
            return satisfied
        satisfied = self._already_satisfied_repair(ctx, task)
        if satisfied is not None:
            return satisfied

        opencode_result = self._maybe_implement_with_opencode(
            ctx,
            task,
            extra_sections=extra_sections,
            include_paths=include_paths,
        )
        if opencode_result is not None:
            return opencode_result

        paths = include_paths if include_paths is not None else self._relevant_paths(ctx)
        applied_files: list[str] = []
        summaries: list[str] = []
        report: ValidationReport | None = None
        granted = 0
        rereads = 0
        round_index = -1
        pinned: list[str] = []
        # Both describe `report` specifically, so both are cleared whenever
        # `report` is replaced by a different failure below. Showing round one's
        # digest of a type error under a round-two patch rejection is worse than
        # showing nothing: the model is asked to fix a failure that is no longer
        # happening and never sees the one that is.
        advice = self._previous_advice(ctx)
        consulted = False
        digest = ""

        while round_index + 1 < MAX_FIX_ROUNDS:
            round_index += 1
            builder = self.builder(ctx)
            # Pinned files are rendered in their own section below; including them
            # here too would spend the context budget printing them twice.
            files = read_files(ctx.root, [p for p in paths if p not in set(pinned)])
            builder.add_files("Files you are working on", files, priority=P_TASK_FILES, max_tokens=9000)
            if pinned:
                # Its own section, ahead of the general one. A file the model
                # asked for by name must not be the file the context budget
                # drops: it asked precisely because it could not see it, and
                # answering by silently including it among 9000 tokens of other
                # files is how the request came back a second time.
                builder.add_files(
                    "Files you asked to see",
                    read_files(ctx.root, pinned),
                    priority=P_FAILURE,
                    # A share of the real budget, not a constant. 12000 was
                    # hardcoded against a budget that is 22400 here: together
                    # with the 9000 for general files that is 94% of the context
                    # spent on file bodies, leaving the goal and the acceptance
                    # criteria to be squeezed out on any large request.
                    max_tokens=max(2000, self.context_budget(ctx) // 2),
                )
            tree = file_tree(ctx.root, limit=200)
            if tree:
                builder.add("Repository layout", tree, priority=P_TREE, max_tokens=1200)
            for name, (content, priority) in (extra_sections or {}).items():
                builder.add(name, content, priority=priority, max_tokens=3000)

            if report is not None and not report.passed:
                # The digest, when we have one: the raw output was already read
                # in a separate local call, so what lands here is signal rather
                # than four hundred lines of cascade competing for the budget.
                builder.add(
                    "What the checks reported"
                    if digest
                    else "Validation failures from your previous attempt",
                    digest or report.render(),
                    priority=P_FAILURE,
                    max_tokens=6000,
                    tail_lines=120,
                )
                if advice:
                    builder.add(
                        "A senior engineer's diagnosis of that failure",
                        advice,
                        priority=P_INTERFACES,
                        max_tokens=1500,
                    )
                round_task = (
                    "Your previous edit did not pass validation. Fix the failures "
                    "listed above. Change only what is needed to make them pass; "
                    "do not rewrite working code.\n\n"
                    f"The original task was:\n{task}"
                )
            else:
                round_task = task

            profile = self.profile(ctx, attempt=round_index)
            selected = ctx.config.models.models.get(ctx.node.tier)
            if selected is not None and selected.hosted == "cloud":
                # Direct cloud authorship happens only after the scheduler has
                # explicitly exhausted the local/decompose/coach stages.
                profile.min_tier = ctx.node.tier
            else:
                # Failed local rounds may move from fast to deep, but they do not
                # silently turn into cloud-authored code.
                profile.max_tier = "mid"
                if ctx.node.tier == "local_deep":
                    profile.min_tier = "local_deep"
            payload = self.ask(ctx, builder, round_task, schema=EDIT_PLAN_SCHEMA, profile=profile)
            plan = EditPlan.from_payload(payload)

            # A plan that asks for files has declared its own edits provisional,
            # and that declaration is authoritative. Applying them anyway is how a
            # node "succeeded" by committing the single line
            # "// Placeholder - will implement after reading existing files":
            # haiku asked for three files, wrote a throwaway it expected to be
            # discarded, and the throwaway became the deliverable because nothing
            # imports the file yet so every gate passed. A false success is worse
            # than a failure -- the graph moves on and builds on a stub.
            if plan.need_files:
                wanted = self._grant_files(ctx, plan.need_files)
                # Re-reading a file this attempt wrote is continuity, not
                # discovery, and must not cost a discovery grant. A test_author
                # node spent both grants exploring, then in a repair round asked
                # to see the `tests/helpers/sim.ts` it had just written itself.
                # It was refused, its repair was discarded as provisional, the
                # round was consumed, and the attempt failed and escalated --
                # throwing away six files of work over a file already on disk.
                own = [path for path in wanted if path in set(applied_files)]
                fresh = [path for path in wanted if path not in set(applied_files)]
                # Capped, unlike the grant below it, which is capped by
                # `granted`. Serving a re-read changes nothing the model can
                # see -- the file is already pinned and already on disk -- so a
                # model that asks a second time gets a byte-identical prompt and
                # asks again. Unbounded, that is a full-context call per turn
                # forever, with the round counter rolled back each time so the
                # node never fails either.
                if own and not fresh:
                    pinned = sorted(set(pinned) | set(own))
                    paths = sorted(set(paths) | set(own))
                    if rereads < MAX_REREADS:
                        rereads += 1
                        round_index -= 1
                        ctx.logger().info(
                            "model asked to re-read its own output", files=own, count=rereads
                        )
                        continue
                    # Past the cap the ask has to cost a round, and must not be
                    # allowed to fall through and take a *discovery* grant
                    # instead: these are the model's own files, it already has
                    # them, and spending a grant to pin what is already pinned
                    # produces the same prompt and the same ask one more time.
                    ctx.logger().warn(
                        "model kept asking to re-read its own output", files=own
                    )
                    report = _synthetic_report(
                        "context",
                        "You wrote " + ", ".join(own) + " during this attempt and their "
                        "full current contents are in the section titled 'Files you "
                        "asked to see'. Nothing further will be provided. Implement "
                        "with what you have.",
                    )
                    digest = advice = ""
                    continue
                if wanted and granted < MAX_FILE_REQUESTS:
                    pinned = sorted(set(pinned) | set(wanted))
                    paths = sorted(set(paths) | set(wanted))
                    granted += 1
                    round_index -= 1  # asking is not a failed attempt
                    ctx.logger().info(
                        "model asked to read more files before editing",
                        files=wanted,
                        discarded_edits=len(plan.edits),
                    )
                    continue
                # Out of requests but carrying real edits: apply them. The
                # provisional rule exists for the model that writes a deliberate
                # throwaway alongside its request, and an empty `edits` list is
                # now the correct way to ask -- so a plan with actual edits *and*
                # a request is a model doing its best with what it has. Observed:
                # a node was handed a 1489-character diagnosis, asked for one more
                # file, was refused, had its edits discarded, and escalated. The
                # advice was bought and then thrown away.
                # `granted` is the discriminator, not `plan.edits` alone. A model
                # that has spent its requests has been given what it asked for and
                # is now working with it -- keep that work. A model whose request
                # was never grantable at all is asking for files that do not
                # exist, which is the same confusion that produced the committed
                # placeholder; discard those edits as before.
                if plan.edits and granted >= MAX_FILE_REQUESTS:
                    ctx.logger().info(
                        "no file requests left; keeping the edits it did make",
                        requested=plan.need_files[:6],
                        edits=len(plan.edits),
                    )
                else:
                    # A request and nothing else. Say which case it is: claiming
                    # "everything is already included" when the files were merely
                    # withheld is a lie the model cannot act on, and it answers by
                    # asking again.
                    ctx.logger().warn(
                        "no file requests left and no edits to keep",
                        requested=plan.need_files[:8],
                        grantable=wanted,
                        already_granted=granted,
                    )
                    if wanted:
                        have = sorted(set(pinned) | {p for p in paths if p in set(wanted)})
                        detail = "You already have the full contents of: " + (
                            ", ".join(have) if have else "(nothing yet)"
                        )
                        detail += (
                            ". They are in the section titled 'Files you asked to see'. "
                            "You have used all " + str(MAX_FILE_REQUESTS) + " file "
                            "requests, so nothing further will be provided. Implement "
                            "now using these, and say what you had to assume."
                        )
                    else:
                        detail = (
                            "None of " + ", ".join(plan.need_files[:8]) + " exists in "
                            "this workspace. Check the repository layout for real paths."
                        )
                    report = _synthetic_report("context", detail + " Do not ask again.")
                    digest = advice = ""
                    continue

            # An empty plan that is not a file request is a non-answer. It has to
            # be caught here: `apply_edits` accepts it happily, the gates then
            # pass because nothing changed, and the node is marked succeeded
            # having done nothing at all. That is the same false success as the
            # committed placeholder, arrived at from the other direction.
            if not plan.edits:
                ctx.logger().warn("model returned no edits", summary=plan.summary[:120])
                report = _synthetic_report(
                    "empty",
                    "Your response contained no edits and did not ask for any files. "
                    "Return the actual file operations that implement the task.",
                )
                digest = advice = ""
                continue

            leak = self._check_secrets(plan)
            if leak:
                return AgentResult.failure(
                    f"refused to apply an edit containing what looks like a credential ({leak})"
                )

            try:
                applied = apply_edits(ctx.root, plan)
            except PatchError as exc:
                ctx.logger().warn("edit plan rejected", error=str(exc))
                if round_index == MAX_FIX_ROUNDS - 1:
                    return AgentResult.failure(
                        f"could not apply the edit plan: {exc}", needs_escalation=True
                    )
                # Feed the specific rejection back rather than re-asking blind.
                report = _synthetic_report("patch", str(exc))
                digest = advice = ""
                continue

            applied_files = sorted(set(applied_files) | set(applied.written) | set(applied.deleted))
            summaries.append(plan.summary or "code change")
            paths = sorted(set(paths) | set(applied.written))

            violation = self._semantic_scope_violation(ctx, task)
            if violation:
                report = _synthetic_report("scope", violation)
                digest = advice = ""
                continue

            report = self.run_gates(
                ctx,
                changed_files=applied_files,
                fail_fast=True,
                gate_names=self._coding_gate_names(ctx),
            )
            if report.passed:
                missing = self._unmet_test_requirement(ctx, applied_files)
                if missing and round_index + 1 < MAX_FIX_ROUNDS:
                    ctx.logger().warn(
                        "acceptance criteria require tests that were not written",
                        files=applied_files,
                    )
                    report = _synthetic_report("acceptance", missing)
                    digest = advice = ""
                    continue
                return AgentResult(
                    success=True,
                    summary=plan.summary or f"changed {len(applied_files)} file(s)",
                    data={"rounds": round_index + 1, "plan_summary": plan.summary},
                    changed_files=applied_files,
                    commit_message=self._commit_message(ctx, plan.summary),
                    report=report,
                    memory=[
                        *self._observations(ctx, report),
                        *self._advice_memory(ctx, advice),
                    ],
                )

            blocking = [v for v in report.failures if v.gate not in _NON_BLOCKING]
            if not blocking:
                # Only advisory gates failed. Record them as findings for later
                # rather than burning rounds on them now.
                return AgentResult(
                    success=True,
                    summary=(plan.summary or "code change") + " (advisory gate warnings)",
                    changed_files=applied_files,
                    commit_message=self._commit_message(ctx, plan.summary),
                    report=report,
                    memory=[
                        *self._findings(ctx, report),
                        *self._advice_memory(ctx, advice),
                    ],
                )
            ctx.logger().info(
                "validation failed, retrying inside the node",
                round=round_index + 1,
                failures=[v.gate for v in report.failures],
            )
            # Compress the output locally first, then buy a diagnosis of what is
            # left. The digest is free and shrinks the problem; the consult costs
            # a little and finds the fault. Both only while rounds remain to use
            # them.
            if round_index + 1 < MAX_FIX_ROUNDS:
                digest = self._digest_failures(ctx, report)
                if not consulted:
                    advice = self._consult(
                        ctx, task, report, applied_files, previous=advice
                    ) or advice
                    consulted = True

        decomposition = self._decompose(ctx, task, report)
        if decomposition:
            return AgentResult(
                success=False,
                summary=f"split an oversized failing task into {len(decomposition)} local leaves",
                data={"decomposed": True},
                nodes=decomposition,
                changed_files=applied_files,
                report=report,
                memory=self._advice_memory(ctx, advice),
            )

        return AgentResult(
            success=False,
            summary=f"validation still failing after {MAX_FIX_ROUNDS} attempt(s): "
            + ", ".join(v.gate for v in (report.failures if report else [])),
            data={"preserve_progress": True},
            changed_files=applied_files,
            report=report,
            needs_escalation=True,
            memory=[
                *(self._findings(ctx, report) if report else []),
                *self._advice_memory(ctx, advice),
            ],
        )

    # -- helpers ---------------------------------------------------------

    def _maybe_implement_with_opencode(
        self,
        ctx: AgentContext,
        task: str,
        *,
        extra_sections: dict[str, tuple[str, int]] | None,
        include_paths: list[str] | None,
    ) -> AgentResult | None:
        """Use OpenCode for local authorship, or return ``None`` for native."""
        coding = ctx.config.coding
        if coding.backend == "native":
            return None

        model_name = (
            ctx.node.tier
            if ctx.node.tier in ctx.config.models.models
            else ctx.config.models.default
        )
        spec = ctx.config.models.models[model_name]
        if spec.hosted != "local":
            # The final cloud-solver path remains Forge-native so every cloud
            # call passes through ModelClient's hard admission check.
            return None
        executor = OpenCodeExecutor(
            ctx.config,
            ctx.sandbox,
            node_id=ctx.node.id,
            model_name=model_name,
            attempt=ctx.node.attempts,
        )
        if not executor.available():
            if coding.backend == "opencode" and not coding.fallback_to_native:
                return AgentResult.failure(
                    f"OpenCode executable {coding.opencode_command!r} is unavailable",
                    needs_human=(
                        "Install OpenCode in the configured sandbox or set "
                        "[coding].fallback_to_native = true."
                    ),
                )
            ctx.logger().info(
                "OpenCode unavailable; using Forge's native coding loop",
                command=coding.opencode_command,
            )
            return None

        return self._implement_with_opencode(
            ctx,
            task,
            executor,
            extra_sections=extra_sections,
            include_paths=include_paths,
        )

    def _implement_with_opencode(
        self,
        ctx: AgentContext,
        task: str,
        executor: OpenCodeExecutor,
        *,
        extra_sections: dict[str, tuple[str, int]] | None,
        include_paths: list[str] | None,
    ) -> AgentResult | None:
        """Let a local OpenCode session edit the node worktree, then gate it."""
        report: ValidationReport | None = None
        advice = self._previous_advice(ctx)
        consulted = False
        last_run: OpenCodeResult | None = None
        changed: list[str] = []

        for round_index in range(ctx.config.coding.opencode_rounds):
            if round_index > 0:
                # The next prompt contains the failed gate evidence and any
                # stronger-model diagnosis. Start it compactly from the
                # preserved worktree instead of replaying a large OpenCode
                # transcript that already led to a failed validation.
                executor.start_fresh_session()
            prompt = self._opencode_prompt(
                ctx,
                task,
                report=report,
                advice=advice,
                round_index=round_index,
                extra_sections=extra_sections,
                include_paths=include_paths,
            )
            self._emit_opencode_request(ctx, executor.model_name, round_index)
            spec = ctx.config.models.models[executor.model_name]
            estimated_cost = spec.cost(
                estimate_tokens(prompt) * min(ctx.config.coding.opencode_steps, 4),
                min(spec.max_output_tokens, 32_000),
            )
            ctx.models.budget.check_and_reserve(
                estimated_cost,
                hosted="local",
                node_id=ctx.node.id,
                escalation=ctx.node.attempts > 1,
            )
            try:
                run = executor.execute(prompt)
                last_run = run
                # Inside the reservation: see the note in ModelClient.complete.
                self._record_opencode_run(ctx, executor.model_name, run, round_index)
            finally:
                ctx.models.budget.release(
                    estimated_cost, hosted="local", node_id=ctx.node.id
                )
            round_changed = self._worktree_changes(ctx)
            changed = sorted(set(changed) | set(round_changed))

            if not run.ok and not changed:
                if (
                    _TRANSIENT_OPENCODE_ERROR.search(run.error)
                    and round_index + 1 < ctx.config.coding.opencode_rounds
                ):
                    ctx.logger().warn(
                        "transient OpenCode provider failure; retrying tool-driven loop",
                        error=run.error[:300],
                        round=round_index + 1,
                    )
                    report = _synthetic_report(
                        "provider",
                        "The local inference backend failed transiently. Resume the task "
                        "from the current worktree and inspect it again before editing.",
                    )
                    continue
                if ctx.config.coding.fallback_to_native:
                    ctx.logger().warn(
                        "OpenCode failed before changing files; using native loop",
                        error=run.error[:300],
                    )
                    return None
                return AgentResult.failure(
                    f"OpenCode could not execute the task: {run.error or 'unknown error'}",
                    needs_escalation=True,
                )

            if not round_changed:
                verified = self._verify_unchanged_diagnostic_repair(ctx, task)
                unchanged_kind = "diagnostic repair"
                if verified is None:
                    verified = self._verify_unchanged_audit(ctx)
                    unchanged_kind = "audit"
                if verified is not None and verified.passed:
                    summary = (
                        f"The requested {unchanged_kind} is already satisfied on "
                        "the current integrated tree"
                    )
                    return AgentResult(
                        success=True,
                        summary=summary,
                        data={
                            "backend": "opencode",
                            "already_satisfied": True,
                            "rounds": round_index + 1,
                            "session_id": run.session_id,
                        },
                        changed_files=[],
                        report=verified,
                        memory=self._observations(ctx, verified),
                    )
                report = verified or _synthetic_report(
                    "empty",
                    "OpenCode's latest round left no uncommitted implementation change. "
                    "Inspect the task again and implement it in the worktree; a prose "
                    "answer or a change reverted back to the baseline is not a deliverable.",
                )
                continue

            violation = self._semantic_scope_violation(ctx, task)
            if violation:
                report = _synthetic_report("scope", violation)
                continue

            report = self.run_gates(
                ctx,
                changed_files=changed,
                fail_fast=True,
                gate_names=self._coding_gate_names(ctx),
            )
            if report.passed:
                missing = self._unmet_test_requirement(ctx, changed)
                if missing and round_index + 1 < ctx.config.coding.opencode_rounds:
                    report = _synthetic_report("acceptance", missing)
                    continue
                summary = self._opencode_summary(run, changed)
                return AgentResult(
                    success=True,
                    summary=summary,
                    data={
                        "backend": "opencode",
                        "rounds": round_index + 1,
                        "session_id": run.session_id,
                        "usage_measured": run.usage.measured,
                    },
                    changed_files=changed,
                    commit_message=self._commit_message(ctx, summary),
                    report=report,
                    memory=[
                        *self._observations(ctx, report),
                        *self._advice_memory(ctx, advice),
                    ],
                )

            blocking = [failure for failure in report.failures if failure.gate not in _NON_BLOCKING]
            if not blocking:
                summary = self._opencode_summary(run, changed) + " (advisory gate warnings)"
                return AgentResult(
                    success=True,
                    summary=summary,
                    data={
                        "backend": "opencode",
                        "rounds": round_index + 1,
                        "session_id": run.session_id,
                    },
                    changed_files=changed,
                    commit_message=self._commit_message(ctx, summary),
                    report=report,
                    memory=[
                        *self._findings(ctx, report),
                        *self._advice_memory(ctx, advice),
                    ],
                )

            if round_index + 1 < ctx.config.coding.opencode_rounds and not consulted:
                advice = self._consult(ctx, task, report, changed, previous=advice) or advice
                consulted = True

        decomposition = self._decompose(ctx, task, report)
        if decomposition:
            return AgentResult(
                success=False,
                summary=f"split an oversized failing task into {len(decomposition)} local leaves",
                data={
                    "backend": "opencode",
                    "decomposed": True,
                    "session_id": last_run.session_id if last_run else "",
                },
                nodes=decomposition,
                changed_files=changed,
                report=report,
                memory=self._advice_memory(ctx, advice),
            )

        failures = "; ".join(
            f"{failure.gate}: {failure.summary or failure.evidence[:240]}"
            for failure in (report.failures if report else [])
        )
        if not failures and last_run and last_run.error:
            failures = last_run.error[:500]
        return AgentResult(
            success=False,
            summary=(
                f"OpenCode validation still failing after "
                f"{ctx.config.coding.opencode_rounds} round(s)"
                + (f": {failures}" if failures else "")
            ),
            data={
                "backend": "opencode",
                "preserve_progress": True,
                "session_id": last_run.session_id if last_run else "",
            },
            changed_files=changed,
            report=report,
            needs_escalation=True,
            memory=[
                *(self._findings(ctx, report) if report else []),
                *self._advice_memory(ctx, advice),
            ],
        )

    def _opencode_prompt(
        self,
        ctx: AgentContext,
        task: str,
        *,
        report: ValidationReport | None,
        advice: str,
        round_index: int,
        extra_sections: dict[str, tuple[str, int]] | None,
        include_paths: list[str] | None,
    ) -> str:
        builder = self.builder(ctx)
        builder.budget = min(builder.budget, OPENCODE_PROMPT_BUDGET)
        for name, (content, priority) in (extra_sections or {}).items():
            builder.add(name, content, priority=priority, max_tokens=6000)
        if include_paths:
            builder.add(
                "Likely files; inspect their current contents with tools",
                "\n".join(f"- {path}" for path in include_paths),
                priority=P_TASK_FILES,
                max_tokens=800,
            )
        if report is not None and not report.passed:
            builder.add(
                "Independent Forge validation from the previous OpenCode round",
                report.render(),
                priority=P_FAILURE,
                max_tokens=6000,
                tail_lines=120,
            )
        if advice:
            builder.add(
                "Senior cloud coach advice; follow it while doing the work locally",
                advice,
                priority=P_INTERFACES,
                max_tokens=1800,
            )
        round_task = task
        if round_index:
            round_task = (
                "Continue in the same worktree and session. Preserve correct existing "
                "work, fix every Forge validation failure shown above, and run focused "
                "checks before returning.\n\nOriginal task:\n" + task
            )
        messages = builder.build(
            system_prompt=(
                "You are Forge's local coding executor. Work directly in the current "
                "worktree using OpenCode tools. Inspect files as needed, implement the "
                "task completely, and run focused checks. Do not commit or switch "
                "branches. Return only after the worktree contains the implementation."
            ),
            task=round_task,
        )
        return "\n\n".join(
            f"[{message.role.upper()}]\n{message.content}" for message in messages
        )

    def _emit_opencode_request(
        self, ctx: AgentContext, model_name: str, round_index: int
    ) -> None:
        spec = ctx.config.models.models[model_name]
        ctx.models.ledger.append(
            Event(
                type=EventType.MODEL_REQUEST,
                node_id=ctx.node.id,
                payload={
                    "route": {
                        "model": model_name,
                        "tier": spec.tier,
                        "reason": "OpenCode local execution backend",
                    },
                    "task_class": str(self.task_class),
                    "label": f"opencode:{ctx.node.id[-8:]}",
                    "backend": "opencode",
                    "round": round_index + 1,
                },
            )
        )

    def _record_opencode_run(
        self,
        ctx: AgentContext,
        model_name: str,
        run: OpenCodeResult,
        round_index: int,
    ) -> None:
        spec = ctx.config.models.models[model_name]
        usage = run.usage
        generated = usage.generated_tokens
        if usage.input_tokens or generated:
            ctx.models.budget.record(
                model=model_name,
                tier=spec.tier,
                hosted="local",
                cost=spec.cost(usage.input_tokens, generated),
                input_tokens=usage.input_tokens,
                output_tokens=generated,
                cached_tokens=usage.cached_tokens,
                node_id=ctx.node.id,
                task_class=str(self.task_class),
                escalation=ctx.node.attempts > 1,
            )
        payload: dict[str, Any] = {
            "model": model_name,
            "tier": spec.tier,
            "backend": "opencode",
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": generated,
                "cached_input_tokens": usage.cached_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "steps": usage.steps,
                "measured": usage.measured,
            },
            "latency": round(run.duration, 3),
            "finish_reason": "stop" if run.ok else "error",
            "cost": round(spec.cost(usage.input_tokens, generated), 6),
            "session_id": run.session_id,
            "round": round_index + 1,
        }
        limit = ctx.config.memory.transcript_max_chars
        if ctx.config.memory.keep_transcripts and limit > 0 and run.summary:
            payload["text"] = run.summary[:limit]
            payload["text_truncated"] = len(run.summary) > limit
        ctx.models.ledger.append(
            Event(type=EventType.MODEL_RESPONSE, node_id=ctx.node.id, payload=payload)
        )
        if not run.ok:
            ctx.models.ledger.append(
                Event(
                    type=EventType.MODEL_ERROR,
                    node_id=ctx.node.id,
                    payload={
                        "model": model_name,
                        "backend": "opencode",
                        "error": run.error[:4000],
                        "error_type": "OpenCodeExecutionError",
                        "retryable": True,
                    },
                )
            )

    @staticmethod
    def _worktree_changes(ctx: AgentContext) -> list[str]:
        changed: list[str] = []
        # Failed integrated validation resets main but preserves the node
        # branch. On retry that implementation is committed and the worktree
        # is clean; status alone would call it empty and ask the model to redo
        # already-correct work.
        if ctx.repo.path != ctx.config.workspace_dir:
            main = Repo(ctx.config.workspace_dir)
            if main.has_commits() and ctx.repo.has_commits():
                changed.extend(ctx.repo.changed_files(since=main.head()))
        for _status, path in ctx.repo.status():
            # Git renders a rename as ``old -> new``. Gates and commit metadata
            # need the path that exists after the edit.
            changed.append(path.rsplit(" -> ", 1)[-1])
        return sorted(dict.fromkeys(changed))

    @staticmethod
    def _opencode_summary(run: OpenCodeResult, changed: list[str]) -> str:
        summary = " ".join(run.summary.split())
        if summary:
            return summary[:240]
        return f"implemented changes in {len(changed)} file(s) with OpenCode"

    def _unmet_test_requirement(self, ctx: AgentContext, changed: list[str]) -> str:
        """Empty unless the node was told to prove itself with tests and did not.

        The gates measure generic health -- does it lint, does it compile, do the
        *existing* tests still pass -- and nothing measures whether the node did
        what it was asked. A plunger node whose acceptance said "asserted by
        test" three times over shipped 52 lines and no test at all, passed every
        gate, and was marked succeeded having implemented perhaps a third of its
        criteria. The graph then builds on it.

        This is the narrow, deterministic part of that problem: when the criteria
        name tests explicitly, an attempt that adds none has not met them, and no
        model call is needed to know it.
        """
        acceptance = list(getattr(ctx.node, "acceptance", None) or [])
        if not acceptance:
            return ""
        demanded = [item for item in acceptance if _MENTIONS_TEST.search(item)]
        if not demanded:
            return ""
        if any(_IS_TEST_FILE.search(path) for path in changed):
            return ""
        return (
            "Every gate passed, but these acceptance criteria call for tests and "
            "this change adds none:\n"
            + "\n".join(f"- {item}" for item in demanded[:4])
            + "\nAdd the tests that demonstrate them. Keep the implementation you "
            "already have; add test files alongside it."
        )

    def _digest_failures(self, ctx: AgentContext, report: ValidationReport) -> str:
        """Compress raw gate output into the few lines that carry signal.

        Run on a *local* rung, in its own call, so the four hundred lines of
        `tsc` cascade never enter the repair prompt at all. The alternative is
        what was happening: truncate the output and hope the surviving half
        contains the cause. Head-and-tail retention is a guess; asking a model to
        read all of it and say what matters is not, and locally it is free.

        This is a summarisation task, which is precisely what the small rung is
        good at -- the thing it struggles with is authoring five hundred lines of
        physics, not reading a compiler.

        Advisory: any failure returns "" and the caller falls back to the raw,
        truncated evidence. A digest must never be able to fail a node.
        """
        raw = report.render()
        if len(raw) < 1200:
            return ""  # short enough to read directly; a call would be waste
        try:
            builder = ContextBuilder(budget_tokens=DIGEST_BUDGET_TOKENS)
            builder.add("Raw output from the failing checks", raw, priority=P_FAILURE)
            digest = self.ask(
                ctx,
                builder,
                "Summarise these check failures for the engineer who must fix them.\n\n"
                "List each distinct problem once, as `path:line - what is wrong`, "
                "root causes first. A compiler lists errors in file order, so the "
                "earliest ones usually explain the later ones -- say which are "
                "consequences rather than repeating them.\n\n"
                "Do not suggest fixes and do not write code. Be terse: this replaces "
                "the raw output, so anything you omit the engineer will not see.",
                profile=self.profile(
                    ctx,
                    max_tier="mid",  # never a cloud rung: this must stay free
                    difficulty=0.2,
                    stakes=0.2,
                    label=f"digest:{ctx.node.id[-8:]}",
                ),
            )
            text = str(digest or "").strip()
        except ForgeError as exc:
            ctx.logger().warn("digest failed; using raw output", error=str(exc))
            return ""
        if text:
            ctx.logger().info(
                "digested gate output locally", raw_chars=len(raw), digest_chars=len(text)
            )
        return text[:6000]

    def _consult(
        self,
        ctx: AgentContext,
        task: str,
        report: ValidationReport,
        changed: list[str],
        *,
        previous: str = "",
    ) -> str:
        """Ask a stronger model what is wrong, without asking it to fix anything.

        The local rung can apply a precise instruction; what it cannot reliably do
        is *find* the fault. It failed to locate a single unbalanced brace across
        three repair rounds, and every node that landed today was written by a
        cloud model rather than repaired by one.

        So this buys the diagnosis, not the code. Output is a few hundred words
        against a 6k context, which is roughly a fiftieth of what escalating the
        whole node costs, and it leaves the bulk of the tokens being generated
        locally -- which is the number that was 86% cloud and supposed to be 45%.

        Advisory: a consult that fails for any reason returns "" and the round
        proceeds exactly as it would have. It must never be able to fail a node.
        """
        try:
            builder = ContextBuilder(budget_tokens=CONSULT_BUDGET_TOKENS)
            builder.add(
                "The task being attempted", task, priority=P_ACCEPTANCE, max_tokens=800
            )
            builder.add(
                "What the gates reported",
                report.render(),
                priority=P_FAILURE,
                max_tokens=2500,
                tail_lines=80,
            )
            files = read_files(ctx.root, changed[:3])
            if files:
                builder.add_files(
                    "The code that failed", files, priority=P_TASK_FILES, max_tokens=2500
                )
            project_files = read_files(ctx.root, self._project_files(ctx))
            if project_files:
                builder.add_files(
                    "Build, type, lint, and dependency configuration",
                    project_files,
                    priority=P_INTERFACES,
                    max_tokens=1200,
                )
            if previous:
                builder.add(
                    "Previous advice that did not fully resolve the failure",
                    previous,
                    priority=P_INTERFACES,
                    max_tokens=1200,
                )
            cloud_rung = next(
                (
                    name
                    for name in ctx.config.models.ladder
                    if ctx.config.models.models[name].hosted == "cloud"
                ),
                CONSULT_RUNG,
            )
            advice = self.ask(
                ctx,
                builder,
                "Diagnose this failure for another engineer who will do the fixing.\n\n"
                "State exactly what is wrong and where, and what change would put it "
                "right. Be specific about file, symbol and line where you can.\n\n"
                "Prefer causes directly demonstrated by the supplied evidence. Treat "
                "a command successfully discovering, compiling, or running a file as "
                "evidence against a filename/discovery explanation. For type-aware "
                "lint errors, inspect the manifest and type configuration before "
                "blaming source syntax or naming.\n\n"
                "Do NOT write the corrected code, do not produce an edit plan, and do "
                "not restate the error. A short, precise diagnosis is worth more here "
                "than a rewrite.\n\nReturn one JSON object with these fields: "
                "diagnosis, evidence (array), instructions (ordered array), "
                "expected_result, alternative, and confidence (low/medium/high).",
                profile=self.profile(
                    ctx,
                    min_tier=cloud_rung,
                    difficulty=0.5,
                    label=f"consult:{ctx.node.id[-8:]}",
                ),
                max_output_tokens=1024,
            )
            text = self._render_advice(advice)
        except ForgeError as exc:
            ctx.logger().warn("consult failed; continuing without advice", error=str(exc))
            return ""
        if text:
            ctx.logger().info("consulted a stronger rung for a diagnosis", chars=len(text))
        return text[:4000]

    @staticmethod
    def _render_advice(advice: Any) -> str:
        if isinstance(advice, str):
            try:
                advice = json.loads(advice)
            except (json.JSONDecodeError, TypeError):
                return advice.strip()
        if not isinstance(advice, dict):
            return str(advice or "").strip()
        instructions = advice.get("instructions") or []
        evidence = advice.get("evidence") or []
        parts = [
            f"Diagnosis: {advice.get('diagnosis', '')}",
            "Evidence:\n" + "\n".join(f"- {item}" for item in evidence),
            "Instructions:\n"
            + "\n".join(f"{index + 1}. {item}" for index, item in enumerate(instructions)),
            f"Expected result: {advice.get('expected_result', '')}",
            f"Alternative: {advice.get('alternative', '')}",
            f"Confidence: {advice.get('confidence', '')}",
        ]
        text = "\n\n".join(part for part in parts if part.split(":", 1)[-1].strip())
        return text.strip() or str(advice)

    def _previous_advice(self, ctx: AgentContext) -> str:
        title = f"Coach advice for {ctx.node.id}"
        found = next(
            (record for record in ctx.memory.search(title, limit=5) if record.title == title),
            None,
        )
        return found.body if found else ""

    def _advice_memory(self, ctx: AgentContext, advice: str) -> list[MemoryRecord]:
        if not advice:
            return []
        return [
            fact(
                f"Coach advice for {ctx.node.id}",
                advice,
                source=f"coach:{ctx.node.id}",
                tags=["coach", "diagnosis", ctx.node.kind],
                paths=list(ctx.spec.get("paths", []))[:8],
            )
        ]

    def _decompose(
        self, ctx: AgentContext, task: str, report: ValidationReport | None
    ) -> list[ProposedNode]:
        """Replace a broad local-deep failure with durable smaller local leaves."""
        depth = int(ctx.spec.get("decomposition_depth", 0))
        if (
            depth >= MAX_DECOMPOSITION_DEPTH
            or ctx.spec.get("decomposed")
            or ctx.node.tier != "local_deep"
            or ctx.node.kind not in {
                "implement", "debug", "refactor", "scaffold", "test_author", "document"
            }
            or report is None
        ):
            return []
        try:
            builder = self.builder(ctx)
            builder.add(
                "Failure evidence",
                report.render(),
                priority=P_FAILURE,
                max_tokens=3500,
                tail_lines=100,
            )
            payload = self.ask(
                ctx,
                builder,
                "The task has resisted focused local implementation. Split it into "
                "the smallest independently verifiable implementation tasks that "
                "local workers can finish. Do not solve it. Give every child a "
                "non-empty, explicit file-ownership list. Keep later-child files "
                "out of earlier children even when opportunistic wiring would be "
                "easy; the dependency chain will integrate them in order. Each "
                "child must make the original task materially smaller.",
                schema=DECOMPOSITION_SCHEMA,
                profile=self.profile(
                    ctx,
                    difficulty=0.55,
                    stakes=0.55,
                    attempt=0,
                    min_tier="local_deep",
                    max_tier="mid",
                    label=f"decompose:{ctx.node.id[-8:]}",
                ),
                max_output_tokens=4096,
            )
        except ForgeError as exc:
            ctx.logger().warn("local decomposition failed", error=str(exc))
            return []

        proposals: list[ProposedNode] = []
        for index, child in enumerate(payload.get("tasks", [])[:8]):
            proposals.append(
                ProposedNode(
                    kind="implement",
                    title=str(child["title"])[:120],
                    spec={
                        "objective": child["objective"],
                        "acceptance": child.get("acceptance", []),
                        "paths": child.get("paths", []),
                        "gates": ctx.node.gates or list(ctx.config.validation.gates),
                        "decomposition_depth": depth + 1,
                    },
                # Decomposition is for reducing cognitive load, not for
                # speculative parallelism.  A later leaf must start from the
                # integrated output of every earlier leaf; otherwise two local
                # workers independently redefine the same interface and spend
                # the rest of the run repairing each other's stale baselines.
                deps=list(range(index)),
                    priority=max(1, ctx.node.priority - 1),
                    milestone=ctx.node.milestone,
                )
            )
        return proposals if len(proposals) >= 2 else []

    def _semantic_scope_violation(self, ctx: AgentContext, task: str) -> str:
        """Reject accidental deletion of public contracts before gates run.

        Static checks cannot notice an exported interface disappearing when its
        consumers live in a later task.  This is exactly how a node whose
        objective was to *add* ``InputState`` deleted both ``InputState`` and
        ``BootConfig`` and then escalated on the cascade it created.

        Intentional API-removal/refactor nodes are allowed to do this.  For
        ordinary implementation work, a removed declaration must also appear as
        an added declaration in the patch (a modification), otherwise the local
        worker gets a precise repair instruction instead of a stronger model.
        """
        if ctx.node.kind == "refactor" or _REMOVAL_INTENT.search(task):
            return ""
        # Failed OpenCode attempts are committed as provisional ``wip:``
        # commits so the next round can improve them.  HEAD is therefore not
        # always the node's real baseline: checking only the uncommitted diff
        # makes an API invented by attempt 1 look established, and attempt 2
        # is then forbidden from removing it.  Compare against the parent of
        # the contiguous WIP chain for this node so the guard protects the
        # integrated contract, not speculative work from an earlier round.
        ref: str | None = None
        log_fn = getattr(ctx.repo, "log", None)
        node_id = getattr(ctx.node, "id", None)
        if callable(log_fn) and node_id:
            try:
                for commit in log_fn(limit=max(20, int(getattr(ctx.node, "attempts", 0)) + 5)):
                    if (
                        commit.node_id != node_id
                        or not commit.subject.startswith("wip: preserve attempt ")
                    ):
                        break
                    ref = f"{commit.sha}^"
            except (ForgeError, AttributeError, TypeError, ValueError):
                # The scope check remains useful on repositories whose history
                # cannot be inspected; fall back to the current HEAD diff.
                ref = None
        diff = ctx.repo.diff(ref=ref, max_bytes=500_000)
        removed: set[str] = set()
        added: set[str] = set()
        for match in _EXPORTED_DECLARATION.finditer(diff):
            line = match.group(0)
            (added if line.startswith("+") else removed).add(match.group("name"))
        deleted = sorted(removed - added)
        if not deleted:
            return ""
        return (
            "The patch removes existing exported contract(s) that this task did "
            "not ask to remove: "
            + ", ".join(deleted[:20])
            + ". Restore those declarations and make the requested change "
            "without deleting unrelated public APIs."
        )

    def _coding_gate_names(self, ctx: AgentContext) -> list[str]:
        """Cheap, deterministic gates used during an implementation round.

        The orchestrator runs the complete configured suite on the merged tree.
        A leaf only needs the checks that give it actionable source feedback.
        Unknown/custom gates stay enabled because a project may define its own
        compiler or focused contract check.
        """
        return [
            name for name in self.gate_names(ctx) if name not in _INTEGRATION_ONLY_GATES
        ]

    def _already_satisfied_repair(
        self, ctx: AgentContext, task: str
    ) -> AgentResult | None:
        """Complete an obsolete diagnostic repair before buying a model call.

        This is deliberately narrower than "the gates are green". Generic
        gates do not prove that a feature exists or a behavioural bug is fixed.
        They *do* prove that a named compiler/lint/import diagnostic is absent,
        which is the entire acceptance contract of the small repair leaves this
        path handles.
        """
        report = self._verify_unchanged_diagnostic_repair(ctx, task)
        if report is None or not report.passed:
            return None
        summary = (
            "The requested diagnostic repair is already satisfied on the "
            "current integrated tree"
        )
        ctx.logger().info("diagnostic repair already satisfied; skipping model")
        return AgentResult(
            success=True,
            summary=summary,
            data={"already_satisfied": True, "preflight": True},
            changed_files=[],
            report=report,
            memory=self._observations(ctx, report),
        )

    def _already_satisfied_decomposition(
        self, ctx: AgentContext
    ) -> AgentResult | None:
        """Validate successful decomposition children instead of redoing them.

        Decomposition replaces an oversized parent with focused leaves.  Once
        every recorded leaf has succeeded, replaying the original broad prompt
        uses stale file ownership and invites duplicate implementations.  The
        parent's complete gate set is the integration proof; if it is not green,
        the normal model loop still gets the failure to repair.
        """
        child_ids = list(ctx.spec.get("decomposition_children", []))
        if not ctx.spec.get("decomposed") or not child_ids or ctx.repo.is_dirty():
            return None
        try:
            children = [ctx.graph.get(str(child_id)) for child_id in child_ids]
        except KeyError:
            return None
        if any(str(child.status) != "succeeded" for child in children):
            return None
        gates = self.gate_names(ctx)
        if not gates:
            return None
        report = self.run_gates(
            ctx,
            changed_files=[],
            fail_fast=False,
            gate_names=gates,
        )
        if not report.passed:
            return None
        summary = (
            f"All {len(children)} decomposition children are integrated and the "
            "parent acceptance gates pass"
        )
        ctx.logger().info(
            "decomposed parent already satisfied; skipping model",
            children=len(children),
        )
        return AgentResult(
            success=True,
            summary=summary,
            data={"already_satisfied": True, "decomposition": True},
            changed_files=[],
            report=report,
            memory=self._observations(ctx, report),
        )

    def _verify_unchanged_diagnostic_repair(
        self, ctx: AgentContext, task: str
    ) -> ValidationReport | None:
        text = " ".join(
            (
                ctx.node.title,
                str(ctx.spec.get("objective", "")),
                task,
            )
        )
        if not _DIAGNOSTIC_REPAIR.search(text) or ctx.repo.is_dirty():
            return None
        gates = (
            self.gate_names(ctx)
            if _RUNTIME_DIAGNOSTIC.search(text)
            else self._coding_gate_names(ctx)
        )
        if not gates:
            return None
        return self.run_gates(
            ctx,
            changed_files=[],
            fail_fast=True,
            gate_names=gates,
        )

    def _verify_unchanged_audit(
        self, ctx: AgentContext
    ) -> ValidationReport | None:
        """Accept a clean audit after the model has inspected the worktree.

        Unlike diagnostic repairs, audits are not skipped before the model call:
        generic green gates cannot establish a source-level invariant.  Once the
        model has inspected an explicitly named Audit/Verify node and deliberately
        left no diff, however, requiring an edit encourages fabricated churn.
        """
        if not _AUDIT_TASK.match(ctx.node.title) or ctx.repo.is_dirty():
            return None
        gates = self._coding_gate_names(ctx)
        if not gates:
            return None
        return self.run_gates(
            ctx,
            changed_files=[],
            fail_fast=True,
            gate_names=gates,
        )

    def _grant_files(self, ctx: AgentContext, wanted: list[str]) -> list[str]:
        """Which requested paths to actually add. Possibly none.

        A request is not authority: everything returned here is a real file
        inside the workspace, so a model asking for ``../../etc/passwd`` or for a
        directory gets nothing.

        Deliberately *not* filtered against what is already in context. A model
        that asks for a file it has supposedly been shown is telling us it could
        not see it -- the context budget dropped it, or it arrived buried in
        9000 tokens of other files. Refusing on those grounds is what turned a
        correct request for vec2.ts, tuning.ts and types.ts into no grant at all.
        The caller pins these into their own section instead.
        """
        if not wanted:
            return []
        seen: set[str] = set()
        granted: list[str] = []
        for raw in wanted:
            # `removeprefix`, not `lstrip("./")`: lstrip takes a character *set*,
            # so it turned "../secret.txt" into "secret.txt" and the traversal
            # check below never saw a "..". A file of that name inside the
            # workspace would then have been served in answer to a request that
            # pointed outside it.
            path = str(raw).strip().replace("\\", "/")
            while path.startswith("./"):
                path = path.removeprefix("./")
            if not path or path in seen or path.startswith("/") or ".." in Path(path).parts:
                continue
            if path.startswith((".git/", ".forge/")):
                continue
            target = ctx.root / path
            try:
                if not target.is_file() or not target.resolve().is_relative_to(ctx.root.resolve()):
                    continue
            except OSError:  # pragma: no cover - racing filesystem
                continue
            granted.append(path)
            seen.add(path)
            if len(granted) >= MAX_REQUESTED_FILES:
                break
        return granted

    def _relevant_paths(self, ctx: AgentContext) -> list[str]:
        """Which files to show the model.

        Declared paths first, then anything the node's dependencies changed --
        an implementation almost always needs to see the interface its
        dependency just created. Capped, because past a few thousand lines the
        marginal file adds noise rather than signal.
        """
        declared = [p for p in ctx.spec.get("paths", []) if (ctx.root / p).is_file()]
        related: list[str] = []
        for dep_id in ctx.node.deps:
            dep = ctx.graph.try_get(dep_id)
            if dep and dep.result:
                related.extend(dep.result.get("changed_files", []))
        combined = list(dict.fromkeys(declared + related + self._project_files(ctx)))
        return [p for p in combined if (ctx.root / p).is_file()][:25]

    @staticmethod
    def _project_files(ctx: AgentContext) -> list[str]:
        """Build and dependency manifests, always included.

        Observed live: a node asked for `package.json`, `tsconfig.json`,
        `vite.config.ts` and `vitest.config.ts` on its third file request, was
        refused because the request budget was spent, and returned no edits at
        all -- an entire round lost to files that together are under a thousand
        tokens and that every node writing TypeScript needs to see. Whether a
        module is ESM or CJS, what `strict` is set to, where the test globs
        point: these are not discoveries, they are the ground rules, and paying
        a round-trip to learn them is absurd.
        """
        return [
            name
            for name in _PROJECT_FILES
            if (ctx.root / name).is_file()
        ]

    @staticmethod
    def _check_secrets(plan: EditPlan) -> str:
        for edit in plan.edits:
            found = scan_text_for_secrets(edit.content)
            if found:
                return ", ".join(found)
        return ""

    def _commit_message(self, ctx: AgentContext, summary: str) -> str:
        prefix = {
            "implement": "feat",
            "scaffold": "chore",
            "debug": "fix",
            "refactor": "refactor",
            "test_author": "test",
        }.get(self.kind, "chore")
        subject = (summary or ctx.node.title).strip().rstrip(".")
        return f"{prefix}: {subject[:100]}"

    def _observations(self, ctx: AgentContext, report: ValidationReport) -> list[MemoryRecord]:
        """Record durable facts learned from running the gates.

        Timings in particular: knowing the test suite takes four minutes is what
        lets later planning avoid scheduling six validation nodes in a row.
        """
        records: list[MemoryRecord] = []
        for verdict in report.verdicts:
            if verdict.duration > 60 and not verdict.cached:
                records.append(
                    fact(
                        f"Gate '{verdict.gate}' takes about {verdict.duration:.0f}s",
                        f"Measured while running node {ctx.node.id}.",
                        source="gate",
                        tags=["timing", verdict.gate],
                    )
                )
        return records

    def _findings(self, ctx: AgentContext, report: ValidationReport) -> list[MemoryRecord]:
        from ..memory.store import finding

        records: list[MemoryRecord] = []
        for verdict in report.failures:
            for issue in verdict.issues[:5]:
                records.append(
                    finding(
                        f"{verdict.gate}: {issue.message[:80]}",
                        issue.render(),
                        severity=issue.severity,
                        paths=[issue.path] if issue.path else [],
                        source=f"gate:{verdict.gate}",
                    )
                )
        return records


_NON_BLOCKING = {"format", "visual", "bundle_size", "load_perf", "deps", "dangerous_patterns", "coverage"}


def _synthetic_report(gate: str, message: str) -> ValidationReport:
    from ..validation.types import Verdict

    return ValidationReport(verdicts=[Verdict.failing(gate, message, evidence=message)])


# --------------------------------------------------------------------------
# Concrete coding agents
# --------------------------------------------------------------------------


@register
class ScaffoldAgent(CodingAgent):
    """Creates the project skeleton: tooling, config, entry points, CI.

    Given its own agent rather than being an implementation task because
    scaffolding decisions are load-bearing and hard to change later -- the
    package manager, the test runner, the directory layout -- and because a
    scaffold has no existing code to read, so its context is entirely different.
    """

    kind = "scaffold"
    difficulty = 0.4
    stakes = 0.7
    context_fraction = 0.8

    def run(self, ctx: AgentContext) -> AgentResult:
        objective = ctx.spec.get("objective") or "Create the project skeleton."
        task = (
            f"{objective}\n\n"
            "Create the project skeleton so that other agents can build on it. "
            "Include: dependency manifest with pinned major versions, the build "
            "and test scripts the architecture implies, linter and formatter "
            "configuration, a minimal runnable entry point, and a README stating "
            "how to run and test the project.\n\n"
            "The entry point must actually run and produce visible output. An "
            "empty skeleton that cannot be started gives the validation layer "
            "nothing to check."
        )
        result = self.implement(ctx, task, include_paths=[])
        if result.success:
            # Re-detect the toolchain: the scaffold just defined it, and every
            # later gate depends on knowing the real build and test commands.
            from ..workspace.sandbox import detect_toolchain

            toolchain = detect_toolchain(ctx.sandbox)
            ctx.toolchain.update(toolchain)
            result.data["toolchain"] = toolchain
            if toolchain.get("commands"):
                result.memory.append(
                    fact(
                        "Project commands",
                        "\n".join(f"{k}: {v}" for k, v in toolchain["commands"].items()),
                        source="scaffold",
                        tags=["toolchain"],
                    )
                )
        return result


@register
class ImplementAgent(CodingAgent):
    """The workhorse: implements one task against the architecture."""

    kind = "implement"
    difficulty = 0.5
    stakes = 0.5

    def run(self, ctx: AgentContext) -> AgentResult:
        objective = ctx.spec.get("objective") or ctx.node.title
        acceptance = ctx.node.acceptance
        task = (
            f"Task: {ctx.node.title}\n\n"
            f"{objective}\n\n"
            + (
                "This is done when all of the following are true:\n"
                + "\n".join(f"- {item}" for item in acceptance)
                + "\n\n"
                if acceptance
                else ""
            )
            + "Implement it now. Respect the interfaces and conventions above."
        )
        result = self.implement(ctx, task)
        if result.success:
            result.memory.extend(self._new_interfaces(ctx, result.changed_files))
        return result

    def _new_interfaces(self, ctx: AgentContext, changed: list[str]) -> list[MemoryRecord]:
        """Record what this node exposes, so siblings need not read its source.

        Cheap and high-leverage: one short record per changed module removes the
        need for every downstream agent to load the whole file.
        """
        records: list[MemoryRecord] = []
        for path in changed[:6]:
            full = ctx.root / path
            if not full.is_file() or full.suffix not in {".ts", ".tsx", ".js", ".jsx", ".py", ".rs", ".go"}:
                continue
            exported = _extract_exports(full.read_text(encoding="utf-8", errors="replace"))
            if exported:
                records.append(
                    interface(
                        f"Exports of {path}",
                        "\n".join(exported[:30]),
                        paths=[path],
                        source=f"node:{ctx.node.id}",
                        tags=["exports"],
                    )
                )
        return records


def _extract_exports(source: str) -> list[str]:
    """Pull public declarations out of source, textually.

    A real parser per language would be more accurate and much more code to
    maintain across five languages. The textual version is right often enough
    for a memory record whose purpose is orientation, and it never blocks
    anything when it is wrong.
    """
    out: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if len(stripped) > 200:
            continue
        if stripped.startswith(("export function", "export class", "export const", "export interface",
                                "export type", "export default", "export async function")):
            out.append(stripped.rstrip("{ ").rstrip())
        elif stripped.startswith(("def ", "class ", "async def ")) and not stripped.startswith("def _"):
            out.append(stripped.rstrip(":"))
        elif stripped.startswith(("pub fn ", "pub struct ", "pub trait ", "pub enum ")) or (stripped.startswith("func ") and stripped[5:6].isupper()):
            out.append(stripped.rstrip("{ ").rstrip())
    return out


@register
class DebugAgent(CodingAgent):
    """Diagnoses and fixes a specific observed failure.

    Distinct from implementation in three ways that matter: it is given the
    failure evidence as the highest-priority context, it is asked to state a
    diagnosis before proposing a fix, and it is classified as
    ``TaskClass.DEBUGGING``, which routes it on its own success statistics.
    Debugging is reliably the class where local models struggle most, and
    separating it lets the router learn that without dragging implementation up
    with it.
    """

    kind = "debug"
    task_class = TaskClass.DEBUGGING
    difficulty = 0.75
    stakes = 0.7

    def run(self, ctx: AgentContext) -> AgentResult:
        failure = ctx.spec.get("failure", {})
        evidence = failure.get("evidence") or failure.get("summary") or "No evidence recorded."
        paths = ctx.spec.get("paths", []) or self._paths_from_failure(ctx, failure)

        # Ask for a diagnosis first. Requiring the model to name the cause
        # before proposing a change materially reduces the "change something
        # nearby and hope" failure mode, and the diagnosis is recorded so a
        # repeat of the same bug is recognisable.
        diagnosis = self._diagnose(ctx, evidence, paths)

        task = (
            f"Fix this failure.\n\n"
            f"Diagnosis: {diagnosis.get('cause', 'unknown')}\n"
            f"Proposed fix: {diagnosis.get('fix', '')}\n\n"
            "Apply the minimal change that resolves the cause. Do not suppress "
            "the symptom -- deleting a failing assertion or catching and "
            "ignoring an exception is not a fix."
        )
        result = self.implement(
            ctx,
            task,
            extra_sections={"The failure": (evidence, P_FAILURE)},
            include_paths=paths,
        )
        if result.success and diagnosis.get("cause"):
            result.memory.append(
                fact(
                    f"Bug fixed: {diagnosis['cause'][:70]}",
                    f"{diagnosis.get('cause', '')}\n\nFix: {diagnosis.get('fix', '')}",
                    source=f"node:{ctx.node.id}",
                    tags=["bugfix"],
                )
            )
        result.data["diagnosis"] = diagnosis
        return result

    def _diagnose(self, ctx: AgentContext, evidence: str, paths: list[str]) -> dict[str, Any]:
        builder = self.builder(ctx)
        builder.add("The failure", evidence, priority=P_FAILURE, max_tokens=6000, tail_lines=150)
        builder.add_files(
            "Files likely involved", read_files(ctx.root, paths), priority=P_TASK_FILES, max_tokens=8000
        )
        schema = object_schema(
            {
                "cause": string("The actual root cause, specifically. Not 'a bug in the code'."),
                "evidence_for": string("What in the failure output supports this diagnosis"),
                "fix": string("The change that resolves the cause"),
                "files": array(string(), "Files that need to change"),
                "confidence": string("high, medium or low"),
            },
            required=["cause", "fix"],
        )
        return self.ask(
            ctx,
            builder,
            "Diagnose this failure before fixing it. Identify the root cause and "
            "state what in the output supports that conclusion.",
            schema=schema,
        )

    def _paths_from_failure(self, ctx: AgentContext, failure: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for issue in failure.get("issues", []):
            path = issue.get("path")
            if path and (ctx.root / path).is_file():
                paths.append(path)
        if not paths:
            paths = ctx.repo.changed_files(since="HEAD~1") if ctx.repo.has_commits() else []
        return list(dict.fromkeys(paths))[:15]


@register
class RefactorAgent(CodingAgent):
    kind = "refactor"
    task_class = TaskClass.REFACTORING
    difficulty = 0.55
    stakes = 0.4

    def run(self, ctx: AgentContext) -> AgentResult:
        objective = ctx.spec.get("objective") or ctx.node.title
        task = (
            f"{objective}\n\n"
            "This is a refactor: behaviour must not change. Every existing test "
            "must still pass without being modified. If a test needs to change, "
            "this is not a refactor and you should say so instead of proceeding."
        )
        return self.implement(ctx, task)


@register
class TestAuthorAgent(CodingAgent):
    """Writes tests for behaviour that already exists.

    Kept separate from implementation on purpose. When one agent writes code and
    its tests in a single pass, the tests encode the same misunderstanding as
    the code and pass vacuously. Writing tests against the *specification* and
    the observable behaviour, in a separate node, catches a class of bug that
    self-tested implementation never will.
    """

    kind = "test_author"
    task_class = TaskClass.TEST_AUTHORING
    difficulty = 0.45
    stakes = 0.6

    def run(self, ctx: AgentContext) -> AgentResult:
        objective = ctx.spec.get("objective") or ctx.node.title
        targets = ctx.spec.get("paths", [])
        task = (
            f"{objective}\n\n"
            "Write tests for the behaviour described by the acceptance criteria "
            "and the interfaces, not for the implementation's internals.\n\n"
            "Cover: the normal path, the boundaries, and the failure modes a "
            "user could actually hit. Skip tests that merely restate the code.\n\n"
            "If the code under test is wrong, write the test that demonstrates "
            "it and say so in your summary. Do not write a test that passes "
            "against incorrect behaviour."
        )
        result = self.implement(ctx, task, include_paths=targets)
        if result.success:
            result.nodes.extend(self._follow_up(ctx, result))
        return result

    def _follow_up(self, ctx: AgentContext, result: AgentResult) -> list[ProposedNode]:
        """If the new tests fail, that is a bug to fix, not a test to weaken."""
        if result.report and result.report.passed:
            return []
        return [
            ProposedNode(
                kind="debug",
                title=f"Fix behaviour exposed by tests from {ctx.node.title}",
                spec={
                    "objective": "The newly written tests fail. Fix the implementation, not the tests.",
                    "acceptance": ["The new tests pass without being weakened"],
                    "failure": result.report.to_dict() if result.report else {},
                    "paths": ctx.spec.get("paths", []),
                },
                priority=50,
                milestone=ctx.node.milestone,
            )
        ]
