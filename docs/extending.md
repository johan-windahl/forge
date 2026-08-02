# Extending Forge

The extension points were chosen so that the common additions — a new specialist
agent, a new check, a new model provider — are each one class and one decorator,
with no change anywhere else in the platform.

---

## A new specialist agent

The brief lists security review, performance review and UX review as future
additions. The first two ship; here is the third, as a worked example.

```python
# forge/agents/ux.py
from ..memory.context import P_FAILURE, P_TASK_FILES, read_files
from ..models.types import TaskClass
from .base import Agent, AgentContext, AgentResult
from .registry import register
from .reviewing import FINDINGS_SCHEMA, ReviewingAgent


@register
class UXReviewAgent(ReviewingAgent):
    """Judges whether the interface is usable, not whether it works."""

    kind = "ux_review"                     # the node kind planners can emit
    task_class = TaskClass.VISUAL_JUDGEMENT  # how the router prices it
    difficulty = 0.6
    stakes = 0.6
    commits = False

    def system_prompt(self, ctx: AgentContext) -> str:
        from .base import SHARED_PREAMBLE
        return SHARED_PREAMBLE + """
You are reviewing usability. Correctness is already covered by tests; report
what a first-time user would struggle with.
"""

    def run(self, ctx: AgentContext) -> AgentResult:
        builder = self.builder(ctx)          # goal, digest, interfaces, memory
        builder.add_files(
            "Interface code",
            read_files(ctx.root, ctx.spec.get("paths", [])),
            priority=P_TASK_FILES,
            max_tokens=8000,
        )
        result = self.ask(
            ctx,
            builder,
            "Review this interface for usability problems.",
            schema=FINDINGS_SCHEMA,
        )
        records, nodes = self._process_findings(ctx, result, source=f"ux:{ctx.node.id}")
        return AgentResult(
            success=True,
            summary=f"UX review: {len(result['findings'])} finding(s)",
            memory=records,
            nodes=nodes,
        )
```

Register the import in `forge/agents/registry.py::_load_builtins`, and add
`"ux_review"` to `_NODE_KINDS` in `forge/agents/planning.py` so the planner can
schedule it.

That is the whole change. The router will price it, the budget will account for
it, the retrospective will report on it, and its outcomes will train the routing
policy — none of which required touching those subsystems.

### What you get for free

| From | What |
| --- | --- |
| `self.builder(ctx)` | Goal, architecture digest, acceptance criteria, retrieved memory, interfaces, conventions, relevant lessons — assembled and budget-packed |
| `self.ask(...)` | Routing, budgeting, caching, schema validation, repair, escalation |
| `self.profile(ctx)` | A task profile whose difficulty rises with the node's attempt count |
| `self.run_gates(ctx)` | Gate execution with caching and ledger events |
| `AgentResult.memory` | Persisted transactionally by the orchestrator |
| `AgentResult.nodes` | Added to the graph with index-based dependency resolution |
| `AgentResult.commit_message` | Committed with the node id in a trailer |

### Rules

- **Never write to the ledger or the graph.** Return an `AgentResult`; the
  orchestrator owns the durability boundary.
- **Never choose a model.** Describe the task through `task_class`, `difficulty`
  and `stakes`.
- **Be crash-safe by being pure.** An agent that dies mid-run must have changed
  nothing that matters. Write files, by all means — the workspace is reset before
  every attempt.

---

## A new gate

```python
# forge/validation/gates/accessibility.py
from ..gate import Gate, GateContext, register
from ..types import Issue, Severity, Verdict


@register
class AltTextGate(Gate):
    name = "alt_text"
    description = "Images have alternative text"
    order = 25                      # cheap checks get low numbers
    blocking = False                # advisory: reports without stalling
    cacheable = True                # deterministic over its inputs
    inputs = ("*.html", "*.tsx", "*.jsx")   # what invalidates the cache

    def applicable(self, ctx: GateContext) -> bool:
        return any(ctx.root.rglob("*.html")) or any(ctx.root.rglob("*.tsx"))

    def run(self, ctx: GateContext) -> Verdict:
        import re

        pattern = re.compile(r"<img(?![^>]*\balt=)[^>]*>")
        issues = []
        for path in ctx.root.rglob("*"):
            if path.suffix not in {".html", ".tsx", ".jsx"} or not path.is_file():
                continue
            rel = path.relative_to(ctx.root)
            if any(p in {"node_modules", ".git", "dist"} for p in rel.parts):
                continue
            for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                if pattern.search(line):
                    issues.append(Issue(
                        message="image without alt text",
                        severity=Severity.MEDIUM,
                        path=rel.as_posix(),
                        line=lineno,
                        rule="alt-text",
                    ))
        return Verdict(
            gate=self.name,
            passed=not issues,
            summary=f"{len(issues)} image(s) missing alt text",
            issues=issues,
        )
```

Import it from `forge/validation/gates/__init__.py`, then enable it:

```toml
[validation]
gates = ["format", "lint", "types", "build", "unit", "alt_text"]
```

### Getting the gate contract right

- **`cacheable`.** True only if the verdict is a pure function of the declared
  inputs. Anything involving a server, a browser or the network must set it
  False.
- **`inputs`.** Empty means "the whole tree", which is the safe default.
  Narrowing it speeds things up and risks missing a regression from a file you
  did not list — weigh it deliberately.
- **`blocking`.** False for advisory checks. A non-blocking failure becomes a
  recorded finding rather than stalling a node.
- **`applicable`.** Return False when the gate does not apply. Skipping is
  reported and does not block, but it *is* recorded, so the retrospective can
  notice a project shipping without type checking for three days.
- **`order`.** Cheap first. With `fail_fast`, expensive gates are never reached
  on a broken tree.
- **`version()`.** Bump when the gate's logic changes, to invalidate old cached
  verdicts.

### Project-defined gates at runtime

The `improve` agent registers gates it discovers, using the same mechanism:

```python
from forge.validation.gates.command import custom_command_gate

custom_command_gate("e2e", "npm run test:e2e", description="End-to-end suite")
```

---

## A new model provider

```python
# forge/models/provider.py
class MyProvider(Provider):
    kind = "my_provider"

    def _complete(self, request: Request, spec: ModelSpec) -> Completion:
        payload = {
            "model": spec.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_output_tokens or spec.max_output_tokens,
        }
        data = self._http.post_json(
            f"{self.config.base_url}/generate",
            payload,
            headers={"Authorization": f"Bearer {self.api_key()}"},
            timeout=spec.timeout,
        )
        return Completion(
            text=data["output"],
            model=spec.name,
            tier=spec.tier,
            usage=Usage(
                input_tokens=data["usage"]["input"],
                output_tokens=data["usage"]["output"],
            ),
            finish_reason=data.get("stop_reason", "stop"),
        )


PROVIDER_KINDS["my_provider"] = MyProvider
```

Then in config:

```toml
[models.providers.mine]
kind = "my_provider"
base_url = "https://api.example.com/v1"
api_key_env = "MY_PROVIDER_KEY"

[models.models.my_big]
provider = "mine"
model = "big-model-v2"
tier = "frontier"
hosted = "cloud"
context_window = 200000
input_cost_per_mtok = 5.0
output_cost_per_mtok = 20.0
supports_json_schema = true
supports_vision = true

[models]
ladder = ["local", "local_deep", "my_big"]
```

The base class already handles concurrency limiting, retries with jittered
backoff, timeouts, cost calculation and error classification.

**Get `hosted` right.** It controls cloud accounting and back-pressure, and it is
independent of `tier`. A locally-served large model is `tier = "frontier"`,
`hosted = "local"`.

### A CLI-backed provider

If a tool authenticates with a subscription rather than a key, subclass
`CliProvider` instead. It already handles quota-aware error classification, a
neutral working directory, stdin prompt delivery and API-key stripping:

```python
from forge.models.cli_provider import CliProvider

class MyCliProvider(CliProvider):
    kind = "my_cli"

    def _complete(self, request, spec):
        system_text, conversation = self.split_messages(request.messages)
        argv = [self.executable(), "--non-interactive", "--json"]
        result = self._invoke(argv, conversation, spec)   # raises the right error types
        ...
```

Register it in `forge/models/provider.py::_cli_kinds`, then configure it with
`kind`, `command` and a `quota_per_hour` on the model. Forge validates that a
CLI provider declares a `command`, so it cannot be half-configured.

**Declare capabilities honestly.** `supports_json_schema = false` makes the
client fall back to prompt-level schema instruction plus validate-and-repair,
which is correct but costs tokens. Claiming support the server lacks produces
confusing 400s.

---

## Multiple local models

Nothing special is needed — add them to the ladder:

```toml
[models.models.local_small]
provider = "local"
model = "qwen3-8b"
tier = "local"
hosted = "local"
context_window = 32000
input_cost_per_mtok = 0.005
output_cost_per_mtok = 0.005
extra = { thinking = false }   # omit entirely for models with no thinking mode

[models]
ladder = ["local_small", "local", "local_deep", "codex", "claude"]
```

The router learns which classes the small model can handle from the outcomes.
Watch `forge policy` over a day; classes where `local_small` succeeds will start
routing there on their own.

---

## A new sandbox

Implement `exec`, `background`, and optionally `healthy` and `teardown`:

```python
class RemoteSandbox(Sandbox):
    kind = "remote"

    def exec(self, argv, *, cwd=None, env=None, timeout=None, shell=False) -> ProcResult:
        self._check(argv)     # always: the denylist is defence in depth
        ...

    def background(self, argv, *, cwd=None, env=None, log_path=None) -> BackgroundProcess:
        ...
```

Register it in `build_sandbox`. The interface is deliberately thin — every
capability added there must be secured twice.

---

## Parallel execution and worktrees

Parallelism already works: independent nodes run concurrently up to
`scheduler.workers`. All of them share one workspace, which is correct as long as
nodes touch different files — and the planner is instructed to produce
dependencies that reflect real coupling.

For genuinely conflicting parallel work, `Repo.add_worktree` gives each node an
isolated checkout on its own branch, merged back through `Repo.merge` (which
aborts cleanly on conflict rather than leaving a conflicted tree). Wiring that
into the orchestrator is a per-node policy decision that has not been needed yet;
the primitives are there.

---

## Custom prompts without forking

Every agent's `system_prompt` is a method. Subclass, override, register under the
same `kind` — the registry keeps the last registration:

```python
from forge.agents.coding import ImplementAgent
from forge.agents.registry import register


@register
class HouseStyleImplementer(ImplementAgent):
    def system_prompt(self, ctx):
        return super().system_prompt(ctx) + """

This codebase additionally requires:
- No default exports.
- Every public function has a doc comment stating its failure modes.
"""
```

For project-specific rules, prefer writing a **convention** into memory instead —
it reaches every agent's prompt automatically and needs no code:

```bash
forge memory --kind convention   # see what is already recorded
```

The `improve` agent takes this route by default, and for the same reason: a
sentence in memory beats a subclass.
