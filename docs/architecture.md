# Architecture

## The shape of the problem

An autonomous build is not one long conversation with a model. It is a
long-running distributed system whose workers happen to be language models, and
whose failure modes are the ones distributed systems have always had: partial
failure, lost work, duplicated work, unbounded resource consumption, and state
that drifts from reality.

Forge is organised around that framing. The layers below are ordered by how
often they fail, with the most failure-prone at the top and the most boring at
the bottom.

```
┌─────────────────────────────────────────────────────────────┐
│  agents/     plan · architect · implement · debug · review   │  fails often
│              test · browser QA · visual · security · deploy  │
│              retrospect · improve · goal                     │
├─────────────────────────────────────────────────────────────┤
│  models/     router · policy · budget · cache · structured   │  fails sometimes
│  validation/ gates · runner · verdicts                       │
│  memory/     records · retrieval · context packing · lessons │
│  workspace/  git · atomic patching · sandbox                 │
├─────────────────────────────────────────────────────────────┤
│  kernel/     event ledger · task graph · leases · scheduler  │  must not fail
│              checkpoints · orchestrator                      │
└─────────────────────────────────────────────────────────────┘
```

Dependencies point strictly downward. The kernel contains no reference to a
model; the model layer contains no reference to an agent; agents never touch
scheduling. This is what allows a model to hallucinate, a browser to hang and a
provider to go down without any of it reaching the layer that owns durability.

---

## The kernel

### Events are the truth

Everything that happens is an `Event` appended to a SQLite log. The node table,
the budget, the routing statistics and project memory are all *projections* —
derived tables that can be deleted and recomputed with `forge repair`.

This buys three properties that are otherwise expensive:

- **Crash recovery is structural.** There is no "was the write half-applied"
  question. Either the event is in the log or it is not.
- **Post-hoc analysis is exact.** The retrospective reads the same log the
  scheduler wrote, so its conclusions describe what happened rather than what a
  summary claimed happened.
- **Time travel.** Any prior state is reachable.

Events carry `causation_id` and `correlation_id`, so "this 90k-token frontier
call happened because that browser gate failed" is a graph traversal, not an
inference.

Projections are updated *in the same transaction* as the append. Atomicity
without paying replay cost on every read.

### Work is a growing DAG, not a pipeline

A pipeline cannot express "the renderer can proceed while audio is blocked on a
decision", and it has nowhere to put work discovered halfway through. So the
unit of work is a **node** in a dependency graph that grows as the project is
understood: planning nodes emit implementation nodes, implementation nodes emit
validation and review nodes, a failed review emits a repair node.

Node kinds map one-to-one onto specialist agents. Adding a kind is the primary
extension point.

Two scheduling concepts beyond plain dependencies:

- **Barriers.** A node marked `barrier` in its spec runs only when nothing else
  can make progress. This is how a node depends on *everything*, which static
  edges cannot express in a graph that grows while it executes. The project's
  goal node is the canonical barrier: "is this actually finished?" is only
  answerable once nothing else is left.
- **Deferral.** A failed node gets a `not_before` timestamp rather than being
  retried immediately, which is where backoff lives.

### Leases, not locks

A worker claims a node with a time-bounded lease carrying a token. A lock held
by a process that segfaults is held forever; a lease simply expires. The
`INSERT` into the `leases` table is the mutual-exclusion primitive, so exclusion
is decided by SQLite rather than by application logic.

A worker that stalls past its expiry and wakes up mid-write finds its token no
longer matches and aborts, rather than corrupting whoever took over.

On startup every existing lease is released unconditionally: a process that has
just started holds none, so any lease belongs to a dead predecessor. Waiting for
natural expiry would idle the project for up to a full lease period after every
crash — precisely when resuming promptly matters most.

### Idempotency is structural

At-least-once execution is only safe if attempts cannot compound. Before every
attempt the workspace is reset to the node's base commit, so attempt two never
inherits attempt one's half-written files. This is the single invariant that
makes lease-based retry safe, and it is enforced by the orchestrator rather than
trusted to agents.

### The durability boundary

An agent returns a *description* of what it did. The orchestrator is the only
thing that makes it real, in a fixed order chosen so a crash at any point leaves
a recoverable state:

1. **Git commit** — the code. A crash after this means the retry finds the work
   committed and produces an empty diff. Harmless.
2. **Memory** — what was learned. Idempotent by title.
3. **New nodes** — the work discovered. Duplicates would be the one genuinely
   bad outcome, so this happens after the commit that would make them
   unnecessary.
4. **Node status** — last. Until it lands, the node still belongs to this
   attempt and will simply be retried.

---

## The model layer

### The ladder

Five rungs, cheapest first, and the first two are the *same weights*: a local
Qwen server with chain-of-thought off, then the same server with it on. That
makes the first escalation free -- no network, no quota, no cash -- while still
being a real capability jump.

The cloud rungs shell out to `claude -p`, which authenticates with the
operator's existing subscription. No API key exists anywhere in the system.
See [models.md](models.md) for the full picture, including the
reasoning-budget behaviour that dominates how the local rungs are configured.

### Callers describe tasks, never models

Every request carries a `TaskProfile`: task class, difficulty, stakes, prior
attempts, whether it needs vision. Never a model name. That inversion is what
lets the routing policy change — learned from outcomes, tuned by an operator,
adapted to a new model roster — without touching a single agent.

### Routing is a measurement problem

Asking a model how hard a task is costs money and is unreliable. Forge measures
instead. Every routed call ends in a recorded success or failure, where success
is defined by whatever *deterministic gate* followed it. Those outcomes
accumulate into a Beta posterior per (task class, ladder rung).

The policy picks the cheapest rung whose posterior clears a requirement derived
from stakes, difficulty and prior attempts. Thompson sampling handles
exploration: while a rung's posterior is wide, draws vary and the cheap rung
sometimes wins even when its mean is below requirement — so it gets tried and the
posterior narrows. Once tight, draws cluster at the mean and exploration stops on
its own. No decay schedule to tune.

Two feedback paths worth noting:

- **Escalation, not repetition.** A retry moves *up* the ladder. Repeating an
  identical call with identical context is the least informative thing a system
  can do with its budget.
- **Cloud back-pressure.** The budget reports how far cloud token share is
  running above the operator's target; the policy adds that to its requirement.
  A project drifting cloud-heavy quietly becomes more reluctant to escalate, and
  a frugal one becomes more willing. A proportional controller on a quantity the
  operator actually cares about, rather than a quota that fails at the worst
  moment.

Only *deterministic* outcomes train the router. Model-judged outcomes are
excluded, because letting model opinion train the router closes a loop with no
ground truth in it.

For mutating work, escalation is coach-first. The local fast and thinking modes
author code; persistent failures may be decomposed into smaller graph nodes.
The cheapest cloud rung then returns diagnosis and ordered advice, which is
stored durably and applied by a local repair. A cloud model authors the patch
only after coached execution fails.

Each mutating node owns a persistent git worktree and branch. Failed but useful
candidate commits remain isolated there across retries. Integration is short and
serial: merge into main, rerun the affected gates without cache, then publish the
node result. A failed merge or integrated gate restores main while retaining the
candidate branch for repair.

Compatible local coding rungs may use OpenCode inside that worktree. This is an
execution boundary, not a second control plane: OpenCode performs a bounded
inspect/edit/command loop and retains its node session; Forge supplies the
task, injects any cloud coach advice, meters the reported local tokens, runs
independent gates and owns the commit. Its inline configuration allowlists one
local provider and denies cloud selection. Cross-node decomposition remains in
Forge because its child nodes have leases, branches, acceptance criteria and
durable outcomes; an invisible nested write swarm would have none of those.

### Budget

Ceilings are checked before a call, not after. Spend is derived from the ledger
so it survives restarts. Locally hosted models are exempt from cloud ceilings —
throttling them would trade the resource we have plenty of for the one we are
conserving.

The **escalation reserve** is a fixed fraction of the total budget spendable only
by escalations. Without it, a project that routinely nudges frontier models for
routine work arrives at its hardest problem with nothing left — precisely
backwards.

Note the distinction between `tier` (capability, used for routing) and `hosted`
(where the weights run, used for accounting). The same local weights with
thinking enabled are a higher tier but still local, and must not be billed as
cloud.

Subscription-backed rungs need a third concept again: `quota_per_hour`. A plan is
not billed per token, so a cash budget cannot express its limit, but it *is*
rate-limited. Quota is counted from the ledger over a rolling hour -- not an
in-process counter, because a restart must not grant a fresh allowance -- and a
rung with none left is filtered out of routing entirely rather than failing a
node.

### Structured output

Three mechanisms, strongest first: constrained decoding via `json_schema` where
the server supports it; a forced single-tool call on Anthropic; and
extract-validate-repair everywhere else.

The repair prompt is the part that matters. "That was invalid, try again" wastes
a full generation. `$.files[2].path: missing required property` gets a correct
answer in a fraction of the tokens because the model does not re-derive the whole
structure.

---

## Memory and context

### The premise

A long run cannot re-read its own history. Sending the transcript back is
quadratic in tokens and linear in confusion. So Forge writes what it learns into
**typed records** and assembles each prompt from the small subset a task needs.

| Kind | Why it is its own type |
| --- | --- |
| `requirement` | What must be true. Judged against at the end. |
| `assumption` | What Forge decided on your behalf, with confidence and revisit conditions. The substitute for a requirements conversation. |
| `decision` | An ADR. The *rejected* alternatives matter — without them a later agent re-proposes them every time. |
| `interface` | The contract between modules. Lets an agent implement against a boundary without reading the module behind it — the biggest single lever on prompt size. |
| `convention` | How this project does things. Prevents per-file style drift. |
| `fact` | Observed properties: the build takes 40s, the dev server binds 5173. |
| `finding` | An open problem from a review or gate. |
| `lesson` | Transferable knowledge about *how to work*. The only kind promoted across projects. |
| `digest` | A rollup replacing older records. Compaction made explicit and auditable. |

Writing a record with an existing title supersedes the old one. An assumption
refined three times leaves one active record and a readable chain of three,
rather than three contradictory statements a future prompt has to disambiguate.

### Retrieval

BM25 over record text, weighted by kind and boosted hard by path affinity. At
Forge's corpus sizes — thousands of records dominated by identifiers — a
well-tuned lexical index beats a small embedding model on precision, needs no
GPU and no index rebuild latency. The tokenizer splits `camelCase`,
`snake_case` and dotted paths into both the whole token and its parts, so
`renderPlayer` matches `render_player` and `PlayerRenderer`.

High-value kinds (requirements, conventions) are surfaced even without a lexical
match. An agent that never sees the conventions will violate them, and no
phrasing of the task would have retrieved them.

### Context assembly

Every prompt is built from **sections** with a priority and a token ceiling.
Sections are filled in priority order until the budget is spent, and each knows
how to shrink itself rather than simply disappearing.

Two properties are engineered deliberately:

- **Stable prefix.** Sections that do not change within a node — role, project
  digest, conventions — come first and carry the cache breakpoint. On providers
  with prompt caching this makes repeat calls within a node cost a fraction of
  their nominal input tokens.
- **Evidence last.** The task and the most recent failure output sit nearest the
  generation point, where models weight most heavily.

Priorities encode a claim about what an agent can least afford to lose: it can
write mediocre code without the style guide, but not *correct* code without the
interfaces it must satisfy or the error it is fixing.

### Lessons

Project memory dies with the project; lessons do not. A lesson is knowledge about
how to build software with this platform. The library lives outside any project
as plain JSON files, so it survives project deletion and a human can read and
prune it.

Deduplication matters more here than anywhere else, because retrospectives run
every milestone and will happily rediscover the same lesson a dozen times.
Merging keeps the library sharp and turns repetition into confirmation — which is
what repetition means. Lessons carry a track record; a contradicted lesson is
worse than no lesson, so contradictions are penalised harder than confirmations
reward.

---

## Validation

A gate is a named, cacheable, deterministic check. Verdicts are shaped for three
readers: the scheduler needs one bit, the model needs the smallest excerpt that
explains the failure, and the human needs a summary and a path to the artefact.

### Caching is the largest saving in a long run

A gate declares which inputs it depends on; the runner skips any gate whose
inputs are unchanged since it last passed. Over a multi-day run where most edits
touch one module, this removes the large majority of test-suite executions
without weakening the guarantee — an unchanged input provably produces an
unchanged result for a deterministic check.

Gates that are *not* deterministic say so (`cacheable = False`). The browser gate
does: a real server and a real renderer are involved, and treating a past pass as
proof of a present pass would be wrong.

**Failures are never cached.** A cached failure would survive a fix that touched
a file outside the gate's declared inputs, leaving a permanently red gate.
Re-running failures is worth the time.

### Ordering and isolation

Cheap gates run first — a JSON parse check before a browser boots. Gates that
bind ports or drive browsers are serialised; two dev servers racing for one port
produce failures that look like application bugs and cost hours to diagnose.

### The visual split

"Does it look right?" is split in two:

- **Has it changed?** — a pixel comparison against an approved baseline. Cheap,
  exact, reproducible.
- **Is it good?** — a vision model, which runs *only* when the pixel comparison
  says something changed.

On a run capturing ninety screenshots where three differ, only three reach a
vision model. That split is the entire efficiency argument for visual
verification.

---

## Agents

An agent builds a context, asks for a structured answer, applies it, and reports.
It does not choose a model, manage a budget, retry, checkpoint or decide what to
work on next.

The coding agents have two interchangeable inner loops. The native loop
assembles context, requests an atomic edit plan, applies it and runs gates.
The OpenCode loop lets the local model inspect and edit dynamically, then Forge
runs the same gates outside the session. Both repair within the node before
handing failure back to the scheduler, and both are bounded so a node that
cannot converge escalates rather than grinding.

### Edit plans, not diffs

Models are asked for a list of file operations, not unified diffs. Diffs require
reproducing line numbers exactly and a single miscounted line rejects the whole
patch — a failure mode that worsens as files grow.

Anchored replacement ("replace this exact snippet") is the sweet spot: robust to
line drift, verifiable before anything is written, and when an anchor is
ambiguous the error is specific enough to repair in one turn.

Every plan applies atomically. All edits validate against the current tree first,
then everything writes. A model that gets edit 4 of 6 wrong leaves the tree
exactly as it found it.

### Why some agents are separate

- **Debugging is separate from implementation** because it needs different
  context (the failure evidence first), a different prompt (state a diagnosis
  before proposing a fix), and its own routing statistics. Debugging is reliably
  where local models struggle most, and separating it lets the router learn that
  without dragging implementation up with it.
- **Test authoring is separate** because when one agent writes code and its tests
  in a single pass, the tests encode the same misunderstanding as the code and
  pass vacuously.
- **The goal check is separate** because "all the nodes finished" is not "the
  thing the human asked for exists". A plan can be executed faithfully and still
  miss the point, since it was written before the system knew what it was
  building.

---

## Lifecycle of a node

```
        ┌──────────┐  deps satisfied   ┌───────┐
        │ pending  │──────────────────▶│ ready │◀────────────┐
        └──────────┘                   └───┬───┘             │
                                           │ claim (lease)   │ lease expired
                                           ▼                 │ or backoff due
                                      ┌─────────┐            │
                                      │ running │────────────┤
                                      └────┬────┘            │
                    ┌──────────────────────┼──────────┐      │
                    ▼                      ▼          ▼      │
              ┌───────────┐         ┌──────────┐  ┌────────┐ │
              │ succeeded │         │ deferred │──┘  │blocked│
              └───────────┘         └──────────┘     └───────┘
                    │                                    ▲
                    └── promotes dependents              │
                                          attempts exhausted /
                                          human input required
```

One attempt, in order:

1. Claim a lease. A renewal thread extends it while work proceeds.
2. `NODE_STARTED` — increments attempts durably, so retries stay bounded across
   crashes.
3. Reset the workspace to the base commit.
4. Build the agent context: node spec, project digest, retrieved memory, relevant
   lessons, files.
5. Run the agent, which calls the model client, which routes, budgets, caches and
   validates.
6. Apply the result at the durability boundary.
7. Checkpoint.

Failure produces a scheduling decision — retry, escalate, block, fail — never a
stopped run.

---

## Self-improvement

The division of labour is the point: everything computable from the ledger *is*
computed, and only interpretation is asked of a model.

A retrospective that asks "how did that go?" gets a fluent narrative that may be
entirely wrong. One that asks "here are the numbers; what do they mean?" gets
something a system can act on.

Computed deterministically: cost per task class, escalation rate, rework ratio,
wall-clock distribution, gate timings, flaky gates (fail/pass alternation with no
intervening change), slowest and costliest nodes.

**Promotion detection** answers "could deterministic tooling replace model
reasoning?" by measurement. Findings and fixes are clustered by normalised
signature; any cluster above a threshold is surfaced. The decision that a cluster
is worth acting on is arithmetic; what to do about it is a model's judgement,
because that requires knowing what tooling exists.

Neither agent may silently change how the platform behaves. A retrospective
writes lessons and *proposes* changes; applying one is a node like any other,
recorded in the ledger, reviewable and revertible. A system that rewrites its own
operating rules without leaving a trail is not one anybody can run for five
years.

---

## Extending

See [extending.md](extending.md). The short version: a new specialist agent is
one class with a `kind`, a `task_class` and a `run` method, plus a decorator.
A new gate is one class with a `name` and a `run` method. A new provider is one
adapter implementing `_complete`. Nothing else in the platform learns about any
of them.
