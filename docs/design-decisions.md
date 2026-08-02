# Design decisions

Each entry states the decision, the reasoning, what was rejected, and — where it
exists — the condition that would make us change our mind.

The brief invited challenging its own assumptions. Several entries here do
(§2, §5, §8, §15, §18, §21).

---

## 1. Event sourcing on SQLite

**Decision.** All state is an append-only event log in SQLite (WAL mode).
Everything else is a projection recomputable from it.

**Why.** The system's defining requirement is surviving crashes over days.
Event sourcing makes recovery structural rather than a feature: there is no
"was the write half-applied" question. It also makes the self-improvement loop
honest — the retrospective reads the same log the scheduler wrote, so its
premises are facts.

SQLite specifically, rather than Postgres: the write pattern is one process, a
few worker threads, a few hundred small appends per hour, and reads that are
almost always "everything since sequence N". SQLite serves that with an fsync per
commit, no daemon to supervise, no port to secure, and a database that is one
file an operator can copy, diff or email. For a system whose main job is
surviving crashes on a single box, removing a network dependency from the
durability path is the robust choice, not the lazy one.

**Rejected.** Mutable state in a normal schema (recovery becomes a per-table
reasoning exercise). JSON files (no atomicity across related writes). Postgres
(a daemon to keep alive is a new way for the run to die).

**Would change if.** Multiple machines needed to execute the same project
concurrently. The lease model and event log were chosen to make that migration
possible; the storage engine would be the thing to swap.

---

## 2. A growing task DAG, not a pipeline — and no fixed phase order

**Decision.** Work is a dependency graph that grows during execution. There is no
"planning phase" followed by an "implementation phase".

**Why.** This is the brief's implied architecture that we declined. A fixed
pipeline cannot express that the renderer can proceed while audio is blocked, and
has nowhere to put work discovered halfway through — which is most of the work in
a real build. It also serialises things that are independent, which is pure
wall-clock waste when workers are cheap.

The graph also makes *parallelism* free rather than a feature: two nodes with no
shared dependency simply run at once.

**Rejected.** Sequential phases (simple, and wrong about how software gets
built). A pure agent loop with a scratchpad (no durable structure to recover
into, and no place to express dependencies).

---

## 3. Plan one milestone at a time

**Decision.** The planner emits the task graph for the *current* milestone only,
and replans at each boundary.

**Why.** Planning the whole project up front produces a confident, detailed and
wrong graph for everything past the first milestone, because the decisions that
constrain milestone three have not been made yet. Replanning costs one extra
planning call per milestone and produces a plan informed by the code that now
exists and the mistakes already made.

**Rejected.** Full up-front planning (fast, and stale by milestone two).
Continuous replanning after every node (thrash, and expensive).

---

## 4. Leases, not locks

**Decision.** Workers claim nodes with time-bounded, token-carrying leases.

**Why.** A lock held by a process that segfaults is held forever. A lease
expires. The `INSERT` into a primary-keyed `leases` table is the exclusion
primitive, so correctness is SQLite's job rather than application logic's.

The token matters: a worker that stalls past expiry and wakes mid-write finds its
token stale and aborts, rather than corrupting whoever took over.

**Refinement learned during implementation.** On startup, leases are released
*unconditionally* rather than waiting for expiry. A newly started process holds
none, so every lease belongs to a dead predecessor; waiting would idle the project
for up to a full lease period after each crash. This was found by a test that hung.

---

## 5. Routing by measured outcome, not by asking a model

**Decision.** Callers declare a task *class*; a Beta posterior per (class, ladder
rung), fed by deterministic outcomes, picks the cheapest rung likely to succeed.
Thompson sampling handles exploration.

**Why.** The brief says "the implementation should decide when [frontier models]
are required". The obvious implementation — ask a model how hard this looks — is
both a cost in itself and unreliable, since a model's estimate of its own
difficulty is not calibrated. Measurement is cheap because the outcome is already
being computed: a gate either passed or it did not.

Beta posteriors rather than a running average because they express *uncertainty*.
Eight successes out of eight is treated differently from eighty out of eighty, so
the policy stays willing to try the cheap rung while evidence is thin and stops
second-guessing once it is not. They also give a principled exploration rule
instead of an arbitrary epsilon, and a prior so a brand-new project is not
paralysed by having no data.

**Rejected.** Static rules per task type (never improves, and is wrong for any
codebase unlike the one it was tuned on). A model classifier (cost, latency, and
no ground truth). Multi-armed bandit on raw cost (ignores that a failure costs a
retry, not just its tokens).

**Important constraint.** Only deterministic outcomes update the posterior.
Letting model-judged outcomes train the router would close a loop with no ground
truth in it.

**Two corrections made after watching a live run.** Both were cases where the
policy was defensible in theory and wrong in practice:

*Cold-start ceiling.* With no evidence, every rung's posterior is its prior, and
the priors are deliberately pessimistic about cheap rungs — so the very first
implementation request went straight to the strongest rung, spent the scarcest
resource, and learned nothing about whether local would have worked. That is
backwards: the first attempt at a low-stakes task should be cheap *because*
there is no evidence, since its job is to produce some. Low-stakes cold requests
are now capped at the second rung. High-stakes work (planning, architecture) is
exempt, because no gate catches a bad plan and there is no cheap failure to
learn from.

*Climb, do not leap.* When no rung met the requirement on a retry, the policy
jumped to the top of the ladder. Now it advances exactly one rung per attempt,
so a failed local implementation tries the free thinking rung before spending
subscription quota, and each rung in between generates evidence.

---

## 6. Cloud share as back-pressure, not a quota

**Decision.** The operator sets a *target* cloud token fraction. The router adds
the overshoot to its success requirement, making escalation less attractive when
running hot.

**Why.** A hard quota fails at the worst moment — mid-milestone, on the hard
problem that actually needed the frontier model. A proportional controller on the
quantity the operator cares about degrades smoothly instead: the system quietly
becomes more frugal and more willing to retry locally, and recovers when the
average comes back.

The hard ceilings still exist (total, daily, per-node) as a floor beneath the
controller.

---

## 7. The escalation reserve

**Decision.** A fixed fraction of the total budget is spendable only by
escalations.

**Why.** Without it, a project that routinely nudges frontier models for routine
work arrives at its hardest problem with nothing left to spend on it. That is
exactly backwards: the marginal value of a frontier token is highest on the task
that three local attempts could not solve.

---

## 8. Frontier access through CLIs, on a subscription

**Decision.** The default frontier rungs shell out to `claude -p` and
`codex exec` rather than calling the Anthropic and OpenAI HTTP APIs. No API key
is read anywhere, and both keys are blanked in the subprocess environment.

**Why.** The operator already pays for these subscriptions. Requiring a metered
API key on top means paying twice for the same capability, and it introduces a
long-lived secret that has to be stored, rotated and kept out of logs. Removing
the secret entirely is a smaller attack surface than protecting it well.

Blanking the keys in the subprocess environment is the non-obvious part: if
`ANTHROPIC_API_KEY` happens to be set, the CLIs may prefer it over the
subscription login and bill it. The failure would be silent and expensive.

**Costs, stated plainly.** Each call carries the CLI's own harness prompt --
measured at ~23k input tokens for `claude -p` and ~15k for `codex exec` on a
six-word request. Nothing Forge sends reduces it; `--system-prompt` does not
replace the harness prompt, and `--bare` would but forces API-key auth. Neither
CLI exposes prompt caching, which the direct Anthropic API does. Both are
subprocess calls, so there is process-spawn latency and no streaming.

These costs are why the CLI rungs sit at the *top* of the ladder and are reached
only when the local model has actually failed. On that usage pattern the
overhead is acceptable; on a workload making many small frontier calls it would
not be, and the direct API provider remains supported for exactly that case.

**Rejected.** Direct API with keys (works, costs money twice, adds a secret).
An MCP or SDK integration (heavier dependency, same auth question).

---

## 9. Subscription limits are quota, not money

**Decision.** Subscription-backed models carry `quota_per_hour`, counted from
the ledger over a rolling hour. Their cost figures are notional.

**Why.** A subscription is not billed per token, so a cash budget cannot express
its constraint -- but it is rate-limited, and that is a real limit that stops
work. Modelling it as calls-per-hour matches the actual failure mode.

The cost figures still have to be non-zero, and this is the subtle part: if
subscription rungs cost nothing, the router would always prefer them over the
local model, which costs a notional 0.02 per Mtok. That would invert the entire
design and burn the plan on work the local model handles fine. The notional cost
exists to preserve the router's incentive ordering, not to bill anyone.

Counting from the ledger rather than an in-process counter matters because Forge
is designed to be restarted. An in-memory counter would hand a rung a fresh
allowance after every crash.

**Consequence.** A rung with no allowance left is filtered out of routing
entirely, and a plan limit reported mid-call becomes `RateLimited`, which reroutes
rather than consuming the node's attempts.

---

## 10. The thinking toggle is a ladder rung, not a parameter

**Decision.** `local` and `local_deep` are the same Qwen weights on the same
server. The only difference is `chat_template_kwargs.enable_thinking`, plus an
output budget sized for it.

**Why.** It was going to be `reasoning_effort: high`, copied from an OpenAI-ism
the llama.cpp server ignores entirely -- so the "escalation" would have been a
no-op that quietly did nothing while the router recorded it as a real attempt.

Measuring the actual server showed the real lever, and that it is large: 19
output tokens with thinking off versus 691 with it on, for the same trivial
structured request. That makes the first escalation a genuine capability jump
that costs no network, no quota and no cash -- the best possible rung to have
between "cheap" and "expensive".

**Consequence.** The two settings are inseparable. Chain-of-thought is charged
against the same output budget as the answer, so enabling thinking without
raising `max_output_tokens` produces empty answers. See §11.

---

## 11. Empty-answer-plus-full-token-count is its own error type

**Decision.** `ReasoningBudgetExhausted`, marked retryable and explicitly *not*
escalatable. The client doubles the output budget and retries the same rung.

**Why.** llama.cpp puts chain-of-thought in `reasoning_content` and charges it
against `max_tokens`. A budget that is too small therefore yields an empty
`content` with a full token count -- which is indistinguishable from a broken
model unless you look for it.

Without this, the generic path would treat it as malformed output and escalate
to a subscription rung, spending real quota to fix a number. Verified live: the
same request that returned nothing at 64 tokens returned a correct answer after
one automatic budget bump, with no escalation.

The error carries the requested budget, the reasoning length and the finish
reason, so the log says what happened rather than "model returned nothing".

---

## 12. `tier` and `hosted` are different axes

**Decision.** A model spec carries both a capability tier (used for routing) and
a hosting location (used for accounting).

**Why.** The same local weights with a larger reasoning budget are a genuinely
higher tier — they succeed at things the cheap configuration does not — but they
are still local and must not be billed against the cloud budget. Conflating the
two made local escalations show up as 90% cloud usage, which then triggered
back-pressure that suppressed a free escalation. Found by inspecting a dry run.

This also gives the cheapest possible escalation rung: more thinking on the same
weights, no network, no cloud spend.

---

## 13. Edit plans, not unified diffs

**Decision.** Models return a list of file operations — whole-file writes and
anchored replacements — never diffs.

**Why.** Diffs require reproducing line numbers and context exactly; one
miscounted line rejects the whole patch, and the failure rate grows with file
size. Anchored replacement is robust to line drift, verifiable before writing,
and fails specifically enough to repair in one turn ("anchor matched 3 times in
src/app.ts").

Anchors also fall back to whitespace-normalised matching, because models
reproduce indentation imperfectly far more often than they get logic wrong.

**Rejected.** Unified diffs (brittle). Full-file rewrites only (burns output
tokens proportional to file size, and rewrites working code). Line-range edits
(silently wrong when the model's line numbers drift).

---

## 14. Patches apply atomically

**Decision.** A plan validates entirely in memory, then writes; or it changes
nothing.

**Why.** A half-applied plan is the one failure that is genuinely hard to recover
from, because the tree is now in a state no attempt intended. Making it
impossible costs a two-phase apply and removes an entire class of debugging.

---

## 15. Gates decide correctness; models decide quality

**Decision.** Deterministic checks are authoritative. Model review supplements
them and never overrides them, and review findings become *tracked work* rather
than verdicts.

**Why.** This is the brief's "model judgement should supplement, not replace"
taken further than it stated: reviews do not fail nodes. A review that finds
problems has succeeded at reviewing. High-severity findings become repair nodes;
everything else becomes memory the next planning pass can weigh. Otherwise a
thorough reviewer can generate more tasks than the project has, and a strict one
can stall a correct build.

---

## 16. Gate results are cached by content, and failures never are

**Decision.** A passing verdict is reused while the gate's declared inputs are
unchanged. Failing verdicts are always re-run.

**Why.** Caching passes is the largest single time saving in a long run — most
edits touch one module, and re-running the whole suite every time is pure waste.
The guarantee is preserved because an unchanged input provably produces an
unchanged result for a deterministic check.

Caching *failures* would be wrong in a way that is hard to notice: a fix touching
a file outside the gate's declared inputs would leave the gate permanently red.
The asymmetry is deliberate.

---

## 17. Visual verification is split in two

**Decision.** A pixel comparison decides *whether it changed*; a vision model
decides *whether it is good*, and runs only when the pixel comparison says
something changed.

**Why.** Sending every screenshot to a vision model would dominate the cloud
budget on its own. On a run capturing ninety screenshots where three differ, only
three need judgement. The pixel comparison is also reproducible, which model
judgement is not.

ImageMagick rather than decoding PNGs in Python: faster and more memory-stable
over a multi-day run, and already present on the target host.

---

## 18. The goal node is a barrier that re-judges the original request

**Decision.** A barrier node runs when nothing else can, re-reads the *original
goal* (not the plan), runs full validation, and either declares completion or
creates gap-closing work plus another barrier. Bounded to three rounds.

**Why.** "All nodes finished" is not "the thing asked for exists". A plan can be
executed perfectly and still miss the point, because it was written before the
system understood the problem. This is the closest thing the platform has to a
customer, and it is deliberately strict about the failure modes autonomous builds
actually have — compiles but does not run, feature unreachable from the UI,
placeholder content, literal requirement met while evident intent is not.

The bound matters. A goal check that keeps finding gaps is not converging, and
the honest response is to stop and hand a specific list to a human rather than
spend indefinitely.

**This is an addition to the brief**, which had no completion criterion beyond
the plan.

---

## 19. Typed memory records, superseded by title

**Decision.** Memory is typed records, not notes. Writing a record with an
existing title supersedes the old one.

**Why.** Types let retrieval weight by kind and let the platform reason about
memory ("which assumptions has evidence contradicted?"). Supersession is what
stops memory growing without bound: an assumption refined three times leaves one
active record and a readable chain of three, rather than three contradictory
statements a future prompt must disambiguate.

Assumptions specifically are the substitute for a requirements conversation. The
brief asks for a system that avoids unnecessary questions; that only works if
what it decided instead is written down, confidence-rated, and revisited.

---

## 20. BM25, not embeddings

**Decision.** Lexical retrieval with an identifier-aware tokenizer.

**Why.** Both corpora — memory records and source files — are small (thousands of
documents) and dominated by identifiers, symbol names and error strings. For that
regime a well-tuned lexical index beats a small embedding model on precision,
needs no GPU, no model download, and no index rebuild latency. The tokenizer
splits `camelCase`/`snake_case` into both the whole token and its parts, which is
where most of the recall comes from.

**Would change if.** Retrieval quality became a measured bottleneck in the
retrospective. The `Index` interface is the seam; a hybrid reranker would slot in
behind it.

---

## 21. Context sections with priorities, and a deliberate stable prefix

**Decision.** Prompts are assembled from named sections with priorities and token
ceilings. Sections that do not change within a node come first and carry the
cache breakpoint.

**Why.** This is the brief's "avoid repeatedly sending large prompts", made
mechanical. Priority ordering encodes what an agent can least afford to lose: it
can write mediocre code without the style guide, but not correct code without the
interfaces it must satisfy. Sections shrink rather than vanish, so degradation is
graceful.

The stable prefix is the cloud-token lever: ordering context so everything before
the breakpoint is byte-identical across calls within a node makes provider prompt
caching effective rather than theoretical.

A subtle bug worth recording: the elision marker has to count against the
section's budget. Forgetting that is how a packer overshoots its ceiling by
exactly the amount it spends announcing that it trimmed.

---

## 22. Lessons are global, project memory is not

**Decision.** Only lessons marked as generalising leave the project, into a
file-based library outside any project.

**Why.** A library of five hundred vague lessons is worse than twenty sharp ones,
because retrieval noise taxes every prompt. Retrospectives run every milestone
and will rediscover the same lesson repeatedly, so deduplication is essential —
and merging turns repetition into confirmation, which is what repetition means.

Lessons carry a track record. A contradicted lesson is worse than no lesson, so
contradictions are penalised harder than confirmations reward, and a lesson that
falls below a threshold retires itself.

---

## 23. Compute the retrospective's facts; ask a model only to interpret

**Decision.** Cost, rework ratio, escalation rate, gate timings and flakiness are
computed from the ledger. The model receives the numbers and produces
conclusions.

**Why.** A retrospective that asks "how did that go?" gets a fluent narrative that
may be entirely wrong, and it will be acted on. Handing over measured facts means
only the interpretation can be wrong, and the interpretation is checkable against
numbers a human can see.

---

## 24. Self-improvement proposes; it does not silently apply

**Decision.** Retrospectives write lessons and create `improve` nodes. Applying a
change is a node like any other — recorded, reviewable, revertible.

**Why.** A system that rewrites its own operating rules without leaving a trail
is not one anybody can run for five years. Every behaviour change should be
answerable to "when did it start doing that, and why?".

The `improve` agent also prefers the lightest intervention that works: a recorded
convention (free, reaches every future prompt), then configuration, then a new
gate, then code. Reaching for code first is how a platform accumulates bespoke
machinery for problems a sentence would have solved. `no_action` is a legitimate
and frequently correct outcome.

---

## 25. Promotion detection is arithmetic

**Decision.** Repeated findings are clustered by normalised signature; clusters
above a threshold are surfaced as candidates for deterministic tooling.

**Why.** The brief asks "could deterministic tooling replace model reasoning?".
Asking a model to speculate about that produces plausible answers with no
evidence. Counting how often the same class of problem was found is evidence, and
it is free — the findings are already recorded.

---

## 26. Zero required dependencies

**Decision.** The core runs on the standard library. Playwright and ImageMagick
are optional and degrade to skipped gates.

**Why.** This platform is meant to be resurrected on a new machine years from
now. Every required dependency is a future failure mode: a yanked package, an ABI
break, a Python version bump. Writing the HTTP client and the JSON Schema
validator cost roughly 300 lines and removed the entire class.

The JSON Schema validator earns its place beyond dependency avoidance: it returns
*repair-oriented* errors naming the JSON pointer and the expectation, which are
fed straight back to the model.

**Rejected.** `httpx` + `pydantic` + `jsonschema` (better ergonomics, three more
things that can break an unattended run).

---

## 27. Threads, not async or processes

**Decision.** Worker concurrency is a small thread pool.

**Why.** The work is dominated by waiting on subprocesses and HTTP. Threads make
the SQLite and git interactions straightforward, and the worker count is small
(single digits). Async would colour the entire codebase for no throughput gain
against subprocess-bound work; processes would need real IPC for a problem that
does not have one.

Nothing in the design prevents a distributed executor later — leases and event
sourcing were chosen partly for that — but adding it now would buy complexity,
not throughput.

---

## 28. Git is the durability substrate for code

**Decision.** Every unit of work is a commit; checkpoints are tags; rollback is a
reset. Driven via the `git` CLI, not a binding.

**Why.** No snapshot format to invent, rollback is exact and cheap, and the audit
trail is legible to a human — an operator can `git log` a three-day run and read
what happened, with the node id in every commit trailer. Worktrees give parallel
isolation through the same mechanism a human team would use.

The CLI rather than libgit2/pygit2: it is the interface with the strongest
backwards-compatibility guarantee in the toolchain, needs no build step, and when
something goes wrong the operator can paste the command from the log.

---

## 29. The echo provider is production code, not a fixture

**Decision.** A deterministic stub provider ships in the model layer and backs
`forge run --dry-run`.

**Why.** It makes the entire orchestration path — graph, gates, git, checkpoints,
recovery — exercisable without a token of spend, which is the fastest way to
validate a configuration change on a real project. It is also why the test suite
needs no network and finishes in three seconds.

---

## 30. Error taxonomy drives scheduling without a model

**Decision.** Every error declares `retryable`, `escalatable` and `transient`.

**Why.** The scheduler must decide retry-vs-escalate-vs-give-up thousands of
times without consulting a model. Encoding it in the exception type makes those
decisions total, testable and free.

`transient` in particular matters: a network blip must not consume a node's
attempt budget, or an hour of provider instability permanently fails a project
that had nothing wrong with it.

---

## 31. Sandbox denylist is not presented as security

**Decision.** A denylist of destructive commands, explicitly documented as *not*
a security boundary; the container is.

**Why.** Presenting a regex list as isolation invites relying on it. Its actual
job is catching the mundane accident — a generated build script containing
`rm -rf /` — before it costs an afternoon. Real isolation is `sandbox.kind =
"docker"`, one container per command so the environment on day forty matches day
one.
