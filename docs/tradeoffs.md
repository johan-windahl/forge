# Tradeoffs

What this design is bad at, what it costs, and where it will hurt. Written for
someone deciding whether to run it, not to reassure them.

---

## Structural limits

### Single machine

The kernel assumes one process on one box. SQLite, git worktrees and thread-based
workers all reflect that.

**Cost.** Throughput is bounded by one machine's cores and one local model
server's slots. A project that would parallelise across twenty workers cannot.

**Mitigation.** Leases, event sourcing and content-addressed caching were chosen
so a distributed executor is a storage swap plus a worker protocol, not a
rewrite. Nothing in the agent, model or validation layers assumes locality.

**When it bites.** Large monorepo migrations, or any project where the wall clock
is dominated by parallelisable independent work rather than by dependencies.

### The graph is a DAG, so it cannot express cycles

"Implement, review, revise, review again" is expressed by *creating new nodes*
rather than by looping an existing one.

**Cost.** Node count grows over a long project. A build that goes badly can
produce hundreds of repair nodes, and the graph becomes harder for a human to
read.

**Why we accepted it.** Cycles in a dependency graph make readiness undecidable
without extra machinery, and every node having an immutable outcome is what makes
the ledger interpretable. `forge nodes` hides completed work by default precisely
because of this.

### Milestone-at-a-time planning cannot see far ahead

The planner only emits tasks for the current milestone.

**Cost.** An architectural mistake that only becomes visible in milestone four is
discovered in milestone four. The system cannot notice, in milestone one, that
the module boundary it chose will not survive a requirement it has not been told
about yet.

**Why we accepted it.** The alternative — full up-front planning — has the same
blindness *and* commits to a detailed plan built on it. At least replanning
incorporates what was learned.

---

## Cost and efficiency

### The first milestone is the expensive one

Routing priors are starting points. Until real outcome data accumulates, the
policy escalates more than it eventually will, particularly for review and
debugging.

**Expect.** A first milestone with a higher cloud share than the steady state.
`forge metrics` after milestone two will show it falling as posteriors tighten.

**Mitigation.** Lower `cloud_fraction_target`, or lower `budget.total_cost` so
the affordability filter constrains routing directly. Both trade quality for
cost, honestly.

### Model-judged work does not train the router

Only deterministic outcomes update the posteriors.

**Cost.** Task classes whose output has no gate — visual judgement, retrospective
analysis — learn far more slowly than implementation or debugging, and rely
longer on their priors.

**Why we accepted it.** Letting model opinion train the model router closes a
loop with no ground truth in it. Slow learning beats confident drift.

### A missing tool used to be catastrophically expensive

Fixed, but worth knowing the shape of it. A gate whose command is not installed
returns exit 127, which was read as a failing check: the node retried, escalated
a rung each time, spent the costliest model repeatedly and finally blocked --
over an absent linter.

Gates now probe for the binary and treat 127 as a skip. The residual risk is the
opposite one: a project can ship for days without a linter and only a recorded
skip says so. `forge metrics` reports skipped gates for exactly this reason.

### Gate caching can mask a genuine regression

A gate is skipped when its declared inputs are unchanged. If a gate's real
dependencies are wider than its declared `inputs`, a change outside them will not
invalidate the cache.

**Mitigation.** The default is an empty `inputs` tuple, meaning "the whole tree",
which is conservative. Narrowing it is an explicit opt-in per gate, and the risk
should be weighed each time.

### The CLI rungs carry a large fixed overhead

Each `claude -p` or `codex exec` call loads the CLI's own agent harness prompt
before seeing Forge's content: measured at ~23k and ~15k input tokens
respectively, on a six-word request.

**Cost.** On a subscription that is quota rather than cash, but it means a
frontier call is never cheap regardless of how small the question is. It also
means those rungs cannot be used for high-volume small tasks at all.

**Why we accepted it.** The alternative is an API key, which costs real money for
capability the operator already pays for. Given the router only reaches these
rungs after the local model has failed, the call volume is low and the overhead
is tolerable. A workload that needed many small frontier calls should use the
direct API provider instead, which is still supported.

**Also.** Neither CLI exposes prompt caching, which the direct Anthropic API
does. On a node making several frontier calls, that is a real loss.

### Prompt caching only pays inside a node

The stable prefix is stable *within* a node's calls. Across nodes the digest and
retrieved memory differ, so the cache does not carry over.

**Cost.** Nodes making a single model call get no caching benefit at all. Coding
nodes, which make several, get most of it.

---

## Quality

### Model review finds real problems and also invents them

Review agents produce findings that are not always right. High-severity findings
become repair nodes automatically.

**Cost.** Occasional wasted work chasing a finding that was never a real problem.
The `certain` field on findings partly mitigates it, and severity gating means
only high-severity findings become work — but a confidently-wrong critical finding
will cost a node.

**Watch for.** Repair nodes that succeed with an empty or trivial diff. That is
the signature, and it shows up in `forge metrics` as rework with no cost
justification.

### The goal check is the weakest link in the completion guarantee

Whether the project is *done* is decided by a model looking at validation output,
the file tree and a work summary. It cannot run the software the way a person
would.

**Cost.** It will sometimes declare a project complete that a human would not,
particularly for subjective qualities — "polished", "feels good", "intuitive".

**Mitigation.** It runs after all deterministic gates, it is prompted
specifically against the failure modes autonomous builds have, and browser QA
plus visual review give it real evidence for anything with a UI. It is still
model judgement, and the honest framing is that it catches gross incompleteness
reliably and subtle inadequacy unreliably.

### Tests written by the system can be wrong in the same way as the code

Test authoring is a separate node from implementation, specifically so the tests
are not written by the agent that misunderstood the spec. That helps; it does not
eliminate the problem, since both read the same requirements.

**Watch for.** A test suite that grows while gate failures do not. Passing tests
that never fail are usually testing the implementation rather than the behaviour.

---

## Operational

### The local sandbox runs commands on your host

`sandbox.kind = "local"` executes generated build scripts directly. The denylist
catches obvious accidents and is explicitly not a security boundary.

**Mitigation.** Use `kind = "docker"` for unattended operation on a machine you
care about. It costs a couple of hundred milliseconds per command and gives you
an environment that is the same on day forty as on day one.

### An unattended run consumes subscription quota while you sleep

With the default roster nothing is billed per token, so the cash ceilings are
notional. The real limit is `quota_per_hour` per subscription rung, and hitting
it does not stop the run -- the router simply falls back to the local model,
which may produce lower-quality work overnight without anyone noticing.

**Watch for.** `forge policy` showing a rung consistently at its quota, and
`forge metrics` showing rising rework at the same time. That combination means
the ladder is effectively shorter than configured.

**If you switch to a direct API key**, the cash ceilings become real and matter.
Check `forge policy` shows costs matching your provider's actual pricing before
starting a long run, because a mis-specified per-token cost means a
mis-specified ceiling.

### The event log grows

Every model call, gate run and node transition is an event. A multi-day project
produces tens of thousands.

**Cost.** Ledger files in the hundreds of megabytes for a long project. Read
performance stays fine (indexed, and reads are almost always "since sequence N"),
but the file is not small.

**Mitigation.** `forge repair` runs `VACUUM`. There is deliberately no log
truncation: the log is the audit trail, and truncating it would compromise the
one thing event sourcing was chosen for.

### Recovery restores the workspace, not running processes

A crash mid-build leaves the tree reset and the node retried. It does not
remember that a dev server was running.

**Mitigation.** Background processes are started under process-group control and
killed as a unit, and gates that need a server start their own. A `kill -9` of
the whole process group can still leak a listener, which shows up as a confusing
"port already in use" on the next attempt.

---

## Deliberate omissions

Things we chose not to build, and why.

**No web UI.** The HTML dashboard is generated on demand and self-contained. A
live UI would need a server process — another thing that can die, another port to
secure — for information the terminal already conveys.

**No human-in-the-loop approval flow.** The brief asked for minimal human
involvement. Blocked nodes with specific questions plus `forge unblock` cover the
case where a human is genuinely required; an approval queue would invite using it.

**No multi-project scheduling.** One project per directory. Running several is
several processes, which is simpler and isolates failures.

**No streaming from the CLI rungs.** Forge needs one well-reasoned answer, not
tokens as they arrive. Streaming would add complexity to the durability boundary
for no benefit to an unattended system.

**No fine-tuning or model training.** The adaptive parts are routing and prompts.
Training a model on a project's own output has a plausible failure mode — amplify
its own mistakes — that we did not want in an unattended system.

**No speculative parallel attempts.** Running the same node on three models and
picking the best would improve quality and multiply cost. The escalation ladder
gets most of the benefit sequentially, and only pays for the stronger model when
the cheaper one has actually failed.

---

## What would make us reconsider the architecture

- **Distributed execution becomes necessary.** Swap SQLite for Postgres, add a
  worker protocol. The lease and event model already supports it.
- **Retrieval quality shows up as a measured bottleneck.** The `Index` interface
  is the seam for a hybrid lexical/embedding reranker.
- **Gate caching produces a missed regression in practice.** Move to explicit
  dependency declaration per gate with a build-system-style dependency graph.
- **The goal check proves unreliable in the direction of false completion.**
  Require a human sign-off before a project can be declared done, configurable
  per project.
