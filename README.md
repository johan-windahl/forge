# Forge

An autonomous software engineering platform. You give it one sentence describing
what you want built. It plans, implements, tests, drives a browser, reviews its
own work, deploys, and improves its own workflow — unattended, for days, across
crashes.

```bash
forge init "Build a browser-based Quake-inspired FPS with one polished level."
forge run
```

That is the whole interface. Everything else — the architecture, the milestones,
the test strategy, which model handles which task — Forge decides and writes
down.

> **Forge runs model-generated commands on your machine.** The default sandbox
> is `kind = "local"`, so builds, tests and tooling execute directly on the host
> with your permissions, unattended, for as long as a run lasts. Use
> `kind = "docker"` in `.forge/config.toml`, or a disposable VM, for anything you
> would not hand to an untrusted script. See [docs/operations.md](docs/operations.md).

---

## What makes this different from a coding agent in a loop

Four things, and they are all about what happens on day three rather than in the
first ten minutes.

**Machines decide whether the work is done.** A compiler, a test runner, a pixel
comparison and an HTTP status code are not opinions. Forge's gates are
deterministic and cached by content hash; a model's judgement supplements them
and never replaces them. When a model *is* asked to judge — "does this level look
finished?" — it runs after the objective checks and its findings become tracked
work, not a verdict.

**State is an append-only log.** Every fact is an event in SQLite; the task
graph, the budget and the memory are projections that can be thrown away and
recomputed. Crash recovery is not a feature that was added, it is a consequence
of the storage model. `forge repair` rebuilds every derived table from the log.

**Model choice is measured, not guessed.** Each request declares what *kind of
task* it is — planning, debugging, extraction — never which model it wants. The
router keeps a Beta posterior per (task class, ladder rung) fed by deterministic
outcomes, and picks the cheapest rung likely to succeed. Over a day of running it
learns that this codebase's debugging needs a stronger model while its
refactoring does not, and shifts spend accordingly. Cloud usage above the
configured target quietly raises the bar for escalating.

**It improves its own workflow.** After every milestone Forge computes what
actually happened — cost per task class, rework ratio, where the wall clock went,
which gates are flaky — and turns those numbers into lessons that persist across
projects. When the same problem is found three times, it proposes replacing that
model reasoning with a lint rule or a gate.

---

## Installation

Requires Python 3.12+ and git. **The core has no third-party dependencies.**

```bash
git clone https://github.com/johan-windahl/forge.git && cd forge
pip install -e .

# Optional: browser and visual verification
pip install -e ".[browser]"
playwright install chromium

# Optional but recommended: visual regression comparison
apt install imagemagick

# Optional but recommended: tool-driven local coding sessions
npm install -g opencode-ai@latest

forge doctor        # check this host can run a build
```

### Models: no API keys

Frontier models are reached through your existing CLI logins, so they draw on a
**subscription** rather than a metered API key. Nothing in Forge reads
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; both are actively blanked in the CLI
subprocess environment so a stray key cannot silently get billed instead.

```bash
claude          # once, interactively, to log in
codex login
```

The shipped ladder, cheapest first:

| Rung | Runs on | Auth | Real limit |
| --- | --- | --- | --- |
| `local` | Qwen3.6-27B on your llama.cpp server, thinking off | none (local) | 4 slots |
| `local_deep` | the same weights, thinking on | none (local) | 2 slots |
| `haiku` | `claude -p --model haiku` | Claude subscription | 60 calls/hour |
| `sonnet` | `claude -p --model sonnet` | Claude subscription | 40 calls/hour |
| `opus` | `claude -p --model opus` | Claude subscription | 6 calls/hour |

The two local rungs expect an OpenAI-compatible server (llama.cpp, vLLM, Ollama)
at `http://127.0.0.1:10000/v1`. Point Forge elsewhere with `FORGE_LOCAL_BASE_URL`
or `[models.providers.local].base_url` in `.forge/config.toml`; a box on your own
private network works as well as localhost. The server needs no authentication,
so do not expose it to the public internet. `forge doctor` probes it live.

The first escalation is free: `local` and `local_deep` are the same model, and
the difference is whether chain-of-thought is enabled. On this model that is a
real capability jump — 19 output tokens versus 691 on the same trivial request —
so a failing node gets a genuinely stronger attempt before anything touches a
subscription.

Forge runs with whatever is reachable. No subscription means a shorter ladder
and a router that adapts, not a refusal to start.

For code-writing work the ladder is a workflow, not permission to hand the
whole task upward immediately: local fast, local deep, local decomposition,
short cloud diagnosis, local repair with that advice, and only then direct
cloud implementation. Mutating nodes run in persistent per-node git worktrees;
only merged results that pass the gates on the integrated tree reach `main`.

When OpenCode is installed, Forge uses it as the inner executor for compatible
local coding rungs: OpenCode explores the repository, edits files, runs focused
commands and retains a session for later repair rounds. Forge still owns the
task graph, model ladder, cloud admission, independent validation, Git
integration and stopping rules. The inline OpenCode configuration allowlists
only the local provider, so credentials in the operator's global OpenCode
configuration cannot turn an inner coding run into an unaccounted cloud call.
Set `[coding].backend = "native"` to use Forge's structured edit-plan loop
exclusively.

Cloud share is measured from generated/output tokens. The normal target is 20%
and the hard default ceiling is 60%; projected cloud calls are rejected before
they can cross it.

Seed Forge with verified facts about this host's models:

```bash
forge lessons --seed
```

See [docs/models.md](docs/models.md) for the full picture, including the one
genuine trap in the local setup.

---

## Using it

```bash
forge init "Build a CLI tool that converts CSV to Parquet with a progress bar."
forge run                         # start, or resume after a crash
forge status                      # progress, spend, anything needing attention
forge watch                       # follow along from another terminal
forge report --open               # self-contained HTML dashboard
```

Run it unattended:

```bash
nohup forge run --forever > /dev/null 2>&1 &
```

When something needs a human, it is one node, it says exactly what it needs, and
the rest of the graph keeps building:

```bash
forge status
#   NEEDS ATTENTION -- 1 blocked node(s):
#     * Choose an authentication strategy
#       Node could not be completed after 4 attempts.
#       ...

forge unblock <node-id> "Use session cookies, no third-party identity provider."
forge run
```

To steer it without waiting for something to get stuck:

```bash
forge tell "Loading must feel instant: first interactive frame under 1.5s on a mid-range laptop."
```

### Inspecting what it decided

```bash
forge memory --kind assumption    # what it decided on your behalf, and why
forge memory --kind decision      # architectural choices with rejected alternatives
forge policy                      # which models it is using, and what it has learned
forge metrics                     # cost, rework, gate time, escalation rate
forge lessons                     # what it has learned across all projects
forge nodes --all                 # the task graph
forge node <id>                   # one node's full history
```

### When it goes wrong

```bash
forge checkpoints                 # every restorable point
forge rollback <id> --yes         # reset the workspace to one
forge repair                      # rebuild all derived state from the event log
forge gates --run unit build      # run validation by hand
forge run --dry-run               # exercise the whole pipeline, spend nothing
```

`--dry-run` swaps every provider for a deterministic stub. It drives the real
graph, real gates, real git and real checkpointing without a token of spend,
which makes it the fastest way to verify a configuration change.

---

## Documentation

| Document | What it covers |
| --- | --- |
| [Models](docs/models.md) | The ladder, the local Qwen server, subscription CLIs, and the reasoning-budget trap |
| [Architecture](docs/architecture.md) | How the layers fit together, and the lifecycle of a node |
| [Design decisions](docs/design-decisions.md) | Every significant choice, with the reasoning and the alternatives |
| [Tradeoffs](docs/tradeoffs.md) | What this design is bad at, honestly |
| [Operations](docs/operations.md) | Running it for real: deployment, monitoring, recovery, cost control |
| [Extending](docs/extending.md) | Adding agents, gates, providers and sandboxes |
| [Examples](examples/) | Worked walkthroughs |

---

## Layout

```
forge/
  kernel/       event ledger, task graph, leases, scheduling, checkpoints
  models/       providers, routing policy, budget, caching, structured output
  memory/       typed project memory, retrieval, context packing, lessons
  workspace/    git, atomic patching, sandboxed execution
  validation/   the gate framework and the built-in gates
  agents/       the specialists: plan, architect, implement, debug, review, ship
  improve/      metrics and promotion detection for self-improvement
  report/       terminal and HTML progress reporting
tests/          444 tests, no network, no real models, runs in ~40 seconds
```

The dependency direction is strictly downward: the kernel knows nothing about
models, the model layer knows nothing about agents, and agents know nothing about
scheduling. That is what keeps the failure-prone parts — models, browsers,
networks — above a layer that has no opinion about them.

---

## Status and honesty

This is a working platform, not a demonstration. The kernel, model layer,
memory, validation and orchestration are complete and tested. What you should
know before relying on it:

- The routing priors are starting points, not tuned constants. They stop
  mattering after a day of real outcomes; before that, expect the first
  milestone of a new project to lean on the subscription rungs more than the
  steady state will.
- Each CLI call carries ~15-23k tokens of the CLI's own harness prompt before
  Forge's content. That is quota, not cash, but it is why those rungs sit at the
  top of the ladder.
- Browser and visual gates need Playwright and ImageMagick. Without them those
  gates skip loudly rather than failing, so a headless server still runs
  everything else.
- The `local` sandbox runs commands on the host. For genuinely unattended
  multi-day operation on a box you care about, use `kind = "docker"`.
- Deployment is deliberately conservative and disabled by default.

See [tradeoffs](docs/tradeoffs.md) for the longer version, including what this
architecture is structurally bad at.
