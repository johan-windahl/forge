# Operations

Running Forge for real: setup, unattended operation, monitoring, cost control and
recovery.

---

## Host setup

### Required

- Python 3.12+
- git

That is genuinely all. The core has no third-party Python dependencies.

### Recommended

```bash
# Browser and visual verification
pip install -e ".[browser]"
playwright install chromium
playwright install-deps chromium     # system libraries

# Visual regression comparison
apt install imagemagick

# Stronger isolation for unattended runs
apt install docker.io

# Tool-driven local coding executor
npm install -g opencode-ai@latest
```

### Model access

Frontier rungs use your existing CLI logins. No API key is involved.

```bash
claude          # once, interactively
codex login
forge lessons --seed   # give Forge verified facts about this host's models
```

The local rungs need only network access to your llama.cpp server; it has no
authentication.

### Verify

```bash
forge doctor
```

Every check corresponds to a failure that would otherwise appear hours into a run
disguised as something else. It verifies CLI *login state*, not just that the
binaries exist, and does so without spending a call. A `FAIL` on an individual
rung is fine — the ladder simply gets shorter. A `FAIL` on **usable ladder** is
not.

---

## Configuration

Precedence, lowest to highest:

1. Built-in defaults
2. `/etc/forge/config.toml` — host-wide
3. `~/.config/forge/config.toml` — operator preferences
4. `<project>/.forge/config.toml` — per-project, version-controlled with the project
5. `FORGE_*` environment variables
6. CLI flags

No API keys are involved at all: frontier rungs use CLI logins and the local
server has no authentication. `forge config` is safe to paste into a bug report.

```bash
forge config --write     # starter config with comments
forge config             # the effective configuration
```

### Environment overrides

Double underscore separates levels:

```bash
FORGE_BUDGET__DAILY_COST=5.0
FORGE_SCHEDULER__WORKERS=4
FORGE_SANDBOX__KIND=docker
FORGE_LOG_LEVEL=debug
```

### The settings that matter most

```toml
[budget]
total_cost = 100.0            # hard ceiling; the only one that stops a run
daily_cost = 25.0             # cloud rate limit; local work continues past it
per_node_cost = 4.0           # stops one pathological node eating the budget
cloud_fraction_target = 0.20  # soft target, applied as back-pressure
max_cloud_fraction = 0.60     # hard generated-token admission ceiling
enforce_cost_limits = false   # subscriptions use share/quota, not notional cost
escalation_reserve = 0.25     # reserved for escalations only

[scheduler]
workers = 2                   # match your local model's concurrent slots
max_attempts = 4
lease_seconds = 1800          # longer than your slowest node

[sandbox]
kind = "docker"               # for unattended operation
command_timeout = 900

[coding]
backend = "auto"              # OpenCode when installed, native otherwise
opencode_steps = 40           # bounded inner tool iterations
opencode_rounds = 3           # independent gate/repair cycles per attempt
opencode_timeout = 7200
fallback_to_native = true
opencode_subagents = false    # Forge owns durable task decomposition

[validation]
gates = ["format", "lint", "types", "build", "unit"]
```

**`workers`** should match what your local model server can actually serve
concurrently. Setting it higher does not increase throughput; it queues requests
and makes every node look slow.

**`lease_seconds`** must exceed your slowest node. If a node legitimately takes
40 minutes and the lease is 30, a second worker will pick it up while the first
is still going. The renewal thread normally prevents this, but a hard-stalled
worker cannot renew.

### OpenCode execution

`backend = "auto"` probes `opencode` in the actual execution sandbox. If it is
unavailable, Forge uses its native structured edit-plan loop. Use
`backend = "opencode"` with `fallback_to_native = false` when a missing or
broken installation should stop visibly instead.

Each Forge node gets a durable OpenCode session mapping under
`.forge/opencode/sessions/`. Failed work remains in that node's persistent Git
worktree, and the next OpenCode repair message continues the same session.
Forge records OpenCode's `step_finish` token events in the regular ledger.

For lower startup overhead, run a persistent service:

```bash
forge opencode-config --write
OPENCODE_SERVER_PASSWORD='<local-secret>' \
OPENCODE_CONFIG_CONTENT="$(< .forge/opencode/server.json)" opencode --pure serve \
  --hostname 127.0.0.1 --port 4096
```

Then configure:

```toml
[coding]
backend = "opencode"
opencode_server_url = "http://127.0.0.1:4096"
```

The normal Forge invocation still supplies the local-only inline provider and
permissions. If the service uses basic authentication, provide
`OPENCODE_SERVER_PASSWORD` to the Forge service environment as well. A managed
server is optional: without `opencode_server_url`, each `opencode run` uses its
embedded server while its session remains durable.

For Docker, install OpenCode in `forge/workbench:latest`; Forge probes inside
the container rather than assuming the host installation is visible. The local
model endpoint must also be reachable from the container network.

---

## Starting a project

```bash
mkdir my-project && cd my-project
forge init "Build a browser-based Quake-inspired FPS with one polished level."
forge run --dry-run       # verify the pipeline without spending
forge run
```

`--dry-run` swaps every provider for a deterministic stub. It exercises the real
graph, gates, git and checkpointing with zero spend. Run it after any
configuration change.

What a dry run does **not** do: cost anything, write the response cache, or move
the routing posteriors. Stub output is marked at the source and excluded from all
three, so a rehearsal can never bill you, feed a skeleton answer to a later real
run, or teach the router that a model succeeded when no model was called.

What it *does* do is advance the real graph. Nodes succeed on stub output,
checkpoints land, and progress climbs -- typically to "89%, stalled" within about
ten seconds, ending at a blocked goal node, because the stub answers every
schema with its own field descriptions. That is a successful rehearsal, not a
build. `forge status` labels it:

```
Progress  █████████████████████████░░░ 89%   [STALLED -- needs attention]
          ^ includes work from 1 dry run(s) on the echo stub -- not real output.
```

Because the graph is shared, the cleanest habit is to rehearse in a throwaway
directory, or to `rm -rf .forge workspace` and `forge init` again before the real
run. Note also that a run takes seconds when stubbed and hours when real: if
`forge watch` goes quiet immediately, check whether the run was a dry one. `watch`
prints whether a run is active, and says so when one ends.

### Directory layout

```
my-project/
  .forge/
    config.toml        project configuration
    ledger.db          the event log — the only irreplaceable file
    logs/forge.jsonl   structured logs, rotated
    artifacts/         screenshots, videos, gate output
    cache/responses/   content-addressed model response cache
    opencode/sessions/ Forge-node to OpenCode-session mappings
    reports/index.html the dashboard
  workspace/           the git repository being built
```

**Back up `.forge/ledger.db` and `workspace/`.** Everything else is a cache.

---

## Unattended operation

### As a background process

```bash
nohup forge run --forever > /dev/null 2>&1 &
echo $! > .forge/run.pid
```

`--forever` keeps polling instead of stopping when the graph goes quiet, which is
what you want if you intend to unblock nodes from another terminal while it runs.

### As a systemd service

```ini
# /etc/systemd/system/forge@.service
[Unit]
Description=Forge autonomous build for %i
After=network.target

[Service]
Type=simple
User=forge
WorkingDirectory=/srv/forge/%i
# No API keys. The frontier rungs use the CLI logins belonging to this user, so
# the service account must be the one that ran `claude` and `codex login`.
Environment="HOME=/home/forge"
ExecStart=/usr/local/bin/forge run --forever
Restart=on-failure
RestartPreventExitStatus=3
RestartSec=30
# Forge checkpoints on SIGTERM and finishes in-flight nodes.
KillSignal=SIGTERM
TimeoutStopSec=300

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now forge@my-project
journalctl -fu forge@my-project
```

`Restart=on-failure` resumes after crashes, while `RestartPreventExitStatus=3`
prevents restart churn after Forge deliberately stops on a practically
unsolvable graph or the hard cloud ceiling. Recovery reclaims leases immediately
and redoes at most one node's in-flight attempt.

### Shutdown

`SIGINT` or `SIGTERM` stops accepting new nodes, lets in-flight nodes finish,
checkpoints, and exits. `SIGKILL` is also safe — recovery on next start handles
it — but loses the in-flight attempts.

---

## Monitoring

### Is it working?

```bash
forge status                 # progress, spend, anything needing attention
forge status --verbose       # plus memory, lessons, spend by model
forge watch                  # follow the event stream
forge watch --status         # plus a status block each tick
```

`forge watch` reads the ledger rather than attaching to the process, so it works
against a run started in another terminal, under systemd, or yesterday.

### Liveness from outside

The heartbeat is an event, so a supervising script can distinguish "still
working" from "hung" without a debugger:

```bash
forge ledger --type run.heartbeat --limit 1 --json | jq '.[0].ts'
```

### The dashboard

```bash
forge report --open
```

Self-contained HTML with screenshots embedded as data URIs — copyable, emailable,
serveable from any static host. Useful when the box running the build is
reachable only over SSH.

### Structured logs

`.forge/logs/forge.jsonl` is JSON Lines. Contextual fields (project, node, agent,
attempt) attach automatically, so one node's story can be reconstructed from an
interleaved multi-worker log:

```bash
jq -c 'select(.node == "node_01J2X...")' .forge/logs/forge.jsonl
jq -c 'select(.level == "error")' .forge/logs/forge.jsonl
```

---

## Cost control

```bash
forge policy      # ladder, model costs, learned routing, spend
forge metrics     # cost per task class, escalation rate, rework, gate time
```

### Subscription quota

Subscription rungs are limited by plan rate, not money, so they carry
`quota_per_hour`. `forge policy` shows usage in the rolling hour:

```
Subscription rungs (no API key; they use your existing CLI login):
  codex        7 call(s) in the last hour of 40
  claude       2 call(s) in the last hour of 25
```

Counts come from the ledger, so a restart does not grant a fresh allowance. When
a rung is exhausted the router filters it out and routes around it; the node does
not fail. If a CLI reports a plan limit mid-call, that is treated as "endpoint
unavailable, try another rung" and does not consume the node's attempts.

### If cloud spend is too high

Check `forge metrics` first — `by_task_class` shows exactly where it went.

1. **Lower `cloud_fraction_target`.** Raises the bar for consulting or escalating.
2. **Lower `max_cloud_fraction`.** Hard boundary on cloud-generated tokens;
   calls are rejected before their reservation could cross it.
3. **Lower `per_node_cost`.** Caps pathological nodes without affecting the rest.
4. **Shorten the ladder.** Removing `opus` and then `sonnet` keeps coaching
   available while preventing expensive direct solving.
5. **Lower `quota_per_hour`** on a subscription rung to ration it directly.
6. **Check for a flaky gate.** `forge metrics` reports gates that alternate
   fail/pass. A flaky gate is expensive twice: it trains the router to think a
   capable model is failing, and it triggers debug nodes for bugs that do not
   exist.
7. **Check the cache hit rate** in `forge status --verbose`. A low rate on a
   long run suggests context that varies when it should not.

### If quality is too low

1. **Raise `cloud_fraction_target`** — the same lever, the other way.
2. **Raise stakes for specific classes** by adjusting the relevant agent's
   `stakes` attribute.
3. **Add gates.** A deterministic check that catches the problem is cheaper and
   more reliable than a stronger model, permanently.
4. **Read `forge memory --kind assumption`.** Wrong output often traces to a
   wrong assumption Forge wrote down and told you about.

---

## When something needs a human

Blocked nodes are the only thing requiring intervention, and the rest of the
graph keeps building around them.

```bash
forge status              # blocked nodes with their questions, in full
forge node <id>           # full history of one node
forge unblock <id> "Use session cookies, no third-party identity provider."
forge run
```

The answer is recorded as a human-sourced requirement in project memory, so it
informs every later prompt rather than only the retry.

### Steering without a blocked node

`forge unblock` needs something to be stuck. To add guidance while the build is
running normally -- a quality no gate measures, a number to tune toward, a
preference between two valid approaches -- use `forge tell`:

```bash
forge tell "Flipper strength should feel weighty: a full-power lower-flipper shot reaches the top rollovers in about 0.9s."
forge tell --convention "Every entity owns its own update(); no central switch on entity type."
```

It writes a human-sourced requirement (or convention) into project memory, which
reaches every subsequent agent prompt through the normal retrieval path. This is
the main lever for subjective qualities, because they are precisely what the
deterministic gates cannot check.

To abandon a node instead:

```bash
forge cancel <id> --reason "out of scope"
```

---

## Recovery

### Crash or power loss

```bash
forge run
```

That is the whole procedure. On startup Forge releases leases held by the dead
process, promotes nodes whose dependencies completed, resets any dirty workspace,
and resumes. Redone work is at most the in-flight attempts.

### Corrupted derived state

If node states look wrong or a projection seems inconsistent:

```bash
forge repair
```

Deletes every derived table and recomputes from the event log. The log is the
truth; everything else is recomputable. Also runs `VACUUM`.

### A bad change reached the workspace

```bash
forge checkpoints                    # every restorable point
forge rollback <id> --yes            # reset the workspace
forge rollback --yes                 # defaults to the last milestone
```

Rollback resets the tree and appends a compensating event. It deliberately does
**not** rewind the graph: node state after the checkpoint stays visible, because
erasing the failed attempts would erase the reason for the rollback and the
system would try the same thing again.

### A gate is wrong, not the code

```bash
forge gates                          # what is registered and enabled
forge gates --run unit build         # run by hand
forge gates --run unit --no-cache    # bypass the result cache
```

Disable it in `.forge/config.toml` by removing it from `validation.gates`, or
configure its command:

```toml
[validation]
gates = ["format", "lint", "build", "unit"]   # dropped "types"
```

### The run is stalled

`forge status` reports `STALLED` when nothing can progress and work remains. The
blocked list says why. If nothing is blocked but it still stalls, the graph has
nodes whose dependencies terminally failed — `forge nodes --all` will show them.

---

## Maintenance

```bash
forge ledger --stats            # size, event counts by type
forge memory --export memory.md # human-readable project memory
forge lessons                   # the cross-project library
```

The lesson library lives at `~/.local/share/forge/lessons` by default. It is
plain JSON, one file per lesson, and worth version-controlling — it is the part
of the system that gets better across projects, and it is small enough to read.

Cached model responses under `.forge/cache/responses` expire after two weeks and
can be deleted at any time.

### Feeding defects back into Forge

```bash
forge feedback              # mine this project's ledger for platform defects
forge feedback --history    # everything found across all projects so far
```

`forge lessons` and `forge feedback` answer different questions, and conflating
them is why the distinction is worth stating plainly:

| | `lessons` | `feedback` |
|---|---|---|
| About | building software | Forge itself |
| Acted on by | a model, next time it plans | a human, in this repository |
| Fixed by | better context | a code change |

The reason the second exists: the lesson library already contained *"A gate whose
tool is not installed must skip, never fail"* while the `types` gate was failing
nodes over an uninstalled TypeScript compiler and escalating each one to the
costliest rung. The knowledge was present and structurally unable to act, because
a lesson is advice to a model and the fix was a change to a tuple of strings.

The detectors are arithmetic over the ledger — no model is asked to grade its own
platform. They look for shapes that mean malfunction rather than difficulty:

- a node failing repeatedly with a *byte-identical* error (real problems vary
  their symptoms; bugs repeat verbatim)
- a rung that has never once succeeded at a task class
- a gate that has never once passed
- work that succeeded locally and was then repeated on a cloud rung
- a cloud fraction sustained far above the configured target

Findings accumulate at `~/.local/share/forge/feedback`, across projects on
purpose: a defect that shows up in *every* project is both the most valuable to
fix and the hardest to see from inside any one of them, where it just looks like
how the tool behaves. A finding confirmed in a second project is promoted to
critical automatically.

`forge feedback` exits 3 when anything critical is present, so it can gate a
nightly job. Run it after any long unattended session, and paste the output at
whoever maintains Forge — including a model, which is what it is written for.

---

## Upgrading

The ledger records the schema version it was written with, and refuses to open a
database written by a newer Forge rather than corrupting it. Before upgrading a
long-running project:

```bash
systemctl stop forge@my-project
cp .forge/ledger.db .forge/ledger.db.backup
# upgrade
forge repair          # rebuild projections against the new code
forge run --dry-run   # verify
systemctl start forge@my-project
```

### Dependencies

Forge installs a project's dependencies itself, because a generated project is
not runnable the moment its manifest is written. The scaffold node produces
`package.json`; nothing has fetched `node_modules`, so every tool the validation
layer wants is absent — and a missing compiler reported as a type error is the
most expensive failure mode in the system.

The trigger is a fingerprint of the manifest and lockfile, checked after any node
that writes one. Identical fingerprint means the tree is current and nothing runs.

| | |
|---|---|
| Reproducibility | a lockfile selects the locked installer (`npm ci`, `poetry install`, `cargo fetch --locked`) |
| Idempotence | once per manifest change, not once per node |
| Bounded failure | a fingerprint that failed is recorded and **not** retried |

That last row matters most. A failing install must not become its own retry
loop, so a broken manifest degrades to "gates skip" rather than "everything
spins".

Installing dependencies executes third-party code, including `postinstall`
hooks, with whatever access the sandbox allows. That is inherent to building
software with a package manager rather than something Forge introduces, and the
sandbox is the boundary that contains it. To decline it:

```toml
[sandbox]
install_dependencies = false   # gates that need a toolchain will skip
install_timeout = 900.0
```

### Escalation pairs

When a node fails on the local rung and succeeds on a frontier one, the two
attempts share everything — same task, same context, same instructions — and
differ only in the model. That is a matched pair, and it is the strongest
available evidence about what the local model actually gets wrong.

```bash
forge escalations             # this project's pairs, added to the corpus
forge escalations --history   # the accumulated cross-project corpus
```

Pairs accumulate at `~/.local/share/forge/escalations`, harvested automatically
at each milestone. Cross-project on purpose: one project's mistake is an
anecdote, the same mistake in three unrelated projects is a property of the
model.

Capture requires `memory.keep_transcripts`, which retains each model's output on
the ledger — truncated to `transcript_max_chars`, and only for task classes where
a pair teaches something transferable (implementation, debugging, refactoring,
test authoring, review). Planning output is project-specific prose: ledger cost,
no reusable rule.

**A pair with no rejecting evidence is not recorded.** It would say only that an
escalation happened, which the routing policy already knows. The gate verdict or
validation error that rejected the weak attempt is what makes the difference
diagnosable, so it travels with the pair and its absence disqualifies it.

This module captures and does not conclude. Any future extraction pass must
filter for one thing in particular: during the run this was built from, almost
every "local failure" was a Forge defect rather than a model error — a gate that
could not run reported a missing compiler as a type error, and a schema dialect
mismatch failed a rung 67 times. Mining those pairs would teach a model to work
around bugs that no longer exist. That is why the evidence is kept.
