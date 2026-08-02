# Models

Forge reaches frontier models through the operator's existing CLI logins, and
does the bulk of its work on a local Qwen server. **No API key exists anywhere
in the system.** `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are not read, not
required, and are actively blanked in the environment of the CLI subprocesses.

---

## The ladder

Cheapest first. The router walks it left to right and only climbs when the rung
below has actually failed.

| Rung | Runs on | Auth | Cost | Real limit |
| --- | --- | --- | --- | --- |
| `local` | Qwen3.6-27B on your llama.cpp server, thinking **off** | none (local) | electricity | 4 slots |
| `local_deep` | the same weights, thinking **on** | none (local) | electricity | 2 slots |
| `haiku` | `claude -p --model haiku` | Claude subscription | none | 60 calls/hour |
| `sonnet` | `claude -p --model sonnet` | Claude subscription | none | 40 calls/hour |
| `opus` | `claude -p --model opus` | Claude subscription | none | 6 calls/hour |

The interesting property is that the first escalation is free. `local` and
`local_deep` are the *same model*; the difference is whether chain-of-thought is
enabled. On this model that is a real capability jump, not a knob:

```
trivial structured request, thinking off →  19 output tokens
trivial structured request, thinking on  → 691 output tokens, 226 of them reasoning
```

So a node that fails on `local` gets a genuinely stronger attempt before
anything touches a subscription.

For mutating tasks, touching a subscription first buys a bounded diagnosis.
That advice is stored in project memory and fed to a local repair attempt. A
task that is still too broad is split into durable local child nodes (to a
bounded depth). Direct cloud authorship is the final stage after coached local
execution has failed.

### OpenCode is an executor, not the router

For compatible local coding rungs, `[coding].backend = "auto"` uses OpenCode
when its executable is installed. OpenCode owns a bounded inner tool loop: it
can inspect files, use the LSP, edit the node worktree, run focused checks and
continue a durable session on the next repair round. Forge remains outside that
session and independently runs the configured gates before accepting anything.

Forge supplies an inline, fail-closed OpenCode configuration on every run:

- `enabled_providers = ["forge-local"]`; global cloud credentials are unusable
- the main and small model both point at Forge's local OpenAI-compatible server
- web tools, external directories, skills and interactive questions are denied
- subagents are denied by default; Forge performs durable graph decomposition
- commits, pushes, branch switches, resets and worktree commands are denied
- session sharing and automatic updates are disabled

OpenCode token events are written into Forge's normal spend ledger as **local**
usage. This matters even when local electricity is not budgeted: without the
local denominator, the first cloud coaching call would appear to be 100% cloud.

For the shipped fast Qwen rung, Forge appends Qwen's documented `/no_think`
turn-level switch to the OpenCode task. The shipped `local_deep` rung remains
Forge-native. It requires llama.cpp's
request-specific `chat_template_kwargs.enable_thinking = true` and more than
OpenCode's current 32k per-step output ceiling. OpenCode's generic compatible
provider cannot reliably express that combination, so Forge does not pretend
the deep rung is active when it is not. A separate local model whose reasoning
mode is selected server-side can still use OpenCode normally.

---

## The local server

`http://127.0.0.1:10000/v1` — llama.cpp, OpenAI-compatible, no
authentication. Keep it on localhost or a private network; it has no access control of its own.

Read from its own `/props` rather than assumed:

| Property | Value | Why Forge cares |
| --- | --- | --- |
| Model | `Qwen3.6-27B-UD-Q4_K_XL.gguf` | dense 27B, served as `qwen3.6-27b` |
| Context | **131072 per slot** | `context_window` on both local rungs |
| Slots | 4 | concurrency caps: 4 fast, 2 thinking |
| Modalities | vision **and** video | `supports_vision = true`, so visual review runs locally |
| JSON Schema | GBNF-constrained | `supports_json_schema = true`; local structured output cannot be malformed |
| Reasoning | separate `reasoning_content` field | see below, this is the one that bites |

`model` is optional in requests: the server returns whatever GGUF is loaded
regardless of the id sent. Forge sends the real path anyway so the ledger
records which weights a run believed it was talking to.

### The reasoning-budget trap

This is the single most important thing to understand about this server, and it
is worth being precise because it is easy to misdiagnose as a broken model or a
context-size problem.

**Two different limits are in play:**

- **Context window** — how much the model can *see*. 131072 tokens per slot here.
  Plenty.
- **`max_tokens`** — how much the model may *write* in one response. A per-request
  cap, entirely separate from the context window.

With thinking enabled, llama.cpp puts chain-of-thought in `reasoning_content`,
**and that reasoning is charged against `max_tokens` alongside the answer.** So a
small `max_tokens` produces this:

```json
{"message": {"content": "", "reasoning_content": "Here's a thinking process..."},
 "finish_reason": "length",
 "usage": {"completion_tokens": 200}}
```

An empty answer with a full token count. Nothing is wrong with the server, the
context window or the model. It simply thought until it ran out of room to
answer.

**Forge handles this in three ways:**

1. `local` disables thinking outright (`chat_template_kwargs.enable_thinking =
   false`), so the fast rung never hits it.
2. `local_deep` enables thinking *and* pairs it with `max_output_tokens = 32768`.
   The two settings belong together; raising one without the other is the bug.
3. The provider detects empty-answer-plus-reasoning and raises
   `ReasoningBudgetExhausted`, which the client answers by **doubling the output
   budget and retrying the same rung** — never by escalating. Escalating would
   spend a subscription call to fix a number.

If you configure a custom local rung, the rule is: thinking on means
`max_output_tokens` of at least ~16k.

### One system message only

Qwen3.6's chat template raises inside its own Jinja source if a request contains
several consecutive `system` messages, and llama.cpp returns
`400 Unable to generate parser for this template`.

The OpenAI API tolerates multiple system messages, so this is easy to miss — and
Forge's context builder emits three by design (role instructions, cacheable
stable prefix, volatile context). Before this was handled, *every* local call
failed and silently escalated to a subscription rung: the platform kept working
and never once used the local model.

The OpenAI-compatible provider now merges adjacent same-role text messages
before sending. Messages carrying tool calls or tool results are left alone,
since their structure is meaningful. If you write a custom provider for this
server, do the same.

### Sampling

The server's own defaults are `temperature 1.0, top_p 0.95, top_k 20, min_p
0.05`, which is too loose for code generation. Forge sends Qwen's published
per-mode values instead:

| Mode | temperature | top_p | top_k |
| --- | --- | --- | --- |
| thinking off | 0.7 | 0.8 | 20 |
| thinking on | 0.6 | 0.95 | 20 |

Override per rung in `extra`:

```toml
[models.models.local]
extra = { thinking = false, temperature = 0.3, top_p = 0.8, top_k = 20 }
```

### Checking it

```bash
forge doctor                              # includes a live probe of the endpoint
curl -s $BASE/v1/models                   # what is loaded
curl -s http://127.0.0.1:10000/props | jq '.total_slots, .default_generation_settings.n_ctx'
```

A connection failure means the host is unreachable or the server is
down. It is never a credentials problem — there are no credentials.

---

## The subscription rungs

Both work the same way: Forge shells out to a CLI that is already logged in.

### What Forge passes, and why

**`claude -p`**

```
claude --print --output-format json
       --model opus
       --append-system-prompt "<Forge's system context>"
       --allowed-tools ""        # a generator, not an agent
       --strict-mcp-config       # no MCP servers
       --setting-sources ""      # no project settings
       --permission-mode manual
```

**`codex exec`**

```
codex exec --skip-git-repo-check
           --ephemeral                     # no session files accumulating over days
           --sandbox read-only
           --json                          # event stream, for usage accounting
           --output-last-message <file>    # the clean answer channel
           --output-schema <file>          # when Forge wants structured output
           -                               # prompt on stdin
```

Both run in an **empty scratch directory**, never the project. Both CLIs
auto-discover instruction files (`CLAUDE.md`, `AGENTS.md`) from their working
directory, and letting the project's own agent instructions merge into Forge's
carefully assembled context would quietly defeat the whole context-management
design.

Prompts go over **stdin**, not argv, because a real Forge prompt is tens of
kilobytes and would hit `E2BIG`.

### Structured output differs between them

- `codex` has `--output-schema`, a genuine constraint on the final message. It
  advertises `supports_json_schema = true`.
- `claude -p` has no equivalent. It declares no schema support, so Forge falls
  back to prompt-level schema instruction plus its validate-and-repair loop.

### The fixed overhead

Both CLIs load their own agent harness prompt before seeing Forge's. Measured on
a six-word request:

| CLI | Input tokens |
| --- | --- |
| `claude -p` | ~23000 |
| `codex exec` | ~15000 |

Nothing Forge sends reduces it. `--system-prompt` does not replace the harness
prompt, and `--bare` would but forces API-key authentication — precisely what
this design avoids.

This overhead is why these are the *top* rungs and why the router works hard to
avoid them. It is also why, when a node does need frontier help, Forge gives it
the whole problem in one call rather than a conversation.

### Quota, not money

A subscription is not billed per token, so a cash budget cannot express its
limit. Rate limits can, so each subscription rung carries `quota_per_hour`:

```toml
[models.models.claude]
quota_per_hour = 25
```

The router counts calls in the rolling hour **from the ledger**, not from an
in-process counter — a restart must not hand a rung a fresh allowance it has not
earned. A rung with no allowance left is filtered out of routing entirely, and
the request goes to whatever is available instead.

The cost figures on these rungs (`input_cost_per_mtok = 15.0` and so on) are
**notional**. They exist so the router prefers local; nobody is charged them.
`forge policy` shows both.

### Rate limits are a routing event, not a failure

If a CLI reports a plan limit mid-run, the provider raises `RateLimited`, which
the client treats as "this endpoint is unavailable, try another rung" rather
than as a node failure. The node does not consume an attempt.

### Login

```bash
claude          # once, interactively, to log in
codex login
forge doctor    # confirms both
```

`forge doctor` checks login state without making a model call — the point is to
tell you whether a long unattended run can reach its top rungs, not to spend
quota proving it.

---

## Seeding what Forge knows about this host

Everything above is also available to Forge's own agents as retrievable lessons:

```bash
forge lessons --seed
```

That installs nine verified facts about this deployment into the cross-project
lesson library — the reasoning-budget trap, the slot count, which work suits the
local rung, the CLI overheads. They are retrieved by the same relevance search
as any other lesson, so a debugging agent looking at an empty model response
gets told what it means.

They can be contradicted and retired by evidence like any other lesson, and they
are plain JSON a human can edit:

```bash
forge lessons "model returned nothing"
ls ~/.local/share/forge/lessons
```

---

## Changing the roster

### Use a shorter cloud ladder

```toml
[models]
ladder = ["local", "local_deep", "haiku", "sonnet"]
```

### Drop subscriptions entirely

```toml
[models]
ladder = ["local", "local_deep"]
```

Forge runs fine. The ladder is shorter and the router adapts. See
`examples/configs/local-only.toml`.

### Go back to a direct API key

Supported, just not the default:

```toml
[models.providers.anthropic]
kind = "anthropic"
base_url = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"

[models.models.opus_api]
provider = "anthropic"
model = "claude-opus-5"
tier = "frontier"
hosted = "cloud"
supports_json_schema = true
supports_prompt_cache = true
supports_vision = true
```

Note that the direct API provider supports **prompt caching**, which the CLIs do
not expose. On a workload making several frontier calls per node, that can
outweigh the convenience of subscription auth. Forge validates that any
`anthropic` or `openai` provider declares an `api_key_env`, so this cannot be
half-configured.

### Point at a different local server

```toml
[models.providers.local]
base_url = "http://127.0.0.1:8080/v1"

[models.models.local]
model = "your-model"
context_window = 32768
concurrency = 2
extra = { thinking = false }
```

Omit `extra.thinking` entirely for models without a thinking mode; Forge then
sends no `chat_template_kwargs` and leaves the server's default alone.
