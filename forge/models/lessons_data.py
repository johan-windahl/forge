"""The seed lesson content.

Kept separate from the loader so it reads as data and can be reviewed, edited or
replaced wholesale for a different host without touching code.

Every measurement below was taken on one reference machine: Qwen3.6-27B-UD-Q4_K_XL
served by llama.cpp on a single consumer GPU. The *behaviours* generalise -- an
empty answer means the thinking budget ate the response, one system message per
request, concurrency does not add throughput on a saturated GPU -- but the
numbers do not. Treat the rates as a starting point to remeasure on your own
hardware, then reseed with ``forge lessons --seed``.

Tags matter: they are how retrieval scopes a lesson to the agent that needs it.
``implement``/``debug``/``plan`` match agent kinds; ``local``/``qwen``/``cli``
match the subject.
"""

from __future__ import annotations

from typing import Any

SEED_LESSONS: list[dict[str, Any]] = [
    {
        "title": "The local Qwen rung returns an empty answer if its output budget is too small",
        "body": (
            "The llama.cpp server on the local server separates chain-of-thought into a "
            "`reasoning_content` field, and that reasoning is charged against the same "
            "output budget as the answer. With thinking enabled and a small `max_tokens`, "
            "the model spends the entire allowance thinking and returns an empty `content` "
            "with a full token count.\n\n"
            "This looks exactly like a broken model but is not. Measured: the same trivial "
            "structured request produced 19 output tokens with thinking off and 691 with it "
            "on, and returned nothing at all when capped at 200.\n\n"
            "Forge detects this and retries with a larger budget rather than escalating. "
            "If you see it in a custom rung, raise `max_output_tokens` (32768 is the "
            "shipped value for `local_deep`) or set `extra.thinking = false`."
        ),
        "tags": ["local", "qwen", "llamacpp", "routing", "debug"],
        "context": "Measured against the local server with Qwen3.6-27B-UD-Q4_K_XL.",
    },
    {
        "title": "The local Qwen template rejects more than one system message",
        "body": (
            "Qwen3.6's chat template raises inside its own Jinja source when a request "
            "contains several consecutive `system` messages, and llama.cpp returns a 400: "
            "'Unable to generate parser for this template'.\n\n"
            "The OpenAI API tolerates this, so it is easy to miss. Forge's context builder "
            "emits three system blocks by design, so before this was fixed every local call "
            "failed and silently escalated to a subscription rung -- the platform kept "
            "working while never once using the local model.\n\n"
            "The OpenAI-compatible provider now merges adjacent same-role text messages "
            "before sending. If you write a custom provider for this server, do the same."
        ),
        "tags": ["local", "qwen", "llamacpp", "provider", "debug"],
        "context": "Reproduced against the local server; a single system message succeeds, three fail.",
    },
    {
        "title": "Prefer the local rung for extraction, classification and summarization",
        "body": (
            "Qwen3.6-27B is a dense model: every parameter is active on every token, so "
            "it is slower per token than the 35B-A3B mixture it replaced but markedly "
            "stronger at code. It is entirely adequate for structured "
            "extraction, classification, summarization and routine implementation against "
            "a clear spec.\n\n"
            "It is weaker on multi-step reasoning: novel debugging, architecture and "
            "planning are where it needs help. Escalate to `local_deep` first (same "
            "weights, thinking enabled, still free) before spending subscription quota."
        ),
        "tags": ["local", "qwen", "llamacpp", "routing", "plan"],
    },
    {
        "title": "llama.cpp constrains JSON output properly, so trust the schema",
        "body": (
            "The local server honours `response_format: json_schema` with GBNF grammars, "
            "so structured output from the local rungs cannot be malformed. Do not add "
            "belt-and-braces 'respond with only JSON' instructions to local prompts: they "
            "waste tokens on a guarantee the decoder already provides.\n\n"
            "The `claude` rung has no equivalent and falls back to prompt-level schema "
            "plus validate-and-repair. The `codex` rung uses `--output-schema`, which is a "
            "real constraint."
        ),
        "tags": ["local", "qwen", "llamacpp", "structured", "implement"],
    },
    {
        "title": "The local model can see images, so visual review need not go to the cloud",
        "body": (
            "The local server reports vision and video modalities. Screenshot review, "
            "visual regression judgement and UI critique can run locally at no "
            "subscription cost.\n\n"
            "Keep the pixel-comparison gate in front of it regardless: only send a "
            "screenshot to any model when the deterministic comparison says something "
            "actually changed."
        ),
        "tags": ["local", "qwen", "llamacpp", "visual_review", "vision"],
    },
    {
        "title": "One generation at a time on the local server: slots do not add throughput",
        "body": (
            "The server advertises four slots with 131072 tokens of context each, but "
            "the GPU is saturated by a single dense-27B generation. Measured: 26.7 "
            "tok/s with one request in flight, 13.7 tok/s each with two. Aggregate "
            "throughput is the same either way, so a second concurrent request buys no "
            "extra work -- it only halves the speed of both, which doubles how close "
            "each sits to its timeout and delays every result.\n\n"
            "Forge therefore caps the whole provider at one concurrent request "
            "(`models.providers.local.max_concurrency`), not each rung separately: "
            "`local` and `local_deep` are the same weights on the same box, so "
            "per-model limits cannot express the real constraint.\n\n"
            "Corollary for measurement: take the rate with the server idle. A rate "
            "measured mid-run describes contention, not the model. Check `/slots` first."
        ),
        "tags": ["local", "llamacpp", "scheduler", "performance", "concurrency"],
    },
    {
        "title": "The local endpoint is unauthenticated and private, so unreachability is a network fact",
        "body": (
            "The base URL is http://127.0.0.1:10000/v1 and it has no "
            "authentication. A connection failure means the host is unreachable on your network or "
            "the server is down, never a credentials problem. Check with "
            "`forge doctor` before diagnosing anything else."
        ),
        "tags": ["local", "llamacpp", "environment", "debug"],
    },
    {
        "title": "A gate whose tool is not installed must skip, never fail",
        "body": (
            "Exit code 127 is the shell's 'command not found'. A gate that treats it as a "
            "failing check is the most expensive mistake available: the node retries, "
            "escalates a rung on each retry, spends the costliest model repeatedly, and "
            "finally blocks -- over a tool that was never installed.\n\n"
            "Observed live: an absent `ruff` cost four attempts and drove a scaffold node "
            "all the way to the top rung before blocking the project.\n\n"
            "Forge now probes for the binary in `applicable()` and treats exit 127 as a "
            "skip. If you add a command gate, do the same. Skips are recorded, so a project "
            "silently shipping without a linter still shows up in the retrospective."
        ),
        "tags": ["validation", "gate", "environment", "budget", "debug"],
        "context": "Diagnosed from a live run that blocked on a missing linter.",
    },
    {
        "title": "Frontier rungs run on a subscription, so quota is the limit, not money",
        "body": (
            "The `claude` and `codex` rungs shell out to the operator's logged-in CLIs. "
            "There is no API key and no per-token bill. The real constraint is plan rate "
            "limits, expressed as `quota_per_hour` on the model; the router filters a rung "
            "out when its hourly allowance is spent and routes around it.\n\n"
            "The cost figures on these rungs are notional. They exist so the router prefers "
            "local, not because anyone is charged them."
        ),
        "tags": ["cli", "routing", "budget", "plan"],
    },
    {
        "title": "Each CLI call carries a large fixed prompt overhead",
        "body": (
            "Both CLIs load their own agent harness before seeing Forge's prompt. Measured "
            "on a six-word request: about 23k input tokens for `claude -p`, about 15k for "
            "`codex exec`. Nothing Forge sends reduces it.\n\n"
            "This is why these are the top rungs and why several small frontier calls are "
            "much worse than one well-assembled call. If a node needs frontier help, give "
            "it the whole problem at once rather than a conversation."
        ),
        "tags": ["cli", "routing", "context", "budget"],
    },
    {
        "title": "Never let an API key into the environment of a CLI provider",
        "body": (
            "If ANTHROPIC_API_KEY or OPENAI_API_KEY is present, the CLIs may prefer it over "
            "the subscription login and bill the key instead. Forge blanks both variables "
            "in the subprocess environment for exactly this reason. Do not remove that."
        ),
        "tags": ["cli", "environment", "budget"],
    },
]
