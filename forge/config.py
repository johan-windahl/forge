"""Layered configuration.

Precedence, lowest to highest:

1. Built-in defaults (this file). Forge must run with *no* config file.
2. ``/etc/forge/config.toml`` -- host-wide, e.g. which local model is up.
3. ``~/.config/forge/config.toml`` -- operator preferences.
4. ``<project>/.forge/config.toml`` -- per-project overrides, version-controlled
   with the project so a run is reproducible from the repo alone.
5. Environment variables (``FORGE_*``), for secrets and CI.
6. Explicit CLI flags.

Secrets are never read from config files. API keys come from the environment
only, and the loaded config records *which* variable was consulted rather than
its value, so a config dump can be pasted into a bug report safely.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError

CONFIG_FILENAME = "config.toml"
FORGE_DIR = ".forge"


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ModelSpec:
    """One concrete model behind a provider."""

    name: str
    provider: str = "local"
    model: str = ""
    #: Capability tier, used for routing and eligibility.
    tier: str = "local"  # local | mid | frontier
    #: Where the weights run. Distinct from ``tier`` on purpose: a bigger
    #: reasoning budget on the same local weights is a higher *tier* but is
    #: still locally hosted, and must not be counted against the cloud budget.
    hosted: str = "local"  # local | cloud
    context_window: int = 32_768
    max_output_tokens: int = 8_192
    # Cost per million tokens, in whatever currency the operator cares about.
    # Local models are not free (electricity, wall-clock) but are ~100x cheaper;
    # a small non-zero cost keeps the router from treating them as unlimited.
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    supports_tools: bool = True
    supports_json_schema: bool = False
    supports_prompt_cache: bool = False
    supports_vision: bool = False
    # Requests in flight allowed against this model.
    concurrency: int = 2
    timeout: float = 600.0
    #: Calls allowed per rolling hour. 0 means unlimited. This is how a
    #: subscription-backed model expresses its real constraint: plan quota is
    #: rate-limited, not billed per token, so a cash budget cannot express it.
    quota_per_hour: int = 0
    #: Observed generation rate, output tokens per second. 0 means unmeasured,
    #: which disables the deliverability checks below.
    #:
    #: This exists because ``max_output_tokens`` and ``timeout`` are not
    #: independent: a budget the model cannot physically emit before the socket
    #: deadline is a request that always fails, and it fails after burning the
    #: entire timeout. Swapping a 27B dense model in for a 3B-active mixture
    #: took generation from ~85 tok/s to ~26 and made every local rung impossible
    #: at once -- two hours of a live run produced not one completed call, and
    #: the symptom was a read timeout that named the network.
    #:
    #: Measure it against an *idle* server. A rate measured under load is a
    #: statement about contention, not about the model.
    tokens_per_second: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_mtok / 1_000_000
            + output_tokens * self.output_cost_per_mtok / 1_000_000
        )

    def deliverable_tokens(self, *, reserve: float = 0.15) -> int:
        """How many output tokens can actually arrive before the timeout.

        ``reserve`` holds back part of the window for prompt ingestion and
        network overhead. Returns 0 when the rate is unmeasured, meaning "no
        opinion" rather than "nothing".
        """
        if self.tokens_per_second <= 0 or self.timeout <= 0:
            return 0
        return max(1, int(self.timeout * (1.0 - reserve) * self.tokens_per_second))


@dataclass(slots=True)
class ProviderConfig:
    #: openai_compat | anthropic | openai | claude_cli | codex_cli | echo
    kind: str = "openai_compat"
    base_url: str = "http://127.0.0.1:8080/v1"
    #: Environment variable holding an API key. Empty means none is needed,
    #: which is the case for a private local server and for both CLI
    #: providers (they use the operator's existing subscription login).
    api_key_env: str = ""
    timeout: float = 600.0
    max_retries: int = 4
    #: Ceiling on concurrent requests across *all* models this provider serves.
    #: 0 means only the per-model ``concurrency`` limits apply. Needed when
    #: several rungs are one piece of hardware: two rungs on one saturated GPU
    #: split its throughput rather than adding to it.
    max_concurrency: int = 0
    headers: dict[str, str] = field(default_factory=dict)

    # -- CLI providers ---------------------------------------------------
    #: Executable to invoke, e.g. "claude" or "codex".
    command: str = ""
    #: Extra arguments appended to every invocation.
    args: list[str] = field(default_factory=list)
    #: Working directory for the subprocess. Defaults to a scratch directory.
    #: Deliberately *not* the project workspace: both CLIs auto-discover
    #: instruction files (CLAUDE.md, AGENTS.md) from their cwd, and Forge
    #: assembles its own context on purpose. Letting the project's own files
    #: leak in would be uncontrolled context injection.
    cwd: str = ""


@dataclass(slots=True)
class ModelsConfig:
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    models: dict[str, ModelSpec] = field(default_factory=dict)
    # Ordered escalation ladder. The router walks it left to right.
    ladder: list[str] = field(default_factory=lambda: ["local", "local_deep", "frontier"])
    default: str = "local"
    cache_enabled: bool = True
    cache_ttl_seconds: float = 14 * 24 * 3600


@dataclass(slots=True)
class BudgetConfig:
    """Ceilings, expressed in cost units and tokens.

    ``cloud_fraction_target`` is the interesting knob: the router treats it as a
    soft target for the share of *tokens* served by cloud models and adjusts its
    escalation threshold to hold that share, rather than enforcing a hard quota
    that would stall a project mid-milestone.
    """

    total_cost: float = 100.0
    daily_cost: float = 25.0
    #: Subscription-backed deployments normally govern cloud use by generated
    #: token share and hourly quota, not notional currency. Direct-API users can
    #: enable the legacy monetary ceilings.
    enforce_cost_limits: bool = True
    #: Ceiling on one node's spend. Also, silently, the thing that decides
    #: whether escalation is real: past this, every cloud rung is unaffordable
    #: and each further "escalation" re-runs the strongest *local* model. Set it
    #: below the cost of reaching the top of the ladder and the top of the ladder
    #: does not exist. One node spent 16 attempts against a ceiling of 8 on a
    #: syntax error `haiku` would have fixed in one.
    per_node_cost: float = 25.0
    #: Desired share of generated tokens served by cloud models. This is a soft
    #: routing target; the hard boundary below always wins.
    cloud_fraction_target: float = 0.20
    #: Absolute ceiling on the share of generated/output tokens served by cloud
    #: models. Cloud calls are rejected before they can cross it. Input tokens
    #: are deliberately excluded because subscription CLIs add large harness
    #: prompts that do not represent work performed for the project.
    max_cloud_fraction: float = 0.60
    # Reserve, as a fraction of total, that only escalations may spend. Prevents
    # routine work from consuming the budget needed to unstick a hard task.
    escalation_reserve: float = 0.25
    stop_on_exhaustion: bool = True


def attempts_needed_for_ladder(rungs: int, escalate_after: int) -> int:
    """The smallest `max_attempts` that lets the top rung be served once.

    Escalation is one rung per failure once `escalate_after_attempts` is
    passed, so the top rung is first reached on attempt
    ``escalate_after + rungs - 1``. `forge doctor` multiplied instead, which
    failed configurations that were fine and, worse, passed some that were not.
    """
    return max(1, escalate_after) + max(0, rungs - 1)


@dataclass(slots=True)
class SchedulerConfig:
    workers: int = 2
    lease_seconds: float = 1800.0
    lease_renew_interval: float = 60.0
    #: Must be large enough to walk the whole ladder, or the rungs at the top
    #: are decoration -- see `attempts_needed_for_ladder`. `forge doctor` checks
    #: this, because the failure is silent: the node blocks with "exhausted the
    #: N-attempt budget" having never reached the models that would have
    #: finished the job.
    # Catastrophic safety ceiling, not the ordinary convergence rule. Normal
    # stopping is based on repeated failure after every strategy is exhausted.
    max_attempts: int = 50
    max_no_progress_attempts: int = 3
    backoff_base: float = 5.0
    backoff_max: float = 900.0
    backoff_jitter: float = 0.25
    poll_interval: float = 1.0
    # A node that fails this many times *after* escalating to the top tier is
    # parked as blocked rather than looping forever.
    escalate_after_attempts: int = 2
    #: How many times a *transient* failure may recur before the node is parked.
    #: Transient faults do not consume a normal attempt, which is right for a
    #: blip and wrong for an outage: a subscription rate limit spun for ninety
    #: minutes at twenty-second intervals with nothing surfaced to the operator.
    #: Generous, because the common case really is a blip.
    # Zero means retry transient infrastructure failures indefinitely. A
    # project designed to run for days must not call a temporary outage
    # unsolvable; explicit non-zero values retain a bounded operational policy.
    max_transient_attempts: int = 0
    heartbeat_interval: float = 30.0
    #: How long a stop waits for in-flight nodes before leaving them behind.
    #: Zero derives a drain window from the active model/OpenCode timeout. An
    #: explicit positive value remains available for bounded shutdowns; a
    #: second signal is always the immediate escape hatch.
    shutdown_grace: float = 0.0
    #: How often to log token usage per model. 0 disables. Separate from the
    #: heartbeat because liveness wants to be frequent and a usage table wants
    #: to be readable -- one every 30s would bury everything else in the log.
    usage_report_interval: float = 300.0


@dataclass(slots=True)
class SandboxConfig:
    kind: str = "local"  # local | docker
    image: str = "forge/workbench:latest"
    network: str = "bridge"
    cpus: float = 4.0
    memory: str = "4g"
    command_timeout: float = 900.0
    # Commands matching these prefixes are refused outright, regardless of who
    # asked. Defence in depth behind the sandbox, not instead of it.
    denied_prefixes: list[str] = field(
        default_factory=lambda: ["shutdown", "reboot", "mkfs", "dd if=", ":(){", "rm -rf /"]
    )
    allow_network: bool = True
    #: Install project dependencies when a manifest appears or changes.
    #: On by default: without it the whole validation layer skips on a freshly
    #: scaffolded project, which is how a missing compiler got mistaken for a
    #: type error and escalated to a frontier rung. Installing does execute
    #: third-party code (postinstall hooks); the sandbox is the boundary, and
    #: this switch exists so that boundary can be declined.
    install_dependencies: bool = True
    install_timeout: float = 900.0


@dataclass(slots=True)
class CodingConfig:
    """How mutating coding agents execute local implementation work.

    ``auto`` uses OpenCode when its executable is present and otherwise keeps
    Forge's native structured-edit loop. OpenCode is deliberately only an
    *execution* backend: Forge still owns routing, cloud admission, worktrees,
    validation, commits and termination.
    """

    backend: str = "auto"  # auto | native | opencode
    opencode_command: str = "opencode"
    #: Optional long-lived ``opencode serve`` URL. Empty lets ``opencode run``
    #: start its short-lived embedded server while retaining durable sessions.
    opencode_server_url: str = ""
    opencode_agent: str = "forge-local"
    opencode_steps: int = 40
    opencode_rounds: int = 3
    opencode_timeout: float = 7200.0
    #: An explicit OpenCode deployment may choose to fail closed. ``auto``
    #: always falls back when the executable is absent.
    fallback_to_native: bool = True
    #: Fresh-context repository scouts and critics. They are configured
    #: read-only; the primary remains the only writer, so every mutation is
    #: still visible to Forge's graph and validation.
    opencode_subagents: bool = False


@dataclass(slots=True)
class ValidationConfig:
    gates: list[str] = field(default_factory=lambda: ["format", "lint", "types", "build", "unit"])
    parallel: int = 4
    cache_results: bool = True
    fail_fast: bool = False
    # Visual regression tolerance, as a fraction of differing pixels.
    visual_tolerance: float = 0.005
    browser_headless: bool = True
    browser_timeout: float = 60.0
    coverage_floor: float = 0.0
    #: Per-gate settings, keyed by gate name. Merged beneath any settings a node
    #: supplies, so a project can configure routes and budgets once in config
    #: while a node stays free to override them for its own run.
    gate_settings: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryConfig:
    #: How much context an agent may assemble. Sized against the *smallest*
    #: window on the ladder, which is the local rung's 131072, and deliberately
    #: well under it: a packed prompt is not a better prompt, and every token
    #: here is read at 253 tok/s locally before generation even starts.
    #:
    #: Raised from 24000 after a day of building workarounds for the symptoms of
    #: it being too small. The `need_files` protocol, the pinned-files section
    #: and the six-file request cap all exist because context selection was
    #: guessing and guessing wrong; the model asked for more files round after
    #: round and kept being refused. 40000 is a third of the local window and a
    #: fifth of the cloud ones, so there is still room to grow if it helps.
    max_context_tokens: int = 40_000
    # Fraction of the context budget each section may claim before the packer
    # starts trimming. Sections are filled in priority order regardless.
    retrieval_limit: int = 12
    summarize_after_events: int = 200
    lessons_global_path: str = "~/.local/share/forge/lessons"
    #: Suspected defects in Forge itself, accumulated across projects. Separate
    #: from lessons because a lesson advises a model and this requires a code
    #: change -- see forge/improve/feedback.py.
    feedback_global_path: str = "~/.local/share/forge/feedback"
    #: Retain each model's output on the ledger, truncated, for task classes
    #: where a weak-then-strong pair teaches something. This is what makes
    #: escalation pairs recoverable -- see forge/improve/escalation.py. Without
    #: it the ledger records that local failed and codex succeeded but not
    #: *what* either produced, which is the only part worth learning from.
    keep_transcripts: bool = True
    #: Per-response cap. Enough to diagnose an edit plan without turning the
    #: ledger into a transcript archive.
    transcript_max_chars: int = 4000
    #: Escalation pairs accumulated across projects, the corpus a future
    #: skill-extraction pass reads.
    escalations_global_path: str = "~/.local/share/forge/escalations"


@dataclass(slots=True)
class DeployConfig:
    enabled: bool = False
    strategy: str = "static"  # static | docker | script
    command: str = ""
    healthcheck_url: str = ""
    healthcheck_timeout: float = 120.0
    rollback_on_failure: bool = True


@dataclass(slots=True)
class ImprovementConfig:
    retrospective_after_milestone: bool = True
    promote_findings_after: int = 3
    prompt_ab_testing: bool = True
    min_samples_for_routing_update: int = 8


@dataclass(slots=True)
class Config:
    """The whole configuration tree."""

    project_dir: Path = field(default_factory=Path.cwd)
    state_dir: Path | None = None
    log_level: str = "info"
    models: ModelsConfig = field(default_factory=ModelsConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    coding: CodingConfig = field(default_factory=CodingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    improvement: ImprovementConfig = field(default_factory=ImprovementConfig)
    sources: list[str] = field(default_factory=list)

    # ---- derived paths -------------------------------------------------

    @property
    def forge_dir(self) -> Path:
        return self.state_dir or (self.project_dir / FORGE_DIR)

    @property
    def ledger_path(self) -> Path:
        return self.forge_dir / "ledger.db"

    @property
    def log_path(self) -> Path:
        return self.forge_dir / "logs" / "forge.jsonl"

    @property
    def artifacts_dir(self) -> Path:
        return self.forge_dir / "artifacts"

    @property
    def cache_dir(self) -> Path:
        return self.forge_dir / "cache"

    @property
    def workspace_dir(self) -> Path:
        return self.project_dir / "workspace"

    @property
    def reports_dir(self) -> Path:
        return self.forge_dir / "reports"

    @property
    def worktrees_dir(self) -> Path:
        return self.forge_dir / "worktrees"

    def ensure_dirs(self) -> None:
        for path in (
            self.forge_dir,
            self.forge_dir / "logs",
            self.artifacts_dir,
            self.cache_dir,
            self.reports_dir,
            self.worktrees_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def redacted(self) -> dict[str, Any]:
        """Config safe to print: no secret values, only the env var names."""
        return self.to_dict()


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------


#: The local llama.cpp server, OpenAI-compatible and unauthenticated. The
#: default assumes it runs on this host; point it elsewhere with
#: ``FORGE_LOCAL_BASE_URL`` or ``[models.providers.local].base_url`` in
#: ``.forge/config.toml``. A remote server reached over a private network
#: (Tailscale or similar) needs no auth either, because the network is the
#: boundary; do not expose it to the public internet.
LOCAL_BASE_URL = os.environ.get("FORGE_LOCAL_BASE_URL", "http://127.0.0.1:10000/v1")

#: The id the server advertises on /v1/models. llama.cpp serves whatever GGUF is
#: loaded regardless of the id sent, so this is documentation rather than a
#: selector. It is sent anyway so the ledger records which weights a run
#: believed it was talking to: Qwen3.6-27B-UD-Q4_K_XL, a dense 27B replacing the
#: 35B-A3B mixture -- fewer total parameters but far more active ones per token,
#: which is where coding accuracy comes from.
LOCAL_MODEL_ID = "qwen3.6-27b"

#: Measured from the server's own /props: 131072 tokens per slot, four slots.
LOCAL_CONTEXT_WINDOW = 131_072
LOCAL_SLOTS = 4

#: Measured generation rate from the server's own `timings`, one request at a
#: time against an idle server. A *dense* 27B at Q4: roughly a third of the
#: 35B-A3B mixture it replaced, because a mixture activates ~3B parameters per
#: token and this activates all 27B. Prompt ingestion is unaffected at ~253
#: tok/s, so long contexts stay cheap and long *answers* are what costs.
#:
#: Measure this with the server idle. The first attempt at it was taken while a
#: run was live and came out at 12 tok/s -- that was two requests sharing the
#: GPU, not the model's speed, and it sent every local timeout to twice the size
#: it needed to be. `/slots` shows what is in flight.
LOCAL_TOKENS_PER_SECOND = 26.0
LOCAL_PROMPT_TOKENS_PER_SECOND = 253.0

#: Concurrent requests allowed against the local server across *both* local rungs.
#:
#: One, because the GPU is saturated by a single generation: 26.7 tok/s alone
#: versus 13.7 each when doubled. Aggregate throughput is identical, so a second
#: slot buys no extra work -- it only halves the speed of both calls, which
#: doubles how close each sits to its timeout and delays every result. Serialised,
#: the first node's answer arrives in half the time and its gates start that much
#: sooner. Raise this only for hardware where concurrency actually adds capacity.
LOCAL_MAX_CONCURRENCY = 1


def default_models() -> ModelsConfig:
    """The shipped model roster.

    Four rungs, cheapest first, and the interesting property is that the first
    two are the *same weights*:

    ``local``
        Qwen3.6-27B with thinking disabled. Dense rather than a mixture, so
        every parameter is active per token: slower than the 35B-A3B it
        replaced, and better at code. Handles the bulk of the work:
        extraction, classification, summarization, routine implementation.

    ``local_deep``
        The same server with thinking enabled and a large output budget. The
        cheapest possible escalation -- no network, no subscription quota, no
        cash -- and on this model a real capability jump rather than a knob.
        Measured on one trivial structured request: 19 output tokens without
        thinking, 691 with.

    ``codex`` / ``claude``
        Frontier models reached through the operator's existing CLI logins, so
        they draw on a *subscription* rather than an API key. See
        :mod:`forge.models.cli_provider`.

    Override any of it in ``.forge/config.toml``.
    """
    providers = {
        "local": ProviderConfig(
            kind="openai_compat",
            base_url=LOCAL_BASE_URL,
            api_key_env="",  # private network only, no auth
            # A ceiling, not a target: each model's own timeout is what applies.
            # It has to clear the slowest of them (`local_deep`, 3600s) or the
            # provider deadline would silently truncate the rung that needs the
            # most time.
            timeout=3600.0,
            max_concurrency=LOCAL_MAX_CONCURRENCY,
        ),
        # Both CLI providers authenticate with the login the operator already
        # has. No API key is read, required or set anywhere in Forge.
        "claude_cli": ProviderConfig(kind="claude_cli", command="claude", timeout=1800.0),
        "codex_cli": ProviderConfig(kind="codex_cli", command="codex", timeout=1800.0),
    }

    # Qwen3's published sampling settings per mode. The server's own defaults
    # (temperature 1.0) are too loose for code generation.
    non_thinking = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    thinking = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}

    models = {
        "local": ModelSpec(
            name="local",
            provider="local",
            model=LOCAL_MODEL_ID,
            tier="local",
            hosted="local",
            context_window=LOCAL_CONTEXT_WINDOW,
            max_output_tokens=16_384,
            # Not free: electricity and wall-clock are real, and a zero here
            # would make the router treat local as unlimited.
            input_cost_per_mtok=0.02,
            output_cost_per_mtok=0.02,
            supports_json_schema=True,  # llama.cpp constrains decoding with GBNF
            supports_tools=True,
            supports_vision=True,  # the server reports vision and video
            concurrency=LOCAL_SLOTS,
            # 16384 tokens at 26 tok/s is 11 minutes of generation. The old
            # 900s was sized for a model three times faster and left no room for
            # the ceiling to be reached at all.
            timeout=1800.0,
            tokens_per_second=LOCAL_TOKENS_PER_SECOND,
            extra={"thinking": False, **non_thinking},
        ),
        "local_deep": ModelSpec(
            name="local_deep",
            provider="local",
            model=LOCAL_MODEL_ID,
            tier="mid",
            hosted="local",
            context_window=LOCAL_CONTEXT_WINDOW,
            # Thinking needs room, and 32768 was measurably not enough: four
            # consecutive implementation calls spent 28.5k-29.2k tokens reasoning
            # (88% of the allowance), hit the cap, and returned an empty answer.
            # Every one of those cost 21 minutes before the retry that worked.
            #
            # This is a ceiling, not a target -- a short answer still costs one
            # short generation -- so the only price of raising it is the worst
            # case, and 49152 at 26 tok/s is 32 minutes inside a 60 minute
            # timeout. Kept below `deliverable_tokens` by enough that the
            # overflow bump is still available if reasoning runs longer.
            max_output_tokens=49_152,
            input_cost_per_mtok=0.04,
            output_cost_per_mtok=0.04,
            supports_json_schema=True,
            supports_tools=True,
            supports_vision=True,
            # Thinking runs are long; leave slots free for the fast rung.
            concurrency=max(1, LOCAL_SLOTS // 2),
            # Thinking plus a 32768-token budget is 21 minutes at this rate, and
            # 42 if the budget is bumped once. 1800s could not deliver the budget
            # it advertised: calls ran the socket deadline out and reported a read
            # timeout, which reads as a network fault and is not one.
            timeout=3600.0,
            tokens_per_second=LOCAL_TOKENS_PER_SECOND,
            extra={"thinking": True, **thinking},
        ),
        "codex": ModelSpec(
            name="codex",
            provider="codex_cli",
            model="",  # let the CLI use whatever the subscription allows
            tier="frontier",
            hosted="cloud",
            context_window=200_000,
            max_output_tokens=32_000,
            # Notional. A subscription is not billed per token, but the router
            # must see cloud rungs as dearer than local or it would never pick
            # local. The real limit is quota_per_hour.
            input_cost_per_mtok=3.0,
            output_cost_per_mtok=15.0,
            supports_json_schema=True,  # codex exec --output-schema
            supports_tools=False,
            supports_vision=False,
            concurrency=2,
            timeout=1800.0,
            quota_per_hour=40,
        ),
        # Three Anthropic rungs on one subscription login, cheapest first. Split
        # out because a single frontier rung is all-or-nothing: the choice was
        # "another go on the 27B" or "the most expensive model there is", and the
        # gap between them is where most stuck nodes actually live. `haiku` fixes
        # a brace error perfectly well and costs a fraction of `opus`.
        #
        # `quota_per_hour` is per rung but the subscription is shared, so these
        # are deliberately conservative: the aim is to reach for `opus` rarely,
        # not to discover its real ceiling.
        "haiku": ModelSpec(
            name="haiku",
            provider="claude_cli",
            model="haiku",
            tier="frontier",
            hosted="cloud",
            context_window=200_000,
            max_output_tokens=32_000,
            input_cost_per_mtok=1.0,
            output_cost_per_mtok=5.0,
            supports_json_schema=False,
            supports_tools=False,
            supports_vision=False,
            concurrency=2,
            timeout=1800.0,
            quota_per_hour=60,
        ),
        "sonnet": ModelSpec(
            name="sonnet",
            provider="claude_cli",
            model="sonnet",
            tier="frontier",
            hosted="cloud",
            context_window=200_000,
            max_output_tokens=32_000,
            input_cost_per_mtok=3.0,
            output_cost_per_mtok=15.0,
            supports_json_schema=False,
            supports_tools=False,
            supports_vision=False,
            concurrency=2,
            timeout=1800.0,
            quota_per_hour=40,
        ),
        "opus": ModelSpec(
            name="opus",
            provider="claude_cli",
            model="opus",
            tier="frontier",
            hosted="cloud",
            context_window=200_000,
            max_output_tokens=32_000,
            input_cost_per_mtok=15.0,
            output_cost_per_mtok=75.0,
            # The CLI cannot constrain decoding, so Forge falls back to
            # prompt-level schema plus validate-and-repair.
            supports_json_schema=False,
            supports_tools=False,
            supports_vision=False,
            concurrency=2,
            # The same 1800s its two siblings get. Left at the 600s default,
            # the slowest model on the ladder had a third of the wall clock of
            # the faster ones -- while being handed the hardest tasks and the
            # same 32k output budget. Observed live: four consecutive
            # `claude timed out after 600s (model='opus')`, each burning ten
            # minutes and returning nothing, classified transient and retried.
            timeout=1800.0,
            # Six an hour. The three Anthropic rungs share one subscription, so
            # this is not a cost limit but a *plan* limit: at 20/hr the top rung
            # alone exhausted the whole plan in under two hours and the run then
            # spun on rate limits for ninety minutes. Scarcity here is what makes
            # the router try haiku and sonnet, which have been landing nodes for
            # cents while opus took 94% of the spend.
            quota_per_hour=6,
        ),
    }
    return ModelsConfig(
        providers=providers,
        models=models,
        ladder=["local", "local_deep", "haiku", "sonnet", "opus"],
        default="local",
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _merge_into(target: Any, data: dict[str, Any], path: str = "") -> None:
    """Apply a TOML table onto a dataclass instance, recursing into sub-dataclasses."""
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        where = f"{path}.{key}" if path else key
        if key not in known:
            raise ConfigError(f"unknown configuration key {where!r}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into(current, value, where)
        elif key == "providers" and isinstance(value, dict):
            merged = dict(current)
            for name, spec in value.items():
                base = merged.get(name, ProviderConfig())
                _merge_into(base, spec, f"{where}.{name}")
                merged[name] = base
            setattr(target, key, merged)
        elif key == "models" and isinstance(value, dict) and path == "models":
            merged = dict(current)
            for name, spec in value.items():
                base = merged.get(name) or ModelSpec(name=name)
                _merge_into(base, dict(spec), f"{where}.{name}")
                base.name = name
                merged[name] = base
            setattr(target, key, merged)
        elif isinstance(current, Path) or (current is None and key.endswith("_dir")):
            setattr(target, key, Path(str(value)).expanduser())
        else:
            setattr(target, key, value)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


_ENV_PREFIX = "FORGE_"

#: ``FORGE_``-prefixed variables that are Forge's own plumbing rather than
#: configuration. Unknown keys under the prefix are an error -- that is what
#: turns a typo into a message instead of a silently ignored setting -- so
#: anything the platform passes to its own subprocesses has to be declared here.
_ENV_INTERNAL = frozenset({"FORGE_RUN_CLAIM"})


def _apply_env(config: Config, environ: dict[str, str]) -> None:
    """Map ``FORGE_SECTION__KEY=value`` onto the config tree.

    Double underscore separates levels, so ``FORGE_BUDGET__DAILY_COST=5`` works
    and single-underscore key names survive intact.
    """
    for raw_key, raw_value in environ.items():
        if not raw_key.startswith(_ENV_PREFIX) or raw_key in _ENV_INTERNAL:
            continue
        trail = raw_key[len(_ENV_PREFIX) :].lower().split("__")
        node: Any = config
        for part in trail[:-1]:
            if not hasattr(node, part):
                raise ConfigError(f"unknown env config path in {raw_key!r}")
            node = getattr(node, part)
        leaf = trail[-1]
        if not hasattr(node, leaf):
            raise ConfigError(f"unknown env config key in {raw_key!r}")
        setattr(node, leaf, _coerce(getattr(node, leaf), raw_value, raw_key))


def _coerce(current: Any, raw: str, key: str) -> Any:
    try:
        if isinstance(current, bool):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(current, int) and not isinstance(current, bool):
            return int(raw)
        if isinstance(current, float):
            return float(raw)
        if isinstance(current, list):
            return [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(current, Path):
            return Path(raw).expanduser()
    except ValueError as exc:
        raise ConfigError(f"cannot parse {key}={raw!r}: {exc}") from exc
    return raw


def candidate_paths(project_dir: Path) -> list[Path]:
    return [
        Path("/etc/forge") / CONFIG_FILENAME,
        Path.home() / ".config" / "forge" / CONFIG_FILENAME,
        project_dir / FORGE_DIR / CONFIG_FILENAME,
    ]


def load_config(
    project_dir: Path | str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    extra_files: list[Path] | None = None,
) -> Config:
    """Assemble the effective configuration."""
    root = Path(project_dir or Path.cwd()).expanduser().resolve()
    config = Config(project_dir=root, models=default_models())

    for path in candidate_paths(root) + list(extra_files or []):
        if path.is_file():
            _merge_into(config, _read_toml(path))
            config.sources.append(str(path))

    _apply_env(config, dict(environ if environ is not None else os.environ))

    if overrides:
        _merge_into(config, overrides)
        config.sources.append("<cli>")

    _validate(config)
    return config


def _validate(config: Config) -> None:
    models = config.models
    for rung in models.ladder:
        if rung not in models.models:
            raise ConfigError(f"ladder references undefined model {rung!r}", available=sorted(models.models))
    if models.default not in models.models:
        raise ConfigError(f"default model {models.default!r} is not defined")
    for name, provider in models.providers.items():
        if provider.kind in ("claude_cli", "codex_cli") and not provider.command:
            raise ConfigError(
                f"provider {name!r} is a CLI provider and needs a `command`",
                kind=provider.kind,
            )
        if provider.kind in ("anthropic", "openai") and not provider.api_key_env:
            raise ConfigError(
                f"provider {name!r} uses a direct API and needs `api_key_env`",
                kind=provider.kind,
                hint="or use kind = 'claude_cli' / 'codex_cli' to go through a subscription instead",
            )
    for name, spec in models.models.items():
        if spec.provider not in models.providers:
            raise ConfigError(f"model {name!r} references undefined provider {spec.provider!r}")
        if spec.hosted not in ("local", "cloud"):
            raise ConfigError(f"model {name!r}: hosted must be 'local' or 'cloud'", got=spec.hosted)
        if spec.quota_per_hour < 0:
            raise ConfigError(f"model {name!r}: quota_per_hour must not be negative")
        if spec.context_window <= spec.max_output_tokens:
            raise ConfigError(
                f"model {name!r}: context_window must exceed max_output_tokens",
                context_window=spec.context_window,
                max_output_tokens=spec.max_output_tokens,
            )
    if config.scheduler.workers < 1:
        raise ConfigError("scheduler.workers must be >= 1")
    if config.scheduler.max_attempts < 1:
        raise ConfigError("scheduler.max_attempts must be >= 1")
    if config.scheduler.max_no_progress_attempts < 1:
        raise ConfigError("scheduler.max_no_progress_attempts must be >= 1")
    if config.scheduler.max_transient_attempts < 0:
        raise ConfigError("scheduler.max_transient_attempts must be >= 0")
    if not 0.0 <= config.budget.escalation_reserve < 1.0:
        raise ConfigError("budget.escalation_reserve must be in [0, 1)")
    if not 0.0 <= config.budget.cloud_fraction_target <= 1.0:
        raise ConfigError("budget.cloud_fraction_target must be in [0, 1]")
    if not 0.0 < config.budget.max_cloud_fraction <= 1.0:
        raise ConfigError("budget.max_cloud_fraction must be in (0, 1]")
    if config.budget.cloud_fraction_target > config.budget.max_cloud_fraction:
        raise ConfigError("budget.cloud_fraction_target must not exceed max_cloud_fraction")
    if config.sandbox.kind not in ("local", "docker"):
        raise ConfigError(f"unknown sandbox.kind {config.sandbox.kind!r}")
    if config.coding.backend not in ("auto", "native", "opencode"):
        raise ConfigError(
            "coding.backend must be 'auto', 'native' or 'opencode'",
            got=config.coding.backend,
        )
    if not config.coding.opencode_command.strip():
        raise ConfigError("coding.opencode_command must not be empty")
    if not config.coding.opencode_agent.strip():
        raise ConfigError("coding.opencode_agent must not be empty")
    if config.coding.opencode_steps < 1:
        raise ConfigError("coding.opencode_steps must be >= 1")
    if config.coding.opencode_rounds < 1:
        raise ConfigError("coding.opencode_rounds must be >= 1")
    if config.coding.opencode_timeout <= 0:
        raise ConfigError("coding.opencode_timeout must be > 0")


def write_default_config(path: Path) -> None:
    """Emit a commented starter config next to a new project."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_STARTER_CONFIG, encoding="utf-8")


_STARTER_CONFIG = '''\
# Forge project configuration. Every key is optional; defaults are in
# forge/config.py. No API keys belong here or anywhere else: the frontier rungs
# use your existing `claude` and `codex` CLI logins.

log_level = "info"

[budget]
# Notional cost units, not real billing. Local models are priced low but not
# zero so the router does not treat them as unlimited; the subscription rungs
# are priced high so it prefers local. The real limit on a subscription is
# quota_per_hour on the model, not this.
total_cost = 100.0
daily_cost = 25.0
enforce_cost_limits = false # subscriptions: use cloud share and hourly quota
cloud_fraction_target = 0.20 # normal target: keep 80% of generated tokens local
max_cloud_fraction = 0.60    # hard ceiling; no cloud call may cross this
escalation_reserve = 0.25    # a quarter of the budget is escalation-only

[scheduler]
workers = 2                  # the local server has 4 slots; 2 leaves headroom
# Escalation is one rung per failure after the second attempt, so a 5-rung
# ladder needs 2 + 4 attempts before the top rung is ever served. At 5, a node
# blocked with "exhausted the 4-attempt budget" having never once been given to
# the model that would have finished it. `forge doctor` checks this.
max_attempts = 50               # catastrophic safety bound, not normal stopping
max_no_progress_attempts = 3    # same failure after all strategies => unsolvable
lease_seconds = 1800

[sandbox]
kind = "local"               # or "docker" for stronger isolation
command_timeout = 900

[coding]
# Use OpenCode's tool-driven coding loop when installed, while Forge retains
# task routing, cloud coaching, worktrees, validation and budget enforcement.
# `native` restores Forge's structured edit-plan loop.
backend = "auto"
opencode_command = "opencode"
opencode_steps = 40           # bounded inner tool loop; Forge owns outer retries
opencode_rounds = 3           # gate/repair rounds inside one node attempt
opencode_timeout = 7200       # local reasoning may legitimately take hours
fallback_to_native = true
opencode_subagents = false    # opt in only after validating nested tasks on the local backend
# opencode_server_url = "http://127.0.0.1:4096" # optional persistent server

[validation]
gates = ["format", "lint", "types", "build", "unit"]
parallel = 4
visual_tolerance = 0.005

[models]
# Cheapest first. The first two rungs are the same weights on the same local
# server: `local` runs with thinking off, `local_deep` with thinking on. That
# makes the first escalation free -- no network, no quota, no cash.
ladder = ["local", "local_deep", "haiku", "sonnet", "opus"]
default = "local"

[models.providers.local]
# Your llama.cpp (or other OpenAI-compatible) server. No auth: run it on this
# host, or on another box reachable over a private network. Overridable with
# the FORGE_LOCAL_BASE_URL environment variable.
base_url = "http://127.0.0.1:10000/v1"

[models.models.local]
# 131072 context per slot, 4 slots, vision-capable, ~85 tok/s.
context_window = 131072
max_output_tokens = 16384
concurrency = 4

[models.models.local_deep]
# Thinking is on for this rung, so it needs room: chain-of-thought is charged
# against the same output budget as the answer. Do not lower this.
max_output_tokens = 32768
concurrency = 2

[models.models.haiku]
quota_per_hour = 60          # cheap cloud diagnosis and critique

[models.models.sonnet]
quota_per_hour = 40          # stronger diagnosis or implementation

[models.models.opus]
quota_per_hour = 6           # direct solving is the final escape hatch

[deploy]
enabled = false
strategy = "static"
'''


def asdict_config(config: Config) -> dict[str, Any]:  # pragma: no cover - convenience
    return asdict(config)
