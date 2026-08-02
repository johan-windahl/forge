"""Subscription-backed CLI providers, and the local Qwen server's quirks.

No network and no subscription usage: the CLI providers are driven against a
fake executable, and the local provider against a fake HTTP client. What is
being tested is Forge's handling of each tool's real output shapes, which were
captured from live runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.config import ModelSpec, ProviderConfig, default_models
from forge.errors import (
    ConfigError,
    ModelError,
    RateLimited,
    ReasoningBudgetExhausted,
)
from forge.kernel.ledger import Ledger
from forge.models.budget import Budget
from forge.models.cli_provider import (
    ClaudeCliProvider,
    CliProvider,
    CodexCliProvider,
    cli_login_state,
)
from forge.models.policy import RoutingPolicy
from forge.models.provider import OpenAICompatProvider
from forge.models.registry import Registry
from forge.models.router import Router
from forge.models.types import Message, Request, TaskClass, TaskProfile
from forge.util.clock import ManualClock

# --------------------------------------------------------------------------
# Fake executables
# --------------------------------------------------------------------------


def _fake_cli(tmp_path: Path, name: str, body: str) -> str:
    """Write an executable shell script standing in for a real CLI."""
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


#: A real `claude -p --output-format json` response, trimmed.
CLAUDE_OK = {
    "is_error": False,
    "subtype": "success",
    "result": '{"answer": "ok"}',
    "session_id": "abc",
    "total_cost_usd": 0.0576,
    "stop_reason": "end_turn",
    "num_turns": 1,
    "usage": {
        "input_tokens": 2,
        "output_tokens": 17,
        "cache_creation_input_tokens": 8316,
        "cache_read_input_tokens": 23684,
    },
}

#: A real `codex exec --json` event stream, trimmed.
CODEX_EVENTS = "\n".join(
    [
        '{"type":"thread.started","thread_id":"t1"}',
        '{"type":"turn.started"}',
        '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"{\\"answer\\": \\"ok\\"}"}}',
        '{"type":"turn.completed","usage":{"input_tokens":15003,"cached_input_tokens":0,'
        '"output_tokens":19,"reasoning_output_tokens":4}}',
    ]
)


def _spec(name: str = "claude", **kwargs) -> ModelSpec:
    defaults = dict(provider="p", tier="frontier", hosted="cloud", timeout=30.0)
    defaults.update(kwargs)
    return ModelSpec(name=name, **defaults)


def _request(schema: dict | None = None) -> Request:
    return Request(
        messages=[Message("system", "Be terse."), Message("user", "Say ok.")],
        profile=TaskProfile(task_class=TaskClass.EXTRACTION),
        schema=schema,
    )


# --------------------------------------------------------------------------
# No API keys anywhere
# --------------------------------------------------------------------------


def test_default_roster_needs_no_api_key() -> None:
    """The whole point: frontier access without a key existing."""
    models = default_models()
    for name, provider in models.providers.items():
        assert not provider.api_key_env, f"provider {name} still wants an API key"


def test_default_ladder_is_local_first_then_subscriptions() -> None:
    models = default_models()
    assert models.ladder == ["local", "local_deep", "haiku", "sonnet", "opus"]
    assert models.models["local"].hosted == "local"
    assert models.models["local_deep"].hosted == "local"
    assert models.models["codex"].provider == "codex_cli"
    assert models.models["opus"].provider == "claude_cli"


def test_cli_providers_strip_api_keys_from_the_subprocess() -> None:
    """A stray key in the environment would silently bill instead of the plan."""
    provider = ClaudeCliProvider("claude_cli", ProviderConfig(kind="claude_cli", command="claude"))
    env = provider.env()
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["OPENAI_API_KEY"] == ""


def test_cli_providers_run_outside_the_project() -> None:
    """Otherwise the project's own CLAUDE.md would leak into every prompt."""
    provider = ClaudeCliProvider("claude_cli", ProviderConfig(kind="claude_cli", command="claude"))
    workdir = provider.working_dir()
    assert workdir.is_dir()
    assert not any(workdir.iterdir()), "the CLI working directory must be empty"


def test_direct_api_provider_still_requires_a_key() -> None:
    from forge.config import Config, ModelsConfig
    from forge.config import _validate as validate_config

    config = Config(models=ModelsConfig(
        providers={"anthropic": ProviderConfig(kind="anthropic", api_key_env="")},
        models={"m": ModelSpec(name="m", provider="anthropic")},
        ladder=["m"],
        default="m",
    ))
    with pytest.raises(ConfigError, match="api_key_env"):
        validate_config(config)


def test_cli_provider_requires_a_command() -> None:
    from forge.config import Config, ModelsConfig
    from forge.config import _validate as validate_config

    config = Config(models=ModelsConfig(
        providers={"c": ProviderConfig(kind="claude_cli", command="")},
        models={"m": ModelSpec(name="m", provider="c")},
        ladder=["m"],
        default="m",
    ))
    with pytest.raises(ConfigError, match="needs a `command`"):
        validate_config(config)


# --------------------------------------------------------------------------
# Claude CLI
# --------------------------------------------------------------------------


def test_claude_cli_parses_a_real_response(tmp_path: Path) -> None:
    command = _fake_cli(tmp_path, "claude", f"cat > /dev/null\necho '{json.dumps(CLAUDE_OK)}'\n")
    provider = ClaudeCliProvider("c", ProviderConfig(kind="claude_cli", command=command))

    completion = provider.complete(_request(), _spec())

    assert completion.text == '{"answer": "ok"}'
    # Fresh, cache-read and cache-creation were all sent to the model.
    assert completion.usage.input_tokens == 2 + 8316 + 23684
    assert completion.usage.output_tokens == 17
    assert completion.usage.cached_input_tokens == 23684
    assert completion.raw["reported_cost_usd"] == 0.0576


def test_claude_cli_disables_tools_and_project_settings(tmp_path: Path) -> None:
    """A frontier rung must be a generator, not an agent with file access."""
    command = _fake_cli(
        tmp_path,
        "claude",
        'cat > /dev/null\nprintf "%s\\n" "$@" > ' + str(tmp_path / "argv.txt") + "\n"
        f"echo '{json.dumps(CLAUDE_OK)}'\n",
    )
    provider = ClaudeCliProvider("c", ProviderConfig(kind="claude_cli", command=command))
    provider.complete(_request(), _spec(model="opus"))

    argv = (tmp_path / "argv.txt").read_text().splitlines()
    assert "--print" in argv
    assert "--strict-mcp-config" in argv
    assert "--allowed-tools" in argv
    assert "--setting-sources" in argv
    assert argv[argv.index("--model") + 1] == "opus"


def test_claude_cli_is_told_the_schema_exactly_once(config, ledger: Ledger, tmp_path: Path) -> None:
    """The shape block must reach the CLI once, not once per layer.

    ``claude`` declares no schema support, so ``ModelClient`` appends the
    instruction. The provider used to append a second copy of its own, sending
    the whole block twice on every structured frontier call. Asserting the
    count, rather than mere presence, is what makes that regression visible --
    and it has to be measured through the client, because neither layer is
    wrong on its own.
    """
    from forge.models.client import ModelClient

    command = _fake_cli(
        tmp_path,
        "claude",
        'cat > /dev/null\nprintf "%s\\n" "$@" > ' + str(tmp_path / "argv.txt") + "\n"
        f"echo '{json.dumps(CLAUDE_OK)}'\n",
    )
    provider = ClaudeCliProvider("claude_cli", ProviderConfig(kind="claude_cli", command=command))
    config.models.ladder = ["opus"]  # force the rung whose spec cannot constrain
    assert not config.models.models["opus"].supports_json_schema, \
        "precondition: the client is the layer that instructs"

    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, provider)
    client.structured(
        [Message("user", "q")],
        TaskProfile(task_class=TaskClass.EXTRACTION),
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    argv = (tmp_path / "argv.txt").read_text()
    assert argv.count("single JSON value") == 1, "the schema instruction was sent more than once"


def test_claude_cli_reports_errors(tmp_path: Path) -> None:
    payload = {"is_error": True, "subtype": "error_during_execution", "result": "something broke"}
    command = _fake_cli(tmp_path, "claude", f"cat > /dev/null\necho '{json.dumps(payload)}'\n")
    provider = ClaudeCliProvider("c", ProviderConfig(kind="claude_cli", command=command))

    with pytest.raises(ModelError, match="something broke"):
        provider.complete(_request(), _spec())


def test_plan_limit_becomes_rate_limited_not_a_node_failure(tmp_path: Path) -> None:
    """Quota exhaustion must back off without consuming the node's attempts."""
    command = _fake_cli(tmp_path, "claude", "cat > /dev/null\necho 'usage limit reached' >&2\nexit 1\n")
    provider = ClaudeCliProvider("c", ProviderConfig(kind="claude_cli", command=command))

    with pytest.raises(RateLimited):
        provider.complete(_request(), _spec())


def test_not_logged_in_is_a_config_error(tmp_path: Path) -> None:
    command = _fake_cli(tmp_path, "claude", "cat > /dev/null\necho 'Please log in' >&2\nexit 1\n")
    provider = ClaudeCliProvider("c", ProviderConfig(kind="claude_cli", command=command))

    with pytest.raises(ConfigError, match="not logged in"):
        provider.complete(_request(), _spec())


def test_missing_executable_is_unavailable() -> None:
    provider = ClaudeCliProvider("c", ProviderConfig(kind="claude_cli", command="definitely-not-installed"))
    assert not provider.available()


# --------------------------------------------------------------------------
# Codex CLI
# --------------------------------------------------------------------------


def test_codex_reads_the_answer_file(tmp_path: Path) -> None:
    """`-o` is the clean channel; the event stream is the fallback."""
    command = _fake_cli(
        tmp_path,
        "codex",
        "cat > /dev/null\n"
        'while [ $# -gt 0 ]; do if [ "$1" = "--output-last-message" ]; then echo -n \'{"answer": "ok"}\' > "$2"; fi; shift; done\n'
        f"cat <<'EOF'\n{CODEX_EVENTS}\nEOF\n",
    )
    provider = CodexCliProvider("c", ProviderConfig(kind="codex_cli", command=command))

    completion = provider.complete(_request(), _spec("codex"))

    assert completion.text == '{"answer": "ok"}'
    assert completion.usage.input_tokens == 15003
    assert completion.usage.output_tokens == 19
    assert completion.usage.reasoning_tokens == 4


def test_codex_falls_back_to_the_event_stream(tmp_path: Path) -> None:
    command = _fake_cli(tmp_path, "codex", f"cat > /dev/null\ncat <<'EOF'\n{CODEX_EVENTS}\nEOF\n")
    provider = CodexCliProvider("c", ProviderConfig(kind="codex_cli", command=command))

    assert provider.complete(_request(), _spec("codex")).text == '{"answer": "ok"}'


def test_codex_passes_a_schema_file(tmp_path: Path) -> None:
    """The one CLI rung with genuine constrained output."""
    marker = tmp_path / "schema-seen.txt"
    command = _fake_cli(
        tmp_path,
        "codex",
        "cat > /dev/null\n"
        'while [ $# -gt 0 ]; do\n'
        f'  if [ "$1" = "--output-schema" ]; then cp "$2" {marker}; fi\n'
        '  if [ "$1" = "--output-last-message" ]; then echo -n \'{"answer": "ok"}\' > "$2"; fi\n'
        "  shift\ndone\n"
        f"cat <<'EOF'\n{CODEX_EVENTS}\nEOF\n",
    )
    provider = CodexCliProvider("c", ProviderConfig(kind="codex_cli", command=command))
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    provider.complete(_request(schema=schema), _spec("codex"))

    assert json.loads(marker.read_text()) == schema


def test_codex_surfaces_stream_errors(tmp_path: Path) -> None:
    events = '{"type":"error","message":"model not supported for this account"}'
    command = _fake_cli(tmp_path, "codex", f"cat > /dev/null\necho '{events}'\n")
    provider = CodexCliProvider("c", ProviderConfig(kind="codex_cli", command=command))

    with pytest.raises(ModelError, match="not supported"):
        provider.complete(_request(), _spec("codex"))


def test_codex_cleans_up_its_temporary_files(tmp_path: Path) -> None:
    command = _fake_cli(
        tmp_path,
        "codex",
        "cat > /dev/null\n"
        'while [ $# -gt 0 ]; do if [ "$1" = "--output-last-message" ]; then echo -n "hi" > "$2"; fi; shift; done\n'
        f"cat <<'EOF'\n{CODEX_EVENTS}\nEOF\n",
    )
    provider = CodexCliProvider("c", ProviderConfig(kind="codex_cli", command=command))
    provider.complete(_request(schema={"type": "object"}), _spec("codex"))

    leftovers = list(provider.working_dir().glob("forge-*"))
    assert not leftovers, f"left behind {leftovers}"


def test_login_state_detects_a_subscription(tmp_path: Path, monkeypatch) -> None:
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": None, "auth_mode": "chatgpt", "tokens": {"a": 1}})
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")

    ok, detail = cli_login_state("codex")
    assert ok and "subscription" in detail


def test_login_state_detects_a_missing_login(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nothing"))
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")

    ok, detail = cli_login_state("codex")
    assert not ok and "login" in detail


# --------------------------------------------------------------------------
# Prompt flattening
# --------------------------------------------------------------------------


def test_messages_flatten_with_roles_preserved() -> None:
    system, conversation = CliProvider.split_messages(
        [
            Message("system", "Rule one."),
            Message("system", "Rule two."),
            Message("user", "Do the thing."),
            Message("assistant", "Here is my attempt."),
            Message("user", "That was wrong, fix it."),
        ]
    )
    assert system == "Rule one.\n\nRule two."
    assert conversation.startswith("Do the thing.")
    assert "Your previous answer" in conversation
    assert "Follow-up" in conversation


# --------------------------------------------------------------------------
# The local Qwen server's reasoning behaviour
# --------------------------------------------------------------------------


class _FakeHttp:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.payload: dict = {}

    def post_json(self, url, payload, *, headers=None, timeout=None):
        self.payload = payload
        return self.response


def _local_spec(**extra) -> ModelSpec:
    return ModelSpec(
        name="local",
        provider="local",
        tier="local",
        hosted="local",
        max_output_tokens=2048,
        supports_json_schema=True,
        extra=extra,
    )


def _local_provider(response: dict) -> tuple[OpenAICompatProvider, _FakeHttp]:
    provider = OpenAICompatProvider("local", ProviderConfig(kind="openai_compat", base_url="http://x/v1"))
    http = _FakeHttp(response)
    provider._http = http
    return provider, http


def test_thinking_toggle_is_sent_to_llama_cpp() -> None:
    """The rung's identity is this flag; it must actually reach the server."""
    provider, http = _local_provider(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}}
    )
    provider.complete(_request(), _local_spec(thinking=False))
    assert http.payload["chat_template_kwargs"] == {"enable_thinking": False}

    provider.complete(_request(), _local_spec(thinking=True))
    assert http.payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_consecutive_system_messages_are_merged() -> None:
    """Qwen's chat template raises on more than one system message.

    Forge's context builder emits three by design (role, stable prefix,
    volatile context), so without merging every local call returns 400 and
    silently escalates to a cloud rung. Found by inspecting a live run; the
    platform kept working, which is why it went unnoticed.
    """
    provider, http = _local_provider(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}}
    )
    request = Request(
        messages=[
            Message("system", "Role instructions."),
            Message("system", "Stable context."),
            Message("system", "Volatile context."),
            Message("user", "Do it."),
        ],
        profile=TaskProfile(task_class=TaskClass.EXTRACTION),
    )
    provider.complete(request, _local_spec(thinking=False))

    roles = [m["role"] for m in http.payload["messages"]]
    assert roles == ["system", "user"], f"expected one system message, got {roles}"
    assert "Role instructions." in http.payload["messages"][0]["content"]
    assert "Volatile context." in http.payload["messages"][0]["content"]


def test_merging_preserves_tool_and_assistant_structure() -> None:
    """Merging must not flatten a repair exchange or a tool result."""
    from forge.models.provider import _merge_same_role

    merged = _merge_same_role([
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "wrong"},
        {"role": "user", "content": "fix it"},
    ])
    assert [m["role"] for m in merged] == ["system", "user", "assistant", "user"]
    assert merged[0]["content"] == "a\n\nb"


def test_qwen_sampling_settings_are_forwarded() -> None:
    provider, http = _local_provider(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {}}
    )
    provider.complete(_request(), _local_spec(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0))
    assert http.payload["temperature"] == 0.7
    assert http.payload["top_p"] == 0.8
    assert http.payload["top_k"] == 20


def test_reasoning_only_response_raises_a_specific_error() -> None:
    """Empty answer plus full token count means the budget went on thinking.

    Reported as its own error so the client bumps the budget instead of
    escalating to an expensive rung to fix a number.
    """
    provider, _ = _local_provider(
        {
            "choices": [
                {
                    "message": {"content": "", "reasoning_content": "Let me think about this..." * 20},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 64},
        }
    )
    with pytest.raises(ReasoningBudgetExhausted) as exc:
        provider.complete(_request(), _local_spec(thinking=True))
    assert exc.value.context["finish_reason"] == "length"
    assert not exc.value.escalatable, "this is a budget problem, not a capability problem"


def test_reasoning_alongside_an_answer_is_kept_out_of_the_text() -> None:
    """Chain-of-thought is never the answer, but it is counted."""
    provider, _ = _local_provider(
        {
            "choices": [
                {
                    "message": {"content": '{"a": 1}', "reasoning_content": "x" * 400},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 200},
        }
    )
    completion = provider.complete(_request(), _local_spec(thinking=True))
    assert completion.text == '{"a": 1}'
    assert completion.usage.reasoning_tokens == 100
    assert completion.raw["had_reasoning"] is True


def test_client_bumps_the_budget_before_escalating(config, ledger: Ledger) -> None:
    """The recovery path, end to end through the client."""
    from forge.models.client import ModelClient

    attempts: list[int] = []

    class _Flaky(OpenAICompatProvider):
        def _complete(self, request, spec):
            budget = request.max_output_tokens or spec.max_output_tokens
            attempts.append(budget)
            if budget < 4000:
                raise ReasoningBudgetExhausted("thought too long", model=spec.name)
            from forge.models.types import Completion, Usage

            return Completion(text="done", model=spec.name, tier=spec.tier, usage=Usage(10, 5))

    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, _Flaky(name, ProviderConfig(kind="openai_compat")))

    completion = client.complete(
        [Message("user", "go")],
        TaskProfile(task_class=TaskClass.SUMMARIZATION),
        max_output_tokens=512,
    )
    assert completion.text == "done"
    assert len(attempts) > 1 and attempts[-1] > attempts[0], "budget must grow, not stay put"


# --------------------------------------------------------------------------
# Subscription quota as a routing constraint
# --------------------------------------------------------------------------


def _router(ledger: Ledger, clock: ManualClock) -> Router:
    from forge.config import BudgetConfig

    models = default_models()
    for provider in models.providers.values():
        provider.kind, provider.api_key_env, provider.command = "echo", "", "x"
    budget = Budget(BudgetConfig(total_cost=1000.0, daily_cost=1000.0, per_node_cost=1000.0), ledger, clock)
    return Router(Registry(models), RoutingPolicy(ledger, seed=5), budget)


def test_quota_is_tracked_from_the_ledger(ledger: Ledger, clock: ManualClock) -> None:
    """Not an in-process counter: a restart must not grant a fresh allowance."""
    from forge.config import BudgetConfig

    budget = Budget(BudgetConfig(), ledger, clock)
    for _ in range(3):
        budget.record(model="claude", tier="frontier", hosted="cloud", cost=0.1,
                      input_tokens=10, output_tokens=10)

    assert budget.calls_in_last_hour("claude") == 3
    assert budget.quota_remaining("claude", 25) == 22
    assert budget.quota_remaining("claude", 0) == -1, "0 means unlimited"

    clock.advance(3601)
    assert budget.calls_in_last_hour("claude") == 0, "the window is rolling"


def test_router_avoids_a_rung_that_is_out_of_quota(ledger: Ledger, clock: ManualClock) -> None:
    router = _router(ledger, clock)
    profile = TaskProfile(task_class=TaskClass.ARCHITECTURE, difficulty=0.95, stakes=0.95)

    # Burn the whole hourly allowance for the top rung.
    spec = router.registry.spec("opus")
    for _ in range(spec.quota_per_hour):
        router.budget.record(model="claude", tier="frontier", hosted="cloud", cost=0.0,
                             input_tokens=1, output_tokens=1)

    route = router.select(profile, messages=[Message("user", "design it")])
    assert route.model != "claude"


def test_quota_recovers_with_the_rolling_window(ledger: Ledger, clock: ManualClock) -> None:
    router = _router(ledger, clock)
    spec = router.registry.spec("codex")
    for _ in range(spec.quota_per_hour):
        router.budget.record(model="codex", tier="frontier", hosted="cloud", cost=0.0,
                             input_tokens=1, output_tokens=1)
    assert router.budget.quota_remaining("codex", spec.quota_per_hour) == 0

    clock.advance(3601)
    assert router.budget.quota_remaining("codex", spec.quota_per_hour) == spec.quota_per_hour


# --------------------------------------------------------------------------
# Strict structured output, and failures that retrying cannot fix
# --------------------------------------------------------------------------


def test_codex_schema_is_rewritten_into_openai_strict_form(tmp_path: Path) -> None:
    """OpenAI strict mode requires every property in `required`.

    Observed live: EDIT_PLAN_SCHEMA marks only path and op required, because a
    `delete` edit carries no content. Codex rejected every implementation
    request with `invalid_json_schema`, and the node retried 36 times.
    """
    from forge.workspace.patch import EDIT_PLAN_SCHEMA

    command = _fake_cli(
        tmp_path,
        "codex",
        'cat > /dev/null\nprintf "%s\\n" "$@" > ' + str(tmp_path / "argv.txt") + "\n"
        "for a in \"$@\"; do case \"$a\" in *forge-schema.json) cp \"$a\" "
        + str(tmp_path / "sent.json") + ";; esac; done\n"
        "for a in \"$@\"; do case \"$a\" in *forge-answer.txt) echo '{}' > \"$a\";; esac; done\n",
    )
    provider = CodexCliProvider("c", ProviderConfig(kind="codex_cli", command=command))
    provider.complete(_request(schema=EDIT_PLAN_SCHEMA), _spec(name="codex"))

    sent = json.loads((tmp_path / "sent.json").read_text())
    item = sent["properties"]["edits"]["items"]
    assert set(item["required"]) == set(item["properties"]), \
        "strict mode rejects a schema whose properties are not all required"
    assert "null" in item["properties"]["content"]["type"], \
        "optionality must survive as a nullable type, not as omission"
    assert item["properties"]["op"]["enum"], "the enum constraint must not be lost"


def test_the_original_schema_keeps_its_real_optionality() -> None:
    """The transform is for codex only; Forge's own contract must not widen."""
    from forge.models.structured import strict_schema
    from forge.workspace.patch import EDIT_PLAN_SCHEMA

    strict_schema(EDIT_PLAN_SCHEMA)
    assert EDIT_PLAN_SCHEMA["properties"]["edits"]["items"]["required"] == ["path", "op"]


def test_a_malformed_request_is_terminal_not_a_retry_loop(tmp_path: Path) -> None:
    """`invalid_json_schema` will fail identically forever; retrying burns quota."""
    from forge.errors import ConfigError

    body = '{"type":"error","message":"invalid_request_error: invalid_json_schema"}'
    command = _fake_cli(tmp_path, "codex", f"cat > /dev/null\necho '{body}' >&2\nexit 1\n")
    provider = CodexCliProvider("c", ProviderConfig(kind="codex_cli", command=command))

    with pytest.raises(ConfigError):
        provider.complete(_request(), _spec(name="codex"))
