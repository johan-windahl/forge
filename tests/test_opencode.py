"""The OpenCode boundary: local-only config, durable sessions and accounting."""

from __future__ import annotations

import json
from pathlib import Path

from forge.cli import main
from forge.config import Config, default_models, load_config
from forge.execution.opencode import OpenCodeExecutor
from forge.util.proc import ProcResult
from forge.workspace.sandbox import LocalSandbox


def _executor(
    tmp_path: Path, model_name: str = "local"
) -> tuple[OpenCodeExecutor, LocalSandbox]:
    config = Config(project_dir=tmp_path, models=default_models())
    config.state_dir = tmp_path / "state"
    config.ensure_dirs()
    root = tmp_path / "worktree"
    root.mkdir()
    sandbox = LocalSandbox(config.sandbox, root)
    return (
        OpenCodeExecutor(
            config,
            sandbox,
            node_id="node/one",
            model_name=model_name,
        ),
        sandbox,
    )


def test_json_events_preserve_real_local_usage() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "text", "sessionID": "ses_1", "text": "working"}),
            json.dumps(
                {
                    "type": "step_finish",
                    "sessionID": "ses_1",
                    "part": {
                        "tokens": {
                            "input": 1200,
                            "output": 80,
                            "reasoning": 20,
                            "cache": {"read": 300},
                        }
                    },
                }
            ),
            json.dumps({"type": "text", "sessionID": "ses_1", "text": "Implemented it."}),
        ]
    )

    result = OpenCodeExecutor.parse_events(raw, prompt="do work")

    assert result.ok
    assert result.session_id == "ses_1"
    assert result.summary == "Implemented it."
    assert result.usage.input_tokens == 1200
    assert result.usage.generated_tokens == 100
    assert result.usage.cached_tokens == 300
    assert result.usage.measured


def test_old_event_streams_get_a_conservative_usage_estimate() -> None:
    raw = json.dumps({"type": "text", "sessionID": "ses_old", "text": "done"})
    result = OpenCodeExecutor.parse_events(raw, prompt="implement this task")

    assert not result.usage.measured
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


def test_step_limit_result_starts_the_next_round_in_a_fresh_session(
    tmp_path: Path,
) -> None:
    executor, sandbox = _executor(tmp_path)
    executor._save_session("ses_maxed_out")
    stdout = json.dumps(
        {
            "type": "text",
            "sessionID": "ses_maxed_out",
            "text": "CRITICAL - MAXIMUM STEPS REACHED",
        }
    )
    sandbox.exec = lambda *_args, **_kwargs: ProcResult(  # type: ignore[method-assign]
        [], 0, stdout, "", 0.1
    )

    result = executor.execute("repair the remaining lint failure")

    assert result.step_limit_reached
    assert result.session_id == "ses_maxed_out"
    assert not executor.session_path.exists()


def test_inline_configuration_cannot_select_a_cloud_provider(tmp_path: Path) -> None:
    executor, _sandbox = _executor(tmp_path)
    executor.config.coding.opencode_subagents = True
    config = executor.configuration()
    agent = config["agent"]["forge-local"]

    assert config["enabled_providers"] == ["forge-local"]
    assert config["model"].startswith("forge-local/")
    assert config["small_model"] == config["model"]
    assert config["share"] == "disabled"
    assert list(config["provider"]) == ["forge-local"]
    assert agent["permission"]["task"] == "allow"
    assert agent["permission"]["external_directory"] == "deny"
    assert agent["permission"]["webfetch"] == "deny"
    assert agent["permission"]["websearch"] == "deny"
    assert agent["permission"]["doom_loop"] == "deny"
    assert agent["permission"]["bash"]["git *"] == "deny"
    assert agent["permission"]["bash"]["git diff*"] == "allow"
    assert "Keep each response compact" in agent["prompt"]
    assert "Verify a suspected cause" in agent["prompt"]
    assert config["agent"]["forge-scout"]["mode"] == "subagent"
    assert config["agent"]["forge-scout"]["permission"]["edit"] == "deny"
    assert config["agent"]["forge-critic"]["permission"]["edit"] == "deny"
    assert "actual diff" in config["agent"]["forge-critic"]["prompt"]
    assert "untracked status as a defect" in config["agent"]["forge-critic"]["prompt"]
    assert "task paths and filenames override" in config["agent"]["forge-critic"]["prompt"]


def test_nested_subagents_are_disabled_by_default(tmp_path: Path) -> None:
    executor, _sandbox = _executor(tmp_path)

    config = executor.configuration()

    assert config["agent"]["forge-local"]["permission"]["task"] == "deny"
    assert "forge-scout" not in config["agent"]
    assert "forge-critic" not in config["agent"]


def test_deep_local_model_keeps_thinking_inside_opencode(tmp_path: Path) -> None:
    executor, _sandbox = _executor(tmp_path, "local_deep")
    argv = executor._argv("integrate the feature", session_id="")
    prompt = argv[argv.index("run") + 1]

    assert prompt.endswith("/think")


def test_session_id_is_reused_on_the_next_round(tmp_path: Path) -> None:
    executor, sandbox = _executor(tmp_path)
    calls: list[list[str]] = []

    def fake_exec(argv, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(argv))
        session = "ses_durable"
        stdout = "\n".join(
            [
                json.dumps({"type": "text", "sessionID": session, "text": "done"}),
                json.dumps(
                    {
                        "type": "step_finish",
                        "sessionID": session,
                        "part": {"tokens": {"input": 10, "output": 4}},
                    }
                ),
            ]
        )
        return ProcResult(list(argv), 0, stdout, "", 0.2)

    sandbox.exec = fake_exec  # type: ignore[method-assign]
    first = executor.execute("first")
    second = executor.execute("repair")

    assert first.session_id == "ses_durable"
    assert second.session_id == "ses_durable"
    assert "--session" not in calls[0]
    session_index = calls[1].index("--session")
    assert calls[1][session_index + 1] == "ses_durable"
    saved = json.loads(executor.session_path.read_text())
    assert saved["session_id"] == "ses_durable"
    assert saved["attempt"] == 1
    assert saved["worktree_root"] == str(sandbox.root)


def test_fresh_repair_session_forgets_history_but_preserves_worktree(
    tmp_path: Path,
) -> None:
    executor, sandbox = _executor(tmp_path)
    implementation = sandbox.root / "implementation.ts"
    implementation.write_text("export const ready = true;\n")
    executor._save_session("ses_failed_validation")

    executor.start_fresh_session()

    assert not executor.session_path.exists()
    assert implementation.read_text() == "export const ready = true;\n"


def test_opencode_always_gets_absolute_local_worktree(tmp_path: Path) -> None:
    executor, sandbox = _executor(tmp_path)

    argv = executor._argv("work")

    assert argv[argv.index("--dir") + 1] == str(sandbox.root)


def test_legacy_or_wrong_worktree_session_is_not_reused(tmp_path: Path) -> None:
    executor, _sandbox = _executor(tmp_path)
    executor.session_path.parent.mkdir(parents=True, exist_ok=True)
    executor.session_path.write_text(
        json.dumps({"session_id": "ses_wrong", "node_id": executor.node_id})
    )

    assert executor._load_session() == ""


def test_session_is_not_reused_across_scheduler_attempts(tmp_path: Path) -> None:
    first, sandbox = _executor(tmp_path)
    first._save_session("ses_stale_failure_context")
    retry = OpenCodeExecutor(
        first.config,
        sandbox,
        node_id=first.node_id,
        model_name=first.model_name,
        attempt=2,
    )

    assert retry._load_session() == ""


def test_server_url_uses_the_headless_service(tmp_path: Path) -> None:
    executor, _sandbox = _executor(tmp_path)
    executor.config.coding.opencode_server_url = "http://127.0.0.1:4096"
    argv = executor._argv("work")

    assert argv[argv.index("--attach") + 1] == "http://127.0.0.1:4096"


def test_qwen_fast_rung_keeps_thinking_disabled(tmp_path: Path) -> None:
    executor, _sandbox = _executor(tmp_path)
    argv = executor._argv("implement this")

    assert argv[3].endswith("/no_think")


def test_coding_backend_configuration_is_validated(tmp_path: Path) -> None:
    config = load_config(
        tmp_path,
        overrides={
            "coding": {
                "backend": "opencode",
                "opencode_steps": 25,
                "opencode_rounds": 4,
            }
        },
        environ={},
    )
    assert config.coding.backend == "opencode"
    assert config.coding.opencode_steps == 25
    assert config.coding.opencode_rounds == 4


def test_cli_writes_a_server_config_with_the_same_local_boundary(
    tmp_path: Path,
) -> None:
    (tmp_path / ".forge").mkdir()

    assert main(["--dir", str(tmp_path), "opencode-config", "--write"]) == 0

    path = tmp_path / ".forge" / "opencode" / "server.json"
    rendered = json.loads(path.read_text())
    assert rendered["enabled_providers"] == ["forge-local"]
    assert rendered["share"] == "disabled"


def test_a_stubbed_provider_makes_opencode_unavailable(tmp_path: Path) -> None:
    """The no-real-inference contract is enforced in the provider layer, and
    OpenCode does not go through it.

    The test fixtures and `--dry-run` both express "no real model" by swapping
    every provider for the echo stub. This executor reaches its model directly,
    so it walked straight past that and drove the operator's actual local
    server: seven coding tests were timing out against a real Qwen instance.
    """
    executor, _ = _executor(tmp_path)
    executor.provider.kind = "echo"

    assert executor.available() is False
