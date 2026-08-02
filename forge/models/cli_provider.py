"""Frontier models through their own CLIs, on a subscription.

``claude -p`` and ``codex exec`` both run non-interactively and both
authenticate with the login the operator already has. Driving them as
completion engines means Forge reaches frontier models without an API key
existing anywhere -- no ``ANTHROPIC_API_KEY``, no ``OPENAI_API_KEY``, nothing to
leak, rotate or forget to set.

The tradeoffs are real and worth stating plainly:

**Fixed overhead per call.** Both CLIs load their own agent harness prompt
before seeing ours. Measured: ~23k input tokens for ``claude -p``, ~15k for
``codex exec``, on a request whose actual content was six words. On a
subscription that is quota rather than cash, but it is the reason these rungs
sit at the top of the ladder and are reached only when the local model has
actually failed. Nothing Forge sends can reduce it -- ``--system-prompt`` does
not replace the harness prompt, and ``--bare`` would but forces API-key auth,
which is precisely what we are avoiding.

**No streaming, no tool protocol.** Forge does not need either from these rungs;
it needs one well-reasoned answer.

**Structured output differs.** ``codex exec --output-schema`` constrains the
final message to a JSON Schema, so ``codex`` advertises schema support. The
Claude CLI has no equivalent, so ``claude`` declares none and falls back to
Forge's prompt-level schema plus validate-and-repair.

Both are run with tools disabled, MCP servers off, project settings ignored and
a neutral working directory. A CLI that discovered the project's own
``CLAUDE.md`` or ``AGENTS.md`` would be injecting context Forge did not choose,
which would quietly defeat the whole context-management design.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..config import ModelSpec, ProviderConfig
from ..errors import (
    ConfigError,
    MalformedOutput,
    ModelError,
    ProviderUnavailable,
    RateLimited,
)
from ..obs.log import get_logger
from ..util.clock import Clock
from ..util.proc import CommandTimeout, ProcResult, run
from .provider import Provider
from .structured import strict_schema
from .types import Completion, Message, Request, Usage

log = get_logger("models.cli")

#: Substrings in CLI output that mean "you have hit a plan limit". Matching on
#: text is unpleasant, but these tools report quota exhaustion as prose and the
#: distinction matters: a quota failure should reroute, not fail the node.
_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "usage limit",
    "quota",
    "too many requests",
    "429",
    "resets at",
    "try again later",
)

#: Failures that are a property of the request, not of the service. Retrying
#: these is pure waste: nothing about the next attempt will differ.
_PERMANENT_MARKERS = (
    "invalid_request_error",
    "invalid_json_schema",
    "invalid schema for response_format",
    "unsupported_parameter",
    "unknown_parameter",
    "context_length_exceeded",
    "string_above_max_length",
)

_AUTH_MARKERS = (
    "not logged in",
    "please log in",
    "authentication",
    "unauthorized",
    "credentials",
    "/login",
)


class CliProvider(Provider):
    """Shared machinery for CLI-backed providers."""

    kind = "cli"
    #: Argument that makes the tool read the prompt from stdin, if it needs one.
    stdin_marker: str | None = None

    def __init__(self, name: str, config: ProviderConfig, clock: Clock | None = None) -> None:
        super().__init__(name, config, clock)
        self._scratch: Path | None = None

    # -- availability ----------------------------------------------------

    def executable(self) -> str:
        return self.config.command or self.kind.replace("_cli", "")

    def available(self) -> bool:
        """Is the CLI installed? Login state is checked by ``forge doctor``.

        Deliberately not shelling out here: ``available`` is called during
        routing, potentially many times a minute, and spawning a process to
        answer it would be a real cost for a question that rarely changes.
        """
        return shutil.which(self.executable()) is not None

    def api_key(self) -> str:
        """CLI providers never use an API key. Overridden to make that explicit."""
        return ""

    # -- working directory -----------------------------------------------

    def working_dir(self) -> Path:
        """A neutral, empty directory for the subprocess.

        Both CLIs read instruction files from their working directory. Running
        them inside the project would silently merge the project's own agent
        instructions into every Forge prompt.
        """
        if self.config.cwd:
            path = Path(self.config.cwd).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            return path
        if self._scratch is None:
            self._scratch = Path(tempfile.mkdtemp(prefix="forge-cli-"))
        return self._scratch

    def env(self) -> dict[str, str]:
        """Environment for the subprocess.

        Any ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` in the parent
        environment is *removed*, not passed through. If one is present, these
        CLIs may prefer it over the subscription login and bill the key
        instead -- which is the exact outcome this provider exists to avoid.
        """
        environment = {
            "CI": "1",
            "TERM": "dumb",
            "NO_COLOR": "1",
            "ANTHROPIC_API_KEY": "",
            "OPENAI_API_KEY": "",
        }
        return environment

    # -- prompt flattening -----------------------------------------------

    @staticmethod
    def split_messages(messages: list[Message]) -> tuple[str, str]:
        """Split into (system text, conversation text).

        These CLIs take one prompt, not a message list. Roles are rendered as
        labelled blocks so a multi-turn repair exchange survives the flattening
        legibly.
        """
        system_parts: list[str] = []
        turns: list[str] = []
        for message in messages:
            content = (message.content or "").strip()
            if not content:
                continue
            if message.role == "system":
                system_parts.append(content)
            elif message.role == "user":
                turns.append(content if len(turns) == 0 else f"## Follow-up\n\n{content}")
            elif message.role == "assistant":
                turns.append(f"## Your previous answer\n\n{content}")
            elif message.role == "tool":
                turns.append(f"## Tool result\n\n{content}")
        return "\n\n".join(system_parts), "\n\n".join(turns)

    # -- execution -------------------------------------------------------

    def _invoke(self, argv: list[str], stdin: str, spec: ModelSpec) -> ProcResult:
        try:
            result = run(
                argv,
                cwd=self.working_dir(),
                env=self.env(),
                timeout=spec.timeout or self.config.timeout,
                stdin=stdin,
                # These prompts and answers are large; do not clip them to the
                # default limit and lose the answer.
                output_limit=8 * 1024 * 1024,
            )
        except CommandTimeout as exc:
            raise ProviderUnavailable(
                f"{self.executable()} timed out after {spec.timeout:.0f}s", model=spec.name
            ) from exc
        except OSError as exc:
            raise ConfigError(
                f"cannot run {self.executable()}: {exc}",
                hint=f"is {self.executable()} installed and on PATH?",
            ) from exc
        self._classify_failure(result, spec)
        return result

    def _classify_failure(self, result: ProcResult, spec: ModelSpec) -> None:
        """Turn CLI failure text into the right error type.

        Getting this right is what lets the client reroute on a quota failure
        instead of burning the node's attempts on a wall it cannot get past.
        """
        if result.ok:
            return
        blob = result.combined.lower()
        if any(marker in blob for marker in _RATE_LIMIT_MARKERS):
            raise RateLimited(
                f"{self.executable()} reports a plan/rate limit",
                model=spec.name,
                detail=result.tail(6),
            )
        if any(marker in blob for marker in _AUTH_MARKERS):
            raise ConfigError(
                f"{self.executable()} is not logged in",
                model=spec.name,
                hint=f"run `{self.executable()} login` (or `{self.executable()}` once interactively)",
                detail=result.tail(6),
            )
        # A rejected request is deterministic: the same bytes will be rejected
        # the same way forever. Reported as ProviderUnavailable it read as a
        # transient outage, and the node retried on a ~20s backoff indefinitely
        # -- observed live, 36 identical failures over a malformed schema.
        # ConfigError is terminal, which surfaces the real problem instead.
        if any(marker in blob for marker in _PERMANENT_MARKERS):
            raise ConfigError(
                f"{self.executable()} rejected the request as malformed; retrying cannot help",
                model=spec.name,
                detail=result.tail(10),
            )
        raise ProviderUnavailable(
            f"{self.executable()} exited {result.returncode}",
            model=spec.name,
            detail=result.tail(10),
        )


class ClaudeCliProvider(CliProvider):
    """Claude Code in print mode, using the operator's Claude subscription."""

    kind = "claude_cli"

    def _complete(self, request: Request, spec: ModelSpec) -> Completion:
        system_text, conversation = self.split_messages(request.messages)

        # No schema instruction is added here. This spec declares
        # ``supports_json_schema=False``, so ``ModelClient._attempt`` has
        # already appended one to the messages; adding a second copy sent the
        # entire shape block twice in every structured call. Codex takes the
        # other path -- it constrains for real via ``--output-schema`` -- so
        # neither provider needs to instruct on its own behalf.

        argv = [
            self.executable(),
            "--print",
            "--output-format",
            "json",
            # A pure generator: no tools, no MCP, no project settings, no
            # CLAUDE.md discovery beyond the empty working directory.
            "--allowed-tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--permission-mode",
            "manual",
        ]
        if spec.model:
            argv += ["--model", spec.model]
        if system_text:
            # Append rather than replace: the harness prompt is loaded either
            # way, and replacing it can confuse the CLI's own behaviour.
            argv += ["--append-system-prompt", system_text]
        argv += list(self.config.args)

        result = self._invoke(argv, conversation or " ", spec)
        return self._decode(result, spec)

    def _decode(self, result: ProcResult, spec: ModelSpec) -> Completion:
        try:
            data = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError as exc:
            raise MalformedOutput(
                "claude CLI did not return JSON",
                model=spec.name,
                preview=result.stdout[:400],
            ) from exc

        if data.get("is_error") or data.get("subtype") not in (None, "success"):
            message = str(data.get("result") or data.get("error") or "unknown error")
            if any(marker in message.lower() for marker in _RATE_LIMIT_MARKERS):
                raise RateLimited(f"claude CLI: {message[:200]}", model=spec.name)
            raise ModelError(f"claude CLI reported an error: {message[:300]}", model=spec.name)

        text = data.get("result") or ""
        raw_usage = data.get("usage") or {}
        usage = Usage(
            # The CLI reports fresh, cache-read and cache-creation separately;
            # all three were sent to the model, so all three are input.
            input_tokens=int(raw_usage.get("input_tokens", 0))
            + int(raw_usage.get("cache_read_input_tokens", 0))
            + int(raw_usage.get("cache_creation_input_tokens", 0)),
            output_tokens=int(raw_usage.get("output_tokens", 0)),
            cached_input_tokens=int(raw_usage.get("cache_read_input_tokens", 0)),
        )
        return Completion(
            text=text,
            model=spec.name,
            tier=spec.tier,
            usage=usage,
            finish_reason=str(data.get("stop_reason") or "stop"),
            raw={
                "session_id": data.get("session_id"),
                # What this would have cost on the API. Informational only:
                # subscription usage is governed by quota, not by this number.
                "reported_cost_usd": data.get("total_cost_usd"),
                "num_turns": data.get("num_turns"),
            },
        )


class CodexCliProvider(CliProvider):
    """Codex in exec mode, using the operator's ChatGPT subscription."""

    kind = "codex_cli"

    def _complete(self, request: Request, spec: ModelSpec) -> Completion:
        system_text, conversation = self.split_messages(request.messages)
        prompt = f"{system_text}\n\n---\n\n{conversation}".strip() if system_text else conversation

        workdir = self.working_dir()
        answer_path = workdir / "forge-answer.txt"
        answer_path.unlink(missing_ok=True)
        schema_path: Path | None = None

        argv = [
            self.executable(),
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",  # do not accumulate session files over a long run
            "--sandbox",
            "read-only",
            "--json",
            "--output-last-message",
            str(answer_path),
        ]
        if spec.model:
            argv += ["--model", spec.model]
        if request.schema:
            # Native constrained final message. The one place a CLI rung gives
            # a stronger guarantee than the prompt-level fallback.
            schema_path = workdir / "forge-schema.json"
            schema_path.write_text(json.dumps(strict_schema(request.schema)), encoding="utf-8")
            argv += ["--output-schema", str(schema_path)]
        argv += list(self.config.args)
        argv.append("-")  # read the prompt from stdin

        try:
            result = self._invoke(argv, prompt or " ", spec)
            text = self._read_answer(answer_path, result, spec)
            usage = self._read_usage(result)
        finally:
            answer_path.unlink(missing_ok=True)
            if schema_path is not None:
                schema_path.unlink(missing_ok=True)

        return Completion(
            text=text,
            model=spec.name,
            tier=spec.tier,
            usage=usage,
            finish_reason="stop",
            raw={"events": result.stdout.count("\n")},
        )

    def _read_answer(self, answer_path: Path, result: ProcResult, spec: ModelSpec) -> str:
        """Prefer the file; fall back to parsing the event stream."""
        if answer_path.is_file():
            text = answer_path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text
        for event in self._events(result):
            if event.get("type") == "error":
                message = str(event.get("message", ""))
                if any(marker in message.lower() for marker in _RATE_LIMIT_MARKERS):
                    raise RateLimited(f"codex CLI: {message[:200]}", model=spec.name)
                raise ModelError(f"codex CLI reported an error: {message[:300]}", model=spec.name)
        messages = [
            item.get("text", "")
            for event in self._events(result)
            if (item := event.get("item", {})).get("type") == "agent_message"
        ]
        if messages:
            return messages[-1].strip()
        raise MalformedOutput(
            "codex CLI produced no final message", model=spec.name, preview=result.tail(10)
        )

    def _read_usage(self, result: ProcResult) -> Usage:
        for event in reversed(self._events(result)):
            if event.get("type") == "turn.completed":
                raw = event.get("usage") or {}
                return Usage(
                    input_tokens=int(raw.get("input_tokens", 0)),
                    output_tokens=int(raw.get("output_tokens", 0)),
                    cached_input_tokens=int(raw.get("cached_input_tokens", 0)),
                    reasoning_tokens=int(raw.get("reasoning_output_tokens", 0)),
                )
        return Usage()

    @staticmethod
    def _events(result: ProcResult) -> list[dict[str, Any]]:
        """Parse the JSONL event stream, ignoring non-JSON chatter."""
        events: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
        return events


def cli_login_state(command: str) -> tuple[bool, str]:
    """Best-effort check that a CLI is installed and logged in.

    Used by ``forge doctor``. Deliberately does not make a model call: the point
    is to tell the operator whether a long unattended run will be able to reach
    its top rungs, not to spend quota proving it.
    """
    if shutil.which(command) is None:
        return False, f"{command} is not on PATH"

    if command == "codex":
        auth = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json"
        if not auth.is_file():
            return False, f"not logged in (run `{command} login`)"
        try:
            data = json.loads(auth.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False, "auth file is unreadable"
        if data.get("tokens"):
            mode = data.get("auth_mode") or "chatgpt"
            return True, f"logged in via {mode} subscription"
        if data.get("OPENAI_API_KEY"):
            return True, "logged in with an API key (subscription preferred)"
        return False, f"not logged in (run `{command} login`)"

    if command == "claude":
        # Claude Code stores credentials in a keychain or a credentials file
        # depending on platform, so presence of the config dir is the honest
        # signal available without spending a call.
        home = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
        if home.is_dir() or (Path.home() / ".claude.json").is_file():
            return True, "CLI present and configured"
        return False, f"no configuration found (run `{command}` once to log in)"

    return True, f"{command} is on PATH"
