"""A local-only OpenCode execution backend.

OpenCode owns the inner coding loop: inspect files, edit them, run focused
commands and compact its session. Forge remains the control plane around it:
the task graph, node worktree, independent gates, cloud coaching, accounting
and commits are deliberately outside this adapter.

The adapter uses ``opencode run --format json`` instead of scraping its human
terminal UI. Sessions are persisted per Forge node so a later repair attempt
continues with the tool history that produced the current worktree.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import ForgeError
from ..models.types import estimate_tokens
from ..obs.log import get_logger
from ..workspace.sandbox import Sandbox

log = get_logger("execution.opencode")

_PROVIDER_ID = "forge-local"
_STEP_LIMIT_REACHED = re.compile(
    r"maximum(?: number of)? steps(?: allowed[^\n]*)? (?:has been )?reached",
    re.IGNORECASE,
)


@dataclass(slots=True)
class OpenCodeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    steps: int = 0
    measured: bool = False

    @property
    def generated_tokens(self) -> int:
        # Reasoning is generated work and some OpenAI-compatible servers report
        # it separately rather than including it in output.
        return self.output_tokens + self.reasoning_tokens


@dataclass(slots=True)
class OpenCodeResult:
    ok: bool
    summary: str = ""
    session_id: str = ""
    error: str = ""
    usage: OpenCodeUsage = field(default_factory=OpenCodeUsage)
    duration: float = 0.0
    returncode: int = 0
    step_limit_reached: bool = False


class OpenCodeExecutor:
    """Invoke OpenCode against exactly one local Forge model and worktree."""

    def __init__(
        self,
        config: Config,
        sandbox: Sandbox,
        *,
        node_id: str,
        model_name: str,
        attempt: int = 1,
    ) -> None:
        self.config = config
        self.sandbox = sandbox
        self.node_id = node_id
        self.model_name = model_name
        self.attempt = attempt
        self.spec = config.models.models[model_name]
        self.provider = config.models.providers[self.spec.provider]

    @property
    def session_path(self) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.node_id)
        return self.config.forge_dir / "opencode" / "sessions" / f"{safe}.json"

    def available(self) -> bool:
        """Check the executable in the actual sandbox, including Docker images."""
        # A stubbed provider means the caller asked for no real inference: the
        # test suite, or `forge run --dry-run`. OpenCode reaches its model
        # directly and would otherwise walk straight past that substitution,
        # because the swap happens in the provider layer and this executor does
        # not go through it. Honour the contract here, where it is visible.
        if self.provider.kind == "echo":
            return False
        try:
            result = self.sandbox.exec(
                [self.config.coding.opencode_command, "--version"],
                timeout=min(15.0, self.config.coding.opencode_timeout),
            )
        except ForgeError:
            return False
        return result.ok

    def start_fresh_session(self) -> None:
        """Forget conversation history while preserving the node worktree.

        A repair round already receives the independent gate report, stronger
        advice and current project memory in its prompt. Replaying the entire
        tool transcript can turn a tiny lint or review repair into tens of
        thousands of input tokens and make the local model construct oversized
        tool calls. The implementation itself remains safely in the worktree.
        """
        self._clear_session()

    def execute(self, prompt: str) -> OpenCodeResult:
        session_id = self._load_session()
        argv = self._argv(prompt, session_id=session_id)
        env = {
            "OPENCODE_CONFIG_CONTENT": json.dumps(
                self.configuration(), separators=(",", ":")
            ),
            "OPENCODE_AUTO_SHARE": "false",
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
        }

        try:
            proc = self.sandbox.exec(
                argv,
                env=env,
                timeout=self.config.coding.opencode_timeout,
                # JSON tool events are intentionally verbose. Losing middle
                # step_finish events to the generic 256 KiB command cap would
                # undercount local generation and distort cloud-share control.
                output_limit=8 * 1024 * 1024,
            )
        except ForgeError as exc:
            return OpenCodeResult(
                ok=False,
                session_id=session_id,
                error=str(exc),
                returncode=-1,
            )

        parsed = self.parse_events(proc.stdout, prompt=prompt)
        parsed.duration = proc.duration
        parsed.returncode = proc.returncode
        if parsed.step_limit_reached:
            # OpenCode unlocks tools on a new user turn, but resuming the same
            # maxed-out session also replays its entire (often hundreds of KiB)
            # tool history. The worktree already contains the useful edits, so
            # make the coached repair round inspect that state from a compact,
            # fresh session instead.
            self._clear_session()
        elif parsed.session_id:
            self._save_session(parsed.session_id)
        elif session_id:
            parsed.session_id = session_id

        if not proc.ok:
            parsed.ok = False
            parsed.error = parsed.error or proc.tail(50) or (
                f"OpenCode exited with status {proc.returncode}"
            )
        return parsed

    def _argv(self, prompt: str, *, session_id: str = "") -> list[str]:
        model = f"{_PROVIDER_ID}/{self.spec.model or self.spec.name}"
        if self.spec.extra.get("thinking") is False:
            # OpenCode V1 cannot add arbitrary llama.cpp request-body fields.
            # Qwen's documented turn-level soft switch is the portable way to
            # retain the fast rung's no-thinking semantics through its tool
            # loop. Preserve explicit thinking for the deep rung in OpenCode.
            prompt = prompt.rstrip() + "\n\n/no_think"
        elif self.spec.extra.get("thinking") is True:
            # Qwen's turn-level switch preserves the deep rung through
            # OpenCode's generic compatible provider while giving it the
            # file/search/test tools broad integration work requires.
            prompt = prompt.rstrip() + "\n\n/think"
        # Never pass a relative directory to OpenCode. OpenCode persists the
        # directory on its session and may resolve "." against the long-lived
        # server/launcher directory rather than this subprocess' cwd. That can
        # silently send edits to Forge's own checkout instead of the node
        # worktree. Docker always mounts the sandbox root at /work.
        directory = str(self.sandbox.root) if self.sandbox.kind == "local" else "/work"
        argv = [
            self.config.coding.opencode_command,
            "--pure",
            "run",
            prompt,
            "--format",
            "json",
            "--model",
            model,
            "--agent",
            self.config.coding.opencode_agent,
            "--title",
            f"Forge {self.node_id}",
            "--dir",
            directory,
            "--auto",
        ]
        if session_id:
            argv.extend(["--session", session_id])
        if self.config.coding.opencode_server_url:
            argv.extend(["--attach", self.config.coding.opencode_server_url])
        return argv

    def configuration(self) -> dict[str, Any]:
        """A fail-closed OpenCode config with no cloud provider available."""
        model_id = self.spec.model or self.spec.name
        options: dict[str, Any] = {
            "baseURL": self.provider.base_url,
            "timeout": int(max(self.provider.timeout, self.spec.timeout) * 1000),
        }
        if self.provider.api_key_env:
            options["apiKey"] = f"{{env:{self.provider.api_key_env}}}"
        if self.provider.headers:
            options["headers"] = dict(self.provider.headers)

        # Subagents inherit the primary local model, while the provider
        # allowlist below makes cloud selection impossible.
        task_permission = "allow" if self.config.coding.opencode_subagents else "deny"

        bash = {
            "*": "allow",
            # Forge alone owns the index, branches and integration. Read-only
            # Git inspection remains useful to the coding agent.
            "git *": "deny",
            "git status*": "allow",
            "git diff*": "allow",
            "git log*": "allow",
            "git show*": "allow",
            "git grep*": "allow",
            "git rev-parse*": "allow",
            "rm -rf *": "deny",
            "rm -fr *": "deny",
            "shutdown*": "deny",
            "reboot*": "deny",
            "mkfs*": "deny",
        }
        permission: dict[str, Any] = {
            "read": "allow",
            "edit": "allow",
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
            "bash": bash,
            "lsp": "allow",
            "todowrite": "allow",
            # Repeating the exact same tool call is not persistence; it is a
            # stuck trajectory. Deny OpenCode's loop permission so the model
            # must change approach or return instead of burning an unattended
            # run on identical greps/tests for hours.
            "doom_loop": "deny",
            "task": task_permission,
            "external_directory": "deny",
            "webfetch": "deny",
            "websearch": "deny",
            "skill": "deny",
            "question": "deny",
        }
        read_only_bash = {
            "*": "deny",
            "pwd": "allow",
            "ls*": "allow",
            "find *": "allow",
            "rg *": "allow",
            "grep *": "allow",
            "sed -n *": "allow",
            "head *": "allow",
            "tail *": "allow",
            "git status*": "allow",
            "git diff*": "allow",
            "git log*": "allow",
            "git show*": "allow",
            "git grep*": "allow",
            "npm test*": "allow",
            "npm run test*": "allow",
            "npm run lint*": "allow",
            "npm run typecheck*": "allow",
            "npm run build*": "allow",
        }
        read_only_permission = {
            **permission,
            "edit": "deny",
            "task": "deny",
            "todowrite": "deny",
            "bash": read_only_bash,
        }
        model_ref = f"{_PROVIDER_ID}/{model_id}"
        agents: dict[str, Any] = {
            self.config.coding.opencode_agent: {
                "description": "Local implementation executor controlled by Forge",
                "mode": "primary",
                "model": model_ref,
                "steps": self.config.coding.opencode_steps,
                "temperature": self.spec.extra.get("temperature", 0.2),
                "permission": permission,
                "prompt": (
                    "Work autonomously inside the current Git worktree. Inspect the "
                    "repository, make the smallest complete implementation, and run "
                    "focused checks. Do not commit, switch branches, alter Git history, "
                    "access external directories, use the web, or merely describe code "
                    "that still needs to be written. Keep each response compact and call "
                    "a read, search, edit, or test tool early instead of spending a full "
                    "response narrating possibilities. Verify a suspected cause in the "
                    "source before editing it; when the source disproves a hypothesis, "
                    "discard it and continue from the evidence. Forge validates and commits after "
                    "you return. For broad work, delegate repository reconnaissance to "
                    "forge-scout. Before returning, ask forge-critic to inspect the "
                    "actual diff and test results against the acceptance criteria; fix "
                    "the largest gap it identifies. Subagents are read-only: you alone "
                    "apply edits."
                ),
            }
        }
        if self.config.coding.opencode_subagents:
            agents.update(
                {
                    "forge-scout": {
                        "description": (
                            "Read-only repository scout for locating concrete interfaces, "
                            "dependencies, and the smallest relevant file set"
                        ),
                        "mode": "subagent",
                        "model": model_ref,
                        "steps": min(20, self.config.coding.opencode_steps),
                        "temperature": 0.1,
                        "permission": read_only_permission,
                        "prompt": (
                            "Inspect the real repository with read/search tools. Return a "
                            "compact implementation brief naming exact files, symbols, "
                            "interfaces, and constraints. Do not edit anything and do not "
                            "grade a builder summary."
                        ),
                    },
                    "forge-critic": {
                        "description": (
                            "Independent read-only critic that checks the actual worktree "
                            "artifact and test output against acceptance criteria"
                        ),
                        "mode": "subagent",
                        "model": model_ref,
                        "steps": min(20, self.config.coding.opencode_steps),
                        "temperature": 0.1,
                        "permission": read_only_permission,
                        "prompt": (
                            "Inspect the actual diff, source, and focused test results with "
                            "fresh context. Compare them directly with the task acceptance "
                            "criteria. Identify the single largest meaningful remaining gap. "
                            "Do not edit files and do not accept the builder's explanation "
                            "as evidence. Forge intentionally expects modified and untracked "
                            "deliverable files and will stage and commit them after validation, "
                            "so never report git add/commit or untracked status as a defect. "
                            "Explicit task paths and filenames override inferred repository "
                            "naming preferences."
                        ),
                    },
                }
            )
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": model_ref,
            "small_model": model_ref,
            "enabled_providers": [_PROVIDER_ID],
            "share": "disabled",
            "autoupdate": False,
            "compaction": {"auto": True, "prune": True, "reserved": 10_000},
            "provider": {
                _PROVIDER_ID: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Forge local model (cloud disabled)",
                    "options": options,
                    "models": {
                        model_id: {
                            "name": f"Forge {self.model_name}",
                            "limit": {
                                "context": self.spec.context_window,
                                "output": min(self.spec.max_output_tokens, 32_000),
                            },
                        }
                    },
                }
            },
            "agent": agents,
        }

    @staticmethod
    def parse_events(raw: str, *, prompt: str = "") -> OpenCodeResult:
        """Parse OpenCode's newline-delimited JSON event stream."""
        session_id = ""
        texts: list[str] = []
        errors: list[str] = []
        usage = OpenCodeUsage()

        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            session_id = str(event.get("sessionID") or session_id)
            kind = str(event.get("type") or "")
            part = event.get("part") if isinstance(event.get("part"), dict) else {}

            if kind == "text":
                text = event.get("text") or part.get("text")
                if text:
                    texts.append(str(text))
            elif kind == "step_finish":
                tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}
                cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
                usage.input_tokens += _integer(tokens.get("input"))
                usage.output_tokens += _integer(tokens.get("output"))
                usage.reasoning_tokens += _integer(tokens.get("reasoning"))
                usage.cached_tokens += _integer(cache.get("read"))
                usage.steps += 1
                usage.measured = usage.measured or bool(tokens)
            elif kind == "error":
                value = event.get("error") or part.get("error") or event
                errors.append(
                    value if isinstance(value, str) else json.dumps(value, default=str)
                )

        summary = texts[-1].strip() if texts else ""
        if not usage.measured:
            # Older OpenCode versions omitted step token details. Conservative
            # local estimates keep cloud-share accounting from treating the
            # first coached call as 100% cloud.
            usage.input_tokens = estimate_tokens(prompt)
            usage.output_tokens = estimate_tokens("\n".join(texts))

        return OpenCodeResult(
            ok=not errors,
            summary=summary,
            session_id=session_id,
            error="\n".join(errors)[-4000:],
            usage=usage,
            step_limit_reached=bool(_STEP_LIMIT_REACHED.search(summary)),
        )

    def _load_session(self) -> str:
        try:
            data = json.loads(self.session_path.read_text(encoding="utf-8"))
            # A session is inseparable from the worktree OpenCode recorded for
            # it. Legacy records had no root and are deliberately invalidated:
            # reusing one can make --session override a corrected --dir.
            if data.get("worktree_root") != str(self.sandbox.root):
                return ""
            # Repair rounds within one attempt benefit from shared context.
            # A scheduler retry starts from different validation evidence and
            # may contain committed branch work, so its old instructions are
            # stale and actively harmful (for example, "create a fresh diff").
            if data.get("attempt") != self.attempt:
                return ""
            return str(data.get("session_id") or "")
        except (OSError, ValueError, TypeError):
            return ""

    def _save_session(self, session_id: str) -> None:
        path = self.session_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "node_id": self.node_id,
                    "model": self.model_name,
                    "attempt": self.attempt,
                    "worktree_root": str(self.sandbox.root),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _clear_session(self) -> None:
        """Forget OpenCode conversation state without touching its worktree."""
        self.session_path.unlink(missing_ok=True)


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
