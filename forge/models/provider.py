"""Provider interface and the concrete adapters.

Adding a provider means implementing :meth:`Provider.complete` and registering a
factory in :data:`PROVIDER_KINDS`. Nothing else in Forge changes -- the router
scores it by measured outcome, the budget prices it from config, the cache keys
it by content.

Concurrency is enforced here rather than in the router, because the constraint
belongs to the *endpoint*: a single llama.cpp server with two slots will queue
or reject a third concurrent request no matter how clever the routing was.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Any

from ..config import ModelSpec, ProviderConfig
from ..errors import ConfigError, MalformedOutput, ModelError, ReasoningBudgetExhausted
from ..obs.log import get_logger
from ..util.clock import Clock, default_clock
from ..util.ids import new_id
from .http import HttpClient
from .types import Completion, Message, Request, ToolCall, ToolSpec, Usage

log = get_logger("models.provider")


class Provider(ABC):
    """A backend that can turn messages into a completion."""

    kind: str = "abstract"

    #: Marks a provider standing in for a real model during ``--dry-run``.
    #: Set by the dry-run installer, not by the provider class: the echo
    #: provider is a perfectly valid test double, and output produced *as a
    #: test double* should still exercise pricing, caching and routing. It is
    #: only rehearsal-in-place-of-a-real-run that must leave no trace.
    stub: bool = False

    def __init__(self, name: str, config: ProviderConfig, clock: Clock | None = None) -> None:
        self.name = name
        self.config = config
        self._clock = clock or default_clock()
        self._http = HttpClient(timeout=config.timeout, max_retries=config.max_retries)
        self._semaphores: dict[str, threading.Semaphore] = {}
        self._sem_lock = threading.Lock()
        #: Ceiling across every model this provider serves, when it has one. The
        #: per-model limits below cannot express "these rungs are one GPU": on
        #: the local server `local` and `local_deep` are the same weights on the same
        #: box, so their separate semaphores allowed two concurrent generations
        #: against hardware that is already saturated by one. Measured: 26.7
        #: tok/s alone, 13.7 each when doubled -- identical aggregate throughput,
        #: half the speed per call, so every call sat twice as close to its
        #: timeout for no gain in work done.
        self._provider_sem = (
            threading.Semaphore(config.max_concurrency) if config.max_concurrency > 0 else None
        )

    def _semaphore(self, spec: ModelSpec) -> threading.Semaphore:
        with self._sem_lock:
            sem = self._semaphores.get(spec.name)
            if sem is None:
                sem = threading.Semaphore(max(1, spec.concurrency))
                self._semaphores[spec.name] = sem
            return sem

    def api_key(self) -> str:
        if not self.config.api_key_env:
            return ""
        key = os.environ.get(self.config.api_key_env, "")
        if not key:
            raise ConfigError(
                f"provider {self.name!r} needs {self.config.api_key_env} in the environment"
            )
        return key

    def available(self) -> bool:
        """Cheap readiness probe used by the router to skip dead providers."""
        if self.config.api_key_env and not os.environ.get(self.config.api_key_env):
            return False
        return True

    def complete(self, request: Request, spec: ModelSpec) -> Completion:
        """Run one completion, respecting the model's concurrency limit."""
        sem = self._semaphore(spec)
        acquired = sem.acquire(timeout=spec.timeout)
        if not acquired:  # pragma: no cover - only under sustained saturation
            raise ModelError("timed out waiting for a model slot", model=spec.name)
        # Provider-wide limit second, always in this order: two locks acquired in
        # a consistent order cannot deadlock against each other.
        if self._provider_sem is not None and not self._provider_sem.acquire(timeout=spec.timeout):
            sem.release()  # pragma: no cover - only under sustained saturation
            raise ModelError("timed out waiting for a provider slot", model=spec.name)
        start = self._clock.monotonic()
        try:
            completion = self._complete(request, spec)
        finally:
            if self._provider_sem is not None:
                self._provider_sem.release()
            sem.release()
        completion.latency = self._clock.monotonic() - start
        # Rehearsal output is free by definition. Pricing it would charge a dry
        # run real money against the budget and make `forge status` report
        # progress and a cloud fraction that no model ever produced.
        completion.stub = self.stub
        completion.cost = 0.0 if completion.stub else spec.cost(
            completion.usage.input_tokens - completion.usage.cached_input_tokens,
            completion.usage.output_tokens,
        )
        return completion

    @abstractmethod
    def _complete(self, request: Request, spec: ModelSpec) -> Completion: ...


# --------------------------------------------------------------------------
# OpenAI-compatible (llama.cpp, vLLM, Ollama, LM Studio, OpenAI itself)
# --------------------------------------------------------------------------


class OpenAICompatProvider(Provider):
    """Chat Completions API.

    Covers the local server and, with a different base URL and an API key,
    OpenAI. The differences that matter -- whether the server honours
    ``response_format: json_schema``, whether it supports tools -- are read from
    the :class:`ModelSpec` rather than sniffed at runtime, so a misconfigured
    endpoint fails loudly at startup instead of silently degrading.
    """

    kind = "openai_compat"

    def _complete(self, request: Request, spec: ModelSpec) -> Completion:
        payload: dict[str, Any] = {
            "model": spec.model or spec.name,
            "messages": _merge_same_role([self._encode(m, spec) for m in request.messages]),
            "max_tokens": request.max_output_tokens or spec.max_output_tokens,
            "stream": False,
        }
        temperature = request.temperature
        if temperature is None:
            temperature = spec.extra.get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        if request.stop:
            payload["stop"] = request.stop

        # Reasoning toggle for Qwen-family chat templates on llama.cpp.
        #
        # This is the single most consequential setting against the local
        # server. With thinking on, the model emits chain-of-thought into a
        # separate `reasoning_content` field, which is charged against the same
        # output budget as the answer -- so a small budget produces an empty
        # answer and a full token count. Measured on one trivial structured
        # request: 19 output tokens with thinking off, 691 with it on.
        #
        # Forge therefore makes it explicit per rung rather than leaving it to
        # the server default: the fast rung disables it, the deep rung enables
        # it and pairs it with a large budget.
        thinking = spec.extra.get("thinking")
        if thinking is not None:
            payload["chat_template_kwargs"] = {
                **payload.get("chat_template_kwargs", {}),
                "enable_thinking": bool(thinking),
            }
        if request.tools and spec.supports_tools:
            payload["tools"] = [
                {"type": "function", "function": t.to_dict()} for t in request.tools
            ]
            payload["tool_choice"] = "auto"
        if request.schema and spec.supports_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": request.schema, "strict": True},
            }
        elif request.schema:
            # Server cannot constrain decoding; the client layer falls back to
            # prompt-level instruction plus validate-and-repair.
            payload["response_format"] = {"type": "json_object"}
        for key in ("reasoning_effort", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty"):
            if key in spec.extra:
                payload[key] = spec.extra[key]
        payload.update(request.extra.get("provider_params", {}))

        headers = dict(self.config.headers)
        if self.config.api_key_env:
            headers["Authorization"] = f"Bearer {self.api_key()}"

        data = self._http.post_json(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            payload,
            headers=headers,
            timeout=spec.timeout,
        )
        return self._decode(data, spec, max_output_tokens=payload["max_tokens"])

    def _encode(self, message: Message, spec: ModelSpec) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id or "",
                "content": message.content,
            }
        if message.role == "assistant" and message.tool_calls:
            return {
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in message.tool_calls
                ],
            }
        if message.images and spec.supports_vision:
            parts: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
            parts.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.media_type};base64,{img.data_b64}"},
                }
                for img in message.images
            )
            return {"role": message.role, "content": parts}
        return {"role": message.role, "content": message.content}

    def _decode(
        self, data: dict[str, Any], spec: ModelSpec, *, max_output_tokens: int = 0
    ) -> Completion:
        choices = data.get("choices") or []
        if not choices:
            raise ModelError("provider returned no choices", model=spec.name, raw=str(data)[:500])
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))

        # llama.cpp separates chain-of-thought from the answer. The reasoning is
        # not the answer and must never be returned as one, but it does consume
        # the output budget -- so an empty answer with a full token count means
        # the model thought until it ran out of room. That is a specific,
        # recoverable condition: retry with a bigger budget. Without this check
        # it is indistinguishable from a broken model, and the router would
        # escalate to an expensive rung to fix a problem that is not there.
        reasoning = message.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason") or "stop"
        if not text.strip() and reasoning:
            requested = max_output_tokens or spec.max_output_tokens
            spent = data.get("usage") or {}
            raise ReasoningBudgetExhausted(
                "model spent its entire output budget on reasoning and returned no answer",
                model=spec.name,
                max_output_tokens=requested,
                reasoning_chars=len(reasoning),
                finish_reason=finish_reason,
                # Carried so the layer above can account for work that really
                # happened: a failed reasoning overrun on a local model is
                # minutes of GPU time, and dropping it makes a busy run look idle.
                input_tokens=int(spent.get("prompt_tokens", 0)),
                output_tokens=int(spent.get("completion_tokens", 0)),
                hint="raise max_output_tokens for this rung, or disable thinking",
            )
        calls = [
            ToolCall(
                id=call.get("id") or new_id("call"),
                name=call["function"]["name"],
                arguments=_loads_lenient(call["function"].get("arguments", "{}")),
            )
            for call in message.get("tool_calls") or []
        ]
        raw_usage = data.get("usage") or {}
        details = raw_usage.get("prompt_tokens_details") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("prompt_tokens", 0)),
            output_tokens=int(raw_usage.get("completion_tokens", 0)),
            cached_input_tokens=int(details.get("cached_tokens", 0)),
            # llama.cpp does not break reasoning out of completion_tokens, so
            # estimate it from the text when the server does not report it.
            reasoning_tokens=int(
                (raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            )
            or (len(reasoning) // 4 if reasoning else 0),
        )
        return Completion(
            text=text,
            model=spec.name,
            tier=spec.tier,
            usage=usage,
            tool_calls=calls,
            finish_reason=finish_reason,
            raw={"id": data.get("id"), "had_reasoning": bool(reasoning)},
        )


# --------------------------------------------------------------------------
# Anthropic Messages API
# --------------------------------------------------------------------------


class AnthropicProvider(Provider):
    """Anthropic Messages API, including prompt caching.

    Prompt caching is the single most effective cloud-token lever Forge has.
    Cache breakpoints are placed by the context packer at stable prefix
    boundaries; this adapter simply translates them into ``cache_control``
    markers. On a node that makes several frontier calls sharing an architecture
    digest, this routinely cuts billed input tokens by most of their volume.
    """

    kind = "anthropic"
    API_VERSION = "2023-06-01"

    def _complete(self, request: Request, spec: ModelSpec) -> Completion:
        system_blocks, turns = self._split(request.messages, spec)
        payload: dict[str, Any] = {
            "model": spec.model or spec.name,
            "max_tokens": request.max_output_tokens or spec.max_output_tokens,
            "messages": turns,
        }
        if system_blocks:
            payload["system"] = system_blocks
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop_sequences"] = request.stop

        tools = list(request.tools)
        if request.schema:
            # Anthropic has no response_format; a single-tool forced call is the
            # supported way to guarantee schema-shaped output.
            tools = [
                *tools,
                ToolSpec(
                    name="emit_result",
                    description="Return the result. You must call this exactly once.",
                    parameters=request.schema,
                ),
            ]
            payload["tool_choice"] = {"type": "tool", "name": "emit_result"}
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        payload.update(request.extra.get("provider_params", {}))

        headers = {
            "x-api-key": self.api_key(),
            "anthropic-version": self.API_VERSION,
            **self.config.headers,
        }
        data = self._http.post_json(
            f"{self.config.base_url.rstrip('/')}/messages",
            payload,
            headers=headers,
            timeout=spec.timeout,
        )
        return self._decode(data, spec, schema_tool=bool(request.schema))

    def _split(self, messages: list[Message], spec: ModelSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        system_blocks: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                block: dict[str, Any] = {"type": "text", "text": message.content}
                if message.cache_breakpoint and spec.supports_prompt_cache:
                    block["cache_control"] = {"type": "ephemeral"}
                system_blocks.append(block)
                continue
            if message.role == "tool":
                turns.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id or "",
                                "content": message.content,
                            }
                        ],
                    }
                )
                continue
            content: list[dict[str, Any]] = []
            if message.content:
                block = {"type": "text", "text": message.content}
                if message.cache_breakpoint and spec.supports_prompt_cache:
                    block["cache_control"] = {"type": "ephemeral"}
                content.append(block)
            for img in message.images:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.media_type,
                            "data": img.data_b64,
                        },
                    }
                )
            for call in message.tool_calls:
                content.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            turns.append({"role": message.role, "content": content or [{"type": "text", "text": ""}]})
        return system_blocks, _merge_consecutive(turns)

    def _decode(self, data: dict[str, Any], spec: ModelSpec, *, schema_tool: bool) -> Completion:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.get("id") or new_id("call"),
                        name=block.get("name", ""),
                        arguments=block.get("input") or {},
                    )
                )
        raw_usage = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("input_tokens", 0))
            + int(raw_usage.get("cache_read_input_tokens", 0))
            + int(raw_usage.get("cache_creation_input_tokens", 0)),
            output_tokens=int(raw_usage.get("output_tokens", 0)),
            cached_input_tokens=int(raw_usage.get("cache_read_input_tokens", 0)),
        )
        text = "".join(text_parts)
        if schema_tool:
            emitted = next((c for c in calls if c.name == "emit_result"), None)
            if emitted is None:
                raise MalformedOutput(
                    "model did not call emit_result", model=spec.name, got=[c.name for c in calls]
                )
            text = json.dumps(emitted.arguments)
            calls = [c for c in calls if c.name != "emit_result"]
        return Completion(
            text=text,
            model=spec.name,
            tier=spec.tier,
            usage=usage,
            tool_calls=calls,
            finish_reason=data.get("stop_reason") or "stop",
            raw={"id": data.get("id")},
        )


def _merge_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse adjacent same-role text messages into one.

    The OpenAI API tolerates several consecutive ``system`` messages. Qwen's
    chat template does not: it raises inside the Jinja template and llama.cpp
    returns a 400 ("Unable to generate parser for this template"). Forge's
    context builder emits three system blocks by design -- role instructions,
    the cacheable stable prefix, then volatile context -- so *every* local call
    failed and silently escalated to a cloud rung. The platform kept working,
    which is exactly why it went unnoticed until a live run was inspected.

    Merging is semantically identical on every server and costs two newlines, so
    it is done unconditionally rather than sniffed per model. Messages carrying
    tool calls or tool results are left alone: their structure is meaningful.
    """
    merged: list[dict[str, Any]] = []
    for message in messages:
        previous = merged[-1] if merged else None
        mergeable = (
            previous is not None
            and previous.get("role") == message.get("role")
            and message.get("role") in ("system", "user")
            and isinstance(previous.get("content"), str)
            and isinstance(message.get("content"), str)
            and not previous.get("tool_calls")
            and not message.get("tool_calls")
        )
        if mergeable:
            assert previous is not None
            previous["content"] = f"{previous['content']}\n\n{message['content']}"
        else:
            merged.append(dict(message))
    return merged


def _merge_consecutive(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic rejects two consecutive turns with the same role."""
    merged: list[dict[str, Any]] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["content"] = list(merged[-1]["content"]) + list(turn["content"])
        else:
            merged.append(dict(turn))
    return merged


# --------------------------------------------------------------------------
# Deterministic stub, for tests and dry runs
# --------------------------------------------------------------------------


class EchoProvider(Provider):
    """A provider that answers from a scripted table or a canned default.

    Every test in the suite runs against this, which is why the suite needs no
    network and finishes in seconds. ``forge run --dry-run`` uses it too, so an
    operator can exercise the whole orchestration path -- graph, gates, git,
    checkpoints -- without spending a token.
    """

    kind = "echo"

    def __init__(self, name: str, config: ProviderConfig, clock: Clock | None = None) -> None:
        super().__init__(name, config, clock)
        self.responses: list[str] = []
        self.handler = None
        self.calls: list[Request] = []

    def _complete(self, request: Request, spec: ModelSpec) -> Completion:
        self.calls.append(request)
        if self.handler is not None:
            text = self.handler(request, spec)
        elif self.responses:
            text = self.responses.pop(0)
        elif request.schema:
            text = json.dumps(_skeleton(request.schema))
        else:
            text = f"[echo:{spec.name}] {request.messages[-1].content[:200]}"
        prompt_tokens = sum(len(m.content) for m in request.messages) // 4
        return Completion(
            text=text,
            model=spec.name,
            tier=spec.tier,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=max(1, len(text) // 4)),
        )


def _skeleton(schema: dict[str, Any]) -> Any:
    """Minimal instance satisfying a schema; used by the echo provider."""
    stype = schema.get("type", "object")
    if isinstance(stype, list):
        stype = stype[0]
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    match stype:
        case "object":
            required = schema.get("required", list(schema.get("properties", {})))
            return {k: _skeleton(v) for k, v in schema.get("properties", {}).items() if k in required}
        case "array":
            item = schema.get("items", {"type": "string"})
            return [_skeleton(item)] * max(1, int(schema.get("minItems", 1)))
        case "string":
            return schema.get("description", "")[:40] or "placeholder"
        case "integer" | "number":
            return schema.get("minimum", 0)
        case "boolean":
            return True
    return None


def _loads_lenient(text: str) -> dict[str, Any]:
    """Tool arguments occasionally arrive double-encoded or with trailing text."""
    if isinstance(text, dict):
        return text
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        from .structured import extract_json

        value = extract_json(text)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
    return value if isinstance(value, dict) else {"value": value}


def encode_image(path: str | os.PathLike[str], label: str = ""):
    """Read an image file into a message-attachable reference."""
    from pathlib import Path

    from .types import ImageRef

    p = Path(path)
    suffix = p.suffix.lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        suffix, "image/png"
    )
    return ImageRef(media_type=media, data_b64=base64.b64encode(p.read_bytes()).decode("ascii"), label=label or p.name)


def _cli_kinds() -> dict[str, type[Provider]]:
    """Imported lazily: the CLI providers pull in subprocess machinery that a
    deployment using only HTTP providers has no reason to load."""
    from .cli_provider import ClaudeCliProvider, CodexCliProvider

    return {"claude_cli": ClaudeCliProvider, "codex_cli": CodexCliProvider}


PROVIDER_KINDS: dict[str, type[Provider]] = {
    "openai_compat": OpenAICompatProvider,
    "openai": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
    "echo": EchoProvider,
}


def build_provider(name: str, config: ProviderConfig, clock: Clock | None = None) -> Provider:
    cls = PROVIDER_KINDS.get(config.kind)
    if cls is None:
        cls = _cli_kinds().get(config.kind)
    if cls is None:
        raise ConfigError(
            f"unknown provider kind {config.kind!r}",
            known=sorted([*PROVIDER_KINDS, "claude_cli", "codex_cli"]),
        )
    return cls(name, config, clock)
