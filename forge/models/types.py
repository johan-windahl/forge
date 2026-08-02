"""Provider-neutral types for the model layer.

The shape here is deliberately the intersection of what OpenAI-compatible
servers, Anthropic and local llama.cpp builds can all express, plus the few
Forge-specific concepts (task profile, cache policy) that drive routing. Keeping
this narrow is what allows a new provider to be added by writing one adapter
rather than by threading a new concept through the whole platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


class TaskClass(StrEnum):
    """What kind of thinking a request needs.

    The router keeps success statistics per (task class, tier), so this
    enumeration is the axis along which Forge learns "planning needs a frontier
    model, but renaming a symbol does not". Classes are coarse on purpose:
    finer classes mean slower learning per class.
    """

    PLANNING = "planning"  # decomposition, sequencing, scope
    ARCHITECTURE = "architecture"  # interfaces, module boundaries, tradeoffs
    IMPLEMENTATION = "implementation"  # writing code to a clear spec
    DEBUGGING = "debugging"  # diagnosing an observed failure
    REFACTORING = "refactoring"
    TEST_AUTHORING = "test_authoring"
    CODE_REVIEW = "code_review"
    VISUAL_JUDGEMENT = "visual_judgement"  # does this screenshot look right
    SUMMARIZATION = "summarization"  # rollups, digests, memory compaction
    EXTRACTION = "extraction"  # structured data out of unstructured text
    CLASSIFICATION = "classification"  # cheap, high-volume decisions
    RETROSPECTIVE = "retrospective"  # workflow self-critique
    DOCUMENTATION = "documentation"


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(slots=True)
class ImageRef:
    """An image attached to a message: screenshot, visual diff, video frame."""

    media_type: str
    data_b64: str
    label: str = ""


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    images: list[ImageRef] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    #: Marks the end of a stable prefix for provider-side prompt caching.
    #: Forge orders context so everything before the last such marker is
    #: byte-identical across calls within a node, which is where the large
    #: cloud-token savings come from.
    cache_breakpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "images": [{"media_type": i.media_type, "label": i.label} for i in self.images],
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }


def system(content: str, *, cache: bool = False) -> Message:
    return Message("system", content, cache_breakpoint=cache)


def user(content: str, images: list[ImageRef] | None = None, *, cache: bool = False) -> Message:
    return Message("user", content, images=images or [], cache_breakpoint=cache)


def assistant(content: str, tool_calls: list[ToolCall] | None = None) -> Message:
    return Message("assistant", content, tool_calls=tool_calls or [])


def tool_result(call_id: str, content: str, name: str = "") -> Message:
    return Message("tool", content, tool_call_id=call_id, name=name or None)


@dataclass(slots=True)
class ToolSpec:
    """A tool the model may call. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass(slots=True)
class Request:
    """One completion request, before routing has chosen a model."""

    messages: list[Message]
    profile: TaskProfile
    tools: list[ToolSpec] = field(default_factory=list)
    schema: dict[str, Any] | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] = field(default_factory=list)
    #: Set to skip the response cache; used when the same prompt must produce a
    #: different sample (e.g. deliberately diverse review passes).
    no_cache: bool = False
    node_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Completion:
    """A model's answer, plus everything needed to account for it."""

    text: str
    model: str
    tier: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    latency: float = 0.0
    cost: float = 0.0
    cached: bool = False  # served from Forge's own response cache
    #: Produced by the echo stub rather than a real model. Stub output must
    #: never be priced, cached, or learned from -- see ``EchoProvider``.
    stub: bool = False
    parsed: Any = None  # populated when a schema was supplied
    attempts: int = 1
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason in ("length", "max_tokens")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tier": self.tier,
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason,
            "latency": round(self.latency, 3),
            "cost": round(self.cost, 6),
            "cached": self.cached,
            "stub": self.stub,
            "attempts": self.attempts,
            "text_length": len(self.text),
            "tool_calls": [c.name for c in self.tool_calls],
        }


@dataclass(slots=True)
class TaskProfile:
    """What the caller knows about the request, used to route it.

    Callers state *properties of the task*, never a model preference. That
    inversion is the whole point: it lets the routing policy change -- learned
    from outcomes, tuned by the operator, adapted to a new model roster --
    without touching a single agent.
    """

    task_class: TaskClass
    #: 0.0 trivial ... 1.0 hardest. Estimated by the caller, corrected over time
    #: by observed escalation rates for this class.
    difficulty: float = 0.5
    #: How costly a wrong answer is. High-stakes work (architecture, security)
    #: justifies a stronger model even when the cheap one usually succeeds.
    stakes: float = 0.5
    #: Prior attempts on this same work. Non-zero means something already
    #: failed, which is the strongest signal for escalation there is.
    attempt: int = 0
    needs_vision: bool = False
    needs_tools: bool = False
    #: Force a minimum tier, e.g. the human asked for a frontier review.
    min_tier: str | None = None
    max_tier: str | None = None
    #: Free-form label for grouping in reports.
    label: str = ""

    def escalated(self) -> TaskProfile:
        return TaskProfile(
            task_class=self.task_class,
            difficulty=min(1.0, self.difficulty + 0.2),
            stakes=self.stakes,
            attempt=self.attempt + 1,
            needs_vision=self.needs_vision,
            needs_tools=self.needs_tools,
            min_tier=self.min_tier,
            max_tier=self.max_tier,
            label=self.label,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_class": str(self.task_class),
            "difficulty": self.difficulty,
            "stakes": self.stakes,
            "attempt": self.attempt,
            "needs_vision": self.needs_vision,
            "needs_tools": self.needs_tools,
            "label": self.label,
        }


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: characters over 3.6, plus a floor.

    Every provider tokenises differently and downloading four tokenisers to be
    exact would be absurd for a number used only to *budget* context. The
    estimator is deliberately biased slightly high so the packer under-fills
    rather than overflowing a context window mid-run.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.6) + 1)


def estimate_messages(messages: list[Message]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(message.content) + 4
        for call in message.tool_calls:
            total += estimate_tokens(str(call.arguments)) + 8
        # A 1024x768 screenshot costs roughly this much on vision models.
        total += 800 * len(message.images)
    return total
