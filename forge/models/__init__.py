"""The model layer: providers, routing, budgeting, caching, structured output.

Everything above this layer asks for *a completion for a task*, never for "a
call to model X". Which model actually serves the request is decided by the
router from measured evidence, so the rest of the platform stays ignorant of the
model roster and keeps working when it changes.
"""

from .budget import Budget
from .client import ModelClient
from .registry import Registry
from .router import Router
from .types import Completion, Message, TaskClass, TaskProfile, ToolCall, ToolSpec, Usage

__all__ = [
    "Budget",
    "Completion",
    "Message",
    "ModelClient",
    "Registry",
    "Router",
    "TaskClass",
    "TaskProfile",
    "ToolCall",
    "ToolSpec",
    "Usage",
]
