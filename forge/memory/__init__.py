"""Structured project memory and context assembly.

The premise: a long autonomous run cannot re-read its own history. Sending the
transcript back is quadratic in tokens and linear in confusion. So Forge writes
what it learns into *typed records* -- assumptions, decisions, interfaces,
conventions, lessons -- and assembles each prompt from the small subset of those
records that the current task needs.
"""

from .context import ContextBuilder, Section
from .lessons import LessonLibrary
from .records import MemoryKind, MemoryRecord
from .store import MemoryStore

__all__ = [
    "ContextBuilder",
    "LessonLibrary",
    "MemoryKind",
    "MemoryRecord",
    "MemoryStore",
    "Section",
]
