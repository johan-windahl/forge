"""Seed knowledge about this host's models.

Everything Forge learns normally comes from running. That works, but it means
the first day of a fresh install rediscovers things that are already known --
including a few that are expensive to learn the hard way, like a rung returning
empty answers because its output budget went on chain-of-thought.

These seed lessons close that gap. They go into the same cross-project lesson
library that retrospectives write to, so they are retrieved by the same
relevance search, they can be contradicted and retired by evidence like any
other lesson, and a human can read and edit them as plain JSON.

They are *facts about the deployment*, not advice about software. Each was
verified against the live endpoints rather than assumed:

* the local server's slot count, context, modalities and reasoning behaviour
  come from its own ``/props`` and measured completions;
* the CLI overheads are measured token counts from real invocations.

Install with ``forge lessons --seed``. Re-running is safe: the library merges
duplicates into confirmations rather than accumulating copies.
"""

from __future__ import annotations

from .lessons_data import SEED_LESSONS

__all__ = ["SEED_LESSONS", "seed_library"]


def seed_library(library, *, quiet: bool = False) -> int:
    """Add the seed lessons to a :class:`~forge.memory.lessons.LessonLibrary`.

    Returns the number added or confirmed.
    """
    from ..memory.lessons import Lesson

    count = 0
    for entry in SEED_LESSONS:
        library.add(
            Lesson(
                title=entry["title"],
                body=entry["body"],
                context=entry.get("context", "Verified against the live host."),
                tags=list(entry.get("tags", [])),
                project="__host__",
            )
        )
        count += 1
        if not quiet:
            print(f"  + {entry['title']}")
    return count
