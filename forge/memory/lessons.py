"""Cross-project lessons.

Project memory dies with the project. Lessons do not. A lesson is knowledge
about *how to build software with this platform* that would apply to the next
project too: "browser gates on this host need `--no-sandbox`", "asking the local
model for a full-file rewrite of anything over 400 lines produces truncation",
"scaffolding before deciding the module boundaries causes rework".

The library is a directory of JSON files outside any project, so it survives
project deletion and can be version-controlled, inspected and pruned by a human.
Lessons are retrieved into planning and retrospective prompts by relevance, and
each carries a *usefulness* counter so lessons that never get applied fade and
lessons that keep proving right rise.

Deliberately kept small. A library of five hundred vague lessons is worse than
twenty sharp ones, because retrieval noise is a direct tax on every prompt. The
promotion bar is therefore high: a lesson must be recorded as having helped
before it is considered established.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..obs.log import get_logger
from ..util.bm25 import Document, Index, tokenize
from ..util.clock import Clock, default_clock
from ..util.ids import new_id

log = get_logger("memory.lessons")

MAX_LESSONS = 300


@dataclass(slots=True)
class Lesson:
    title: str
    body: str
    #: What triggered it, so a reader can judge whether it generalises.
    context: str = ""
    #: Tags used for retrieval: "browser", "planning", "node", "routing".
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("lesson"))
    created_at: float = 0.0
    #: Times this lesson was included in a prompt.
    used: int = 0
    #: Times a later outcome confirmed it was right.
    confirmed: int = 0
    #: Times a later outcome contradicted it.
    contradicted: int = 0
    project: str = ""

    @property
    def score(self) -> float:
        """Confidence in the lesson, from its track record.

        Starts neutral, rises with confirmations, falls hard on contradictions.
        A contradicted lesson is worse than no lesson, so it is penalised more
        than a confirmation rewards.
        """
        return (1.0 + self.confirmed) / (1.0 + self.confirmed + 2.0 * self.contradicted)

    @property
    def established(self) -> bool:
        return self.confirmed >= 2 and self.score > 0.6

    def render(self) -> str:
        return f"[lesson] {self.title}\n{self.body.strip()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "context": self.context,
            "tags": self.tags,
            "created_at": self.created_at,
            "used": self.used,
            "confirmed": self.confirmed,
            "contradicted": self.contradicted,
            "project": self.project,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lesson:
        return cls(
            id=data.get("id") or new_id("lesson"),
            title=data["title"],
            body=data.get("body", ""),
            context=data.get("context", ""),
            tags=list(data.get("tags", [])),
            created_at=float(data.get("created_at", 0.0)),
            used=int(data.get("used", 0)),
            confirmed=int(data.get("confirmed", 0)),
            contradicted=int(data.get("contradicted", 0)),
            project=data.get("project", ""),
        )


class LessonLibrary:
    """A directory of lesson files, shared by every project on the host."""

    def __init__(self, root: Path | str, *, clock: Clock | None = None, project: str = "") -> None:
        self.root = Path(root).expanduser()
        self.project = project
        self._clock = clock or default_clock()
        self._lock = threading.Lock()
        self._cache: dict[str, Lesson] | None = None
        self.root.mkdir(parents=True, exist_ok=True)

    # -- storage ---------------------------------------------------------

    def _path(self, lesson_id: str) -> Path:
        return self.root / f"{lesson_id}.json"

    def _load(self) -> dict[str, Lesson]:
        with self._lock:
            if self._cache is None:
                cache: dict[str, Lesson] = {}
                for path in sorted(self.root.glob("*.json")):
                    try:
                        cache_entry = Lesson.from_dict(json.loads(path.read_text(encoding="utf-8")))
                        cache[cache_entry.id] = cache_entry
                    except (OSError, json.JSONDecodeError, KeyError) as exc:
                        log.warn("skipping unreadable lesson", path=str(path), error=str(exc))
                self._cache = cache
            return self._cache

    def _save(self, lesson: Lesson) -> None:
        """Write one lesson atomically.

        The temp file is unique per writer, not per lesson. With a shared name,
        two writers touching the same lesson both wrote ``lesson_X.tmp`` and the
        slower one's rename failed with ENOENT because the faster one had
        already moved it -- which failed a node outright. A thread lock would
        not be enough either: this directory is global, shared by every project
        and every concurrent ``forge`` process, so uniqueness has to come from
        the filename rather than from in-process coordination.
        """
        path = self._path(lesson.id)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(lesson.to_dict(), indent=2), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    # -- writing ---------------------------------------------------------

    def add(self, lesson: Lesson) -> Lesson:
        """Store a lesson, merging with a near-duplicate if one exists.

        Deduplication matters more here than anywhere else in the system:
        retrospectives run after every milestone and will happily rediscover the
        same lesson a dozen times. Merging keeps the library sharp and turns
        repetition into confirmation, which is exactly what repetition means.
        """
        lesson.created_at = lesson.created_at or self._clock.now()
        lesson.project = lesson.project or self.project
        existing = self._find_similar(lesson)
        if existing is not None:
            existing.confirmed += 1
            if len(lesson.body) > len(existing.body):
                existing.body = lesson.body
            existing.tags = sorted(set(existing.tags) | set(lesson.tags))
            self._save(existing)
            log.debug("lesson merged", title=existing.title[:60], confirmed=existing.confirmed)
            return existing
        with self._lock:
            if self._cache is not None:
                self._cache[lesson.id] = lesson
        self._save(lesson)
        log.info("lesson recorded", title=lesson.title[:80])
        self._prune()
        return lesson

    def _find_similar(self, lesson: Lesson, threshold: float = 0.55) -> Lesson | None:
        """Find an existing lesson that says the same thing.

        Jaccard overlap on token sets, not BM25. BM25 answers "which document
        best matches this query", which is not the same question: in a library
        where every lesson shares a subject vocabulary, the best match is always
        reasonably high-scoring even when the two lessons are unrelated. That
        produced real merges of genuinely distinct lessons -- "the endpoint is
        unreachable" folded into "llama.cpp constrains JSON output" -- and a
        merge silently destroys a lesson.

        Jaccard is symmetric and bounded, so the threshold means the same thing
        regardless of library size. The title must agree too, because two
        lessons about one subsystem share most of their body vocabulary while
        making opposite points.
        """
        lessons = self._load()
        if not lessons:
            return None

        def token_set(text: str) -> set[str]:
            return set(tokenize(text))

        def jaccard(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        new_title = token_set(lesson.title)
        new_full = token_set(f"{lesson.title} {lesson.body}")

        best: tuple[float, Lesson] | None = None
        for candidate in lessons.values():
            body_score = jaccard(new_full, token_set(f"{candidate.title} {candidate.body}"))
            title_score = jaccard(new_title, token_set(candidate.title))
            # Both must agree. Either alone produces false merges.
            if body_score >= threshold and title_score >= 0.5:
                combined = (body_score + title_score) / 2
                if best is None or combined > best[0]:
                    best = (combined, candidate)
        return best[1] if best else None

    def confirm(self, lesson_id: str) -> None:
        lesson = self._load().get(lesson_id)
        if lesson:
            lesson.confirmed += 1
            self._save(lesson)

    def contradict(self, lesson_id: str) -> None:
        lesson = self._load().get(lesson_id)
        if lesson:
            lesson.contradicted += 1
            self._save(lesson)
            if lesson.score < 0.25:
                log.info("retiring contradicted lesson", title=lesson.title[:60])
                self.remove(lesson_id)

    def remove(self, lesson_id: str) -> None:
        self._path(lesson_id).unlink(missing_ok=True)
        with self._lock:
            if self._cache is not None:
                self._cache.pop(lesson_id, None)

    def _prune(self) -> None:
        lessons = self._load()
        if len(lessons) <= MAX_LESSONS:
            return
        ranked = sorted(lessons.values(), key=lambda item: (item.score, item.confirmed, item.created_at))
        for lesson in ranked[: len(lessons) - MAX_LESSONS]:
            self.remove(lesson.id)

    # -- reading ---------------------------------------------------------

    def all(self) -> list[Lesson]:
        return sorted(self._load().values(), key=lambda item: (-item.score, -item.confirmed))

    def search(self, query: str, *, limit: int = 6, tags: list[str] | None = None) -> list[Lesson]:
        lessons = self._load()
        if not lessons:
            return []
        index = Index()
        index.add_all(
            [
                Document(
                    id=item.id,
                    text=f"{item.title} {item.body} {item.context} {' '.join(item.tags)}",
                    weight=item.score,
                )
                for item in lessons.values()
                if not tags or set(tags) & set(item.tags)
            ]
        )
        hits = index.search(query, limit=limit)
        found = [lessons[hit.doc.id] for hit in hits if hit.doc.id in lessons]
        for lesson in found:
            lesson.used += 1
            self._save(lesson)
        return found

    def render(self, lessons: list[Lesson] | None = None, *, limit: int = 6) -> str:
        selected = lessons if lessons is not None else self.all()[:limit]
        return "\n\n".join(lesson.render() for lesson in selected)

    def stats(self) -> dict[str, Any]:
        lessons = self._load().values()
        return {
            "count": len(lessons),
            "established": sum(1 for item in lessons if item.established),
            "mean_score": round(sum(item.score for item in lessons) / len(lessons), 3) if lessons else 0.0,
            "root": str(self.root),
        }
