"""Memory: records, retrieval, context packing, lessons."""

from __future__ import annotations

from pathlib import Path

from forge.kernel.ledger import Ledger
from forge.memory.context import ContextBuilder, Section, read_files
from forge.memory.lessons import Lesson, LessonLibrary
from forge.memory.records import MemoryKind, MemoryStatus
from forge.memory.store import (
    MemoryStore,
    assumption,
    convention,
    decision,
    fact,
    finding,
    interface,
    requirement,
)
from forge.models.types import estimate_tokens
from forge.util.bm25 import Document, Index, tokenize
from forge.util.clock import ManualClock

# --------------------------------------------------------------------------
# Tokenizer and index
# --------------------------------------------------------------------------


def test_compound_identifiers_are_split_for_retrieval() -> None:
    """A query for one naming style must match the others."""
    for identifier in ("renderPlayer", "render_player", "RenderPlayer"):
        assert "render" in tokenize(identifier)
        assert "player" in tokenize(identifier)


def test_index_ranks_by_relevance() -> None:
    index = Index()
    index.add(Document(id="a", text="collision detection between the player and walls"))
    index.add(Document(id="b", text="audio mixing and volume control"))
    assert index.search("player wall collision")[0].doc.id == "a"


def test_document_weight_biases_ranking() -> None:
    index = Index()
    index.add(Document(id="light", text="rendering pipeline", weight=0.5))
    index.add(Document(id="heavy", text="rendering pipeline", weight=3.0))
    assert index.search("rendering pipeline")[0].doc.id == "heavy"


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


def _store(ledger: Ledger, clock: ManualClock) -> MemoryStore:
    return MemoryStore(ledger, "proj_test", clock)


def test_records_are_written_and_retrieved(ledger: Ledger, clock: ManualClock) -> None:
    store = _store(ledger, clock)
    store.write(decision("Use TypeScript", "Static types catch integration errors early"))
    found = store.search("typescript types")
    assert found and found[0].kind == MemoryKind.DECISION


def test_search_refreshes_after_another_store_writes(ledger: Ledger, clock: ManualClock) -> None:
    """A live daemon must see guidance written by a separate CLI process."""
    daemon_store = _store(ledger, clock)
    cli_store = _store(ledger, clock)

    daemon_store.write(fact("Warm the index", "existing memory"))
    assert daemon_store.search("camera safe gradient") == []

    cli_store.write(
        requirement(
            "Camera-safe renderer caching",
            "Keep gradients attached to geometry under camera scroll and shake.",
            source="human",
        )
    )

    hits = daemon_store.search("camera safe gradient", limit=3)
    assert hits and hits[0].title == "Camera-safe renderer caching"


def test_same_title_supersedes_rather_than_duplicating(ledger: Ledger, clock: ManualClock) -> None:
    """Otherwise memory fills with contradictory statements of the same fact."""
    store = _store(ledger, clock)
    first = store.write(assumption("Target browser", "Chrome only", confidence=0.4))
    clock.advance(10)
    second = store.write(assumption("Target browser", "Chrome and Firefox", confidence=0.8))

    active = store.by_kind(MemoryKind.ASSUMPTION)
    assert [r.id for r in active] == [second.id]
    assert store.get(first.id).status == MemoryStatus.SUPERSEDED
    assert store.get(first.id).superseded_by == second.id


def test_path_affinity_outranks_lexical_similarity(ledger: Ledger, clock: ManualClock) -> None:
    store = _store(ledger, clock)
    store.write(interface("Renderer contract", "draw(scene) renders the scene", paths=["src/render.ts"]))
    store.write(fact("Rendering notes", "the scene draw call is documented elsewhere"))

    hits = store.search("scene draw rendering", paths=["src/render.ts"], limit=2)
    assert hits[0].title == "Renderer contract"


def test_conventions_surface_even_without_a_lexical_match(ledger: Ledger, clock: ManualClock) -> None:
    """An agent that never sees the conventions will violate them."""
    store = _store(ledger, clock)
    store.write(convention("Error handling", "Return Result types; never throw across a module boundary"))
    titles = [r.title for r in store.search("implement the audio mixer")]
    assert "Error handling" in titles


def test_memory_projection_rebuilds_from_events(ledger: Ledger, clock: ManualClock) -> None:
    store = _store(ledger, clock)
    store.write(decision("A", "body a"))
    store.write(assumption("B", "body b"))
    store.write(assumption("B", "body b revised"))

    before = {(r.title, r.body) for r in store.active()}
    store.rebuild()
    assert {(r.title, r.body) for r in store.active()} == before


def test_compaction_digests_stale_facts_but_keeps_decisions(ledger: Ledger, clock: ManualClock) -> None:
    store = _store(ledger, clock)
    for i in range(60):
        clock.advance(1)
        store.write(fact(f"observation {i}", "something measured"))
    store.write(decision("Load-bearing choice", "must never be compacted away"))

    store.compact(keep_per_kind=10)
    assert len(store.by_kind(MemoryKind.FACT)) <= 11
    assert len(store.by_kind(MemoryKind.DECISION)) == 1
    assert store.by_kind(MemoryKind.DIGEST)


def test_findings_can_be_resolved(ledger: Ledger, clock: ManualClock) -> None:
    store = _store(ledger, clock)
    record = store.write(finding("Leaky listener", "removed on unmount?", severity="high"))
    assert store.open_findings()
    store.resolve_finding(record.id)
    assert not store.open_findings()


def test_markdown_export_groups_by_kind(ledger: Ledger, clock: ManualClock) -> None:
    store = _store(ledger, clock)
    store.write(decision("Choose Vite", "fast dev server", alternatives=["webpack"]))
    store.write(assumption("Single player", "no multiplayer requested"))
    text = store.export_markdown()

    assert "## Architectural decisions" in text
    assert "## Assumptions" in text
    assert "webpack" in text


# --------------------------------------------------------------------------
# Context builder
# --------------------------------------------------------------------------


def test_sections_are_filled_in_priority_order() -> None:
    builder = ContextBuilder(budget_tokens=200)
    builder.add("critical", "c " * 200, priority=1)
    builder.add("optional", "o " * 200, priority=99)
    report = builder.report()

    names = [s["name"] for s in report["sections"]]
    assert names[0] == "critical"
    assert report["used"] <= 200


def test_low_priority_sections_are_dropped_before_high(tmp_path: Path) -> None:
    builder = ContextBuilder(budget_tokens=60)
    builder.add("goal", "g " * 50, priority=1)
    builder.add("trivia", "t " * 500, priority=99)
    report = builder.report()
    assert "goal" in [s["name"] for s in report["sections"]]


def test_stable_sections_form_a_single_cache_breakpoint() -> None:
    builder = ContextBuilder(budget_tokens=10_000)
    builder.add("architecture", "stable content", priority=10, stable=True)
    builder.add("failure", "volatile content", priority=20)
    messages = builder.build(system_prompt="role", task="do it")

    breakpoints = [m for m in messages if m.cache_breakpoint]
    assert len(breakpoints) == 1
    assert "stable content" in breakpoints[0].content
    assert messages[-1].role == "user"


def test_head_tail_trimming_keeps_both_ends() -> None:
    lines = "\n".join(f"line {i}" for i in range(500))
    section = Section(name="log", content=lines, head_lines=3, tail_lines=3)
    trimmed = section.trimmed(budget=40)
    assert "line 0" in trimmed and "line 499" in trimmed and "omitted" in trimmed


def test_a_tail_section_too_big_for_its_head_tail_form_still_gets_the_tail() -> None:
    """The fallback path used to hand back the head instead.

    The head/tail candidate is only tried at the sizes the caller asked for.
    When those do not fit -- which is exactly when the section is enormous and
    trimming matters -- it fell through to a plain prefix, silently inverting
    the request. A validation section asking for the last 80 lines of vitest
    output got the npm banner and lost the failing assertion, which is a fair
    description of three fix rounds spent not finding the error.
    """
    lines = "\n".join(f"line {i}" for i in range(500))
    section = Section(name="log", content=lines, tail_lines=80)

    # Too small for 80 lines plus a marker, so the head/tail form cannot apply.
    trimmed = section.trimmed(budget=30)
    assert "line 499" in trimmed, "the tail is what was asked for"
    assert "line 0" not in trimmed


def test_a_head_and_tail_section_keeps_something_from_each_end() -> None:
    lines = "\n".join(f"line {i}" for i in range(500))
    section = Section(name="log", content=lines, head_lines=80, tail_lines=80)
    trimmed = section.trimmed(budget=30)
    assert "line 0" in trimmed and "line 499" in trimmed


def test_a_plain_section_is_still_truncated_from_the_front() -> None:
    """The change must not turn every trim into a tail."""
    section = Section(name="notes", content="\n".join(f"line {i}" for i in range(500)))
    trimmed = section.trimmed(budget=30)
    assert "line 0" in trimmed and "line 499" not in trimmed


def test_reading_files_cannot_escape_the_workspace(tmp_path: Path) -> None:
    """`read_files` is the chokepoint every model-supplied path passes through.

    `_grant_files` checks the paths a model asks for by name, but planner JSON
    reaches here directly via `spec["paths"]`, and nothing checked those. A
    planned path of "../secret" was read and placed in the prompt.
    """
    root = tmp_path / "wt"
    root.mkdir()
    (root / "inside.ts").write_text("mine")
    (tmp_path / "secret.txt").write_text("credentials")

    files = read_files(root, ["../secret.txt", "inside.ts"])
    assert list(files) == ["inside.ts"]


def test_a_symlink_out_of_the_workspace_is_not_followed(tmp_path: Path) -> None:
    """`is_file()` follows symlinks quite happily, so the check resolves first."""
    root = tmp_path / "wt"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("credentials")
    (root / "link.ts").symlink_to(tmp_path / "secret.txt")

    assert read_files(root, ["link.ts"]) == {}


def test_files_are_numbered_for_precise_reference() -> None:
    builder = ContextBuilder(budget_tokens=10_000)
    builder.add_files("files", {"a.py": "first\nsecond"})
    body = builder.build(system_prompt="r", task="t")[1].content
    assert "1 | first" in body and "2 | second" in body


def test_token_estimate_is_conservative() -> None:
    """Biased high, so the packer under-fills rather than overflowing."""
    text = "hello world " * 100
    assert estimate_tokens(text) >= len(text.split()) / 2


# --------------------------------------------------------------------------
# Lessons
# --------------------------------------------------------------------------


def test_lessons_persist_and_are_searchable(tmp_path: Path, clock: ManualClock) -> None:
    library = LessonLibrary(tmp_path / "lessons", clock=clock, project="p1")
    library.add(Lesson(title="Chromium needs --no-sandbox in containers",
                       body="Browser gates fail to launch otherwise", tags=["browser"]))

    reloaded = LessonLibrary(tmp_path / "lessons", clock=clock)
    assert reloaded.search("chromium sandbox container")


def test_rediscovering_a_lesson_confirms_it_rather_than_duplicating(tmp_path: Path, clock: ManualClock) -> None:
    """Retrospectives run every milestone and will re-derive the same lesson."""
    library = LessonLibrary(tmp_path / "lessons", clock=clock)
    library.add(Lesson(title="Chromium needs --no-sandbox in containers",
                       body="Browser gates fail to launch otherwise"))
    library.add(Lesson(title="Chromium needs --no-sandbox in containers",
                       body="Browser gates fail to launch otherwise"))

    assert len(library.all()) == 1
    assert library.all()[0].confirmed == 1


def test_distinct_lessons_about_one_subject_are_not_merged(tmp_path: Path, clock: ManualClock) -> None:
    """A false merge silently destroys a lesson, so the bar has to be high.

    These share almost all of their vocabulary and make completely different
    points. An earlier BM25-ratio implementation merged them.
    """
    library = LessonLibrary(tmp_path / "lessons", clock=clock)
    library.add(Lesson(
        title="The local endpoint is tailnet-only, so unreachability is a network fact",
        body="The local llama.cpp server has no authentication; a connection failure "
             "means the host is not on the tailnet, never a credentials problem.",
    ))
    library.add(Lesson(
        title="llama.cpp constrains JSON output properly, so trust the schema",
        body="The local llama.cpp server honours response_format json_schema with GBNF "
             "grammars, so structured output from the local rungs cannot be malformed.",
    ))
    assert len(library.all()) == 2


def test_seed_lessons_install_and_reseed_idempotently(tmp_path: Path, clock: ManualClock) -> None:
    from forge.models.host_notes import SEED_LESSONS, seed_library

    library = LessonLibrary(tmp_path / "lessons", clock=clock)
    seed_library(library, quiet=True)
    assert len(library.all()) == len(SEED_LESSONS), "every seed lesson must survive dedup"

    seed_library(library, quiet=True)
    assert len(library.all()) == len(SEED_LESSONS), "re-seeding must confirm, not duplicate"
    assert all(item.confirmed >= 1 for item in library.all())


def test_seed_lessons_are_retrievable_by_symptom(tmp_path: Path, clock: ManualClock) -> None:
    """The point of seeding: the right lesson surfaces from a failure description."""
    from forge.models.host_notes import seed_library

    library = LessonLibrary(tmp_path / "lessons", clock=clock)
    seed_library(library, quiet=True)

    hits = library.search("model returned empty content but used all the output tokens", limit=3)
    assert hits and "empty answer" in hits[0].title


def test_contradicted_lessons_lose_confidence_and_retire(tmp_path: Path, clock: ManualClock) -> None:
    library = LessonLibrary(tmp_path / "lessons", clock=clock)
    lesson = library.add(Lesson(title="Always inline styles", body="questionable advice"))
    for _ in range(4):
        library.contradict(lesson.id)
    assert not library.all()


def test_established_requires_repeated_confirmation(tmp_path: Path, clock: ManualClock) -> None:
    library = LessonLibrary(tmp_path / "lessons", clock=clock)
    lesson = library.add(Lesson(title="Scaffold before deciding boundaries causes rework", body="x"))
    assert not lesson.established
    library.confirm(lesson.id)
    library.confirm(lesson.id)
    assert library.all()[0].established


def test_concurrent_lesson_writes_do_not_collide(tmp_path: Path, clock: ManualClock) -> None:
    """The lessons directory is global, shared by every project and process.

    Observed live: two workers confirmed the same lesson at once, both wrote
    `lesson_X.tmp`, and the slower rename failed with ENOENT because the faster
    one had already moved the file. That failed a node outright, over a
    bookkeeping write.
    """
    import threading

    library = LessonLibrary(tmp_path / "lessons", clock=clock)
    lesson = library.add(Lesson(title="Chromium needs --no-sandbox", body="in containers"))

    errors: list[Exception] = []

    def hammer() -> None:
        try:
            for _ in range(25):
                library.confirm(lesson.id)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent lesson writes collided: {errors[:2]}"
    assert library.all(), "the lesson must survive the hammering"


def test_no_temp_files_are_left_behind(tmp_path: Path, clock: ManualClock) -> None:
    """A stray .tmp would be picked up by nothing and confuse an operator."""
    library = LessonLibrary(tmp_path / "lessons", clock=clock)
    library.add(Lesson(title="A lesson", body="a body"))
    assert not list((tmp_path / "lessons").glob("*.tmp"))


def test_a_supporting_artifact_cannot_displace_the_artwork(tmp_path: Path) -> None:
    """Callers truncate, so ordering decides what a vision model actually sees.

    Observed for real: five references were supplied for a pinball table and
    the goal check received `nightmare-audio-spectrogram.png`, because it asked
    for one and `sorted()` put the spectrogram first. It then judged whether
    the game resembled a frequency plot. Supplying more reference material made
    the comparison worse, which is exactly backwards.
    """
    from forge.memory.context import reference_images

    root = tmp_path / "workspace"
    references = tmp_path / ".forge" / "references"
    references.mkdir(parents=True)
    root.mkdir()
    for name in (
        "nightmare-audio-spectrogram.png",
        "nightmare-table-reference.png",
        "nightmare-video-contact-sheet.jpg",
        "nightmare.png",
    ):
        (references / name).write_bytes(b"\x89PNG")

    assert reference_images(root, limit=1) == [references / "nightmare-table-reference.png"]

    ordered = [path.name for path in reference_images(root)]
    assert ordered.index("nightmare-table-reference.png") < ordered.index(
        "nightmare-audio-spectrogram.png"
    )
    assert len(reference_images(root)) == 4
