"""Reference material: acquisition, description, and what reaches the model."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.workspace.references import (
    MANIFEST_NAME,
    ReferenceError,
    ReferenceStore,
    infer_role,
)


@pytest.fixture
def store(tmp_path: Path) -> ReferenceStore:
    return ReferenceStore(tmp_path / ".forge")


def _file(tmp_path: Path, name: str, content: bytes = b"x") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_a_local_file_is_copied_in_with_its_description(store, tmp_path) -> None:
    origin = _file(tmp_path, "table.png", b"pretend png")

    ref = store.add(str(origin), description="match this composition, not the palette")

    assert (store.root / "table.png").read_bytes() == b"pretend png"
    assert ref.description == "match this composition, not the palette"
    assert ref.role == "visual"
    assert ref.sha256 and ref.source == str(origin.resolve())


def test_the_manifest_survives_a_reload(store, tmp_path) -> None:
    store.add(str(_file(tmp_path, "spec.md", b"# spec")), description="the API contract")

    reloaded = ReferenceStore(store.root.parent).load()

    assert [(r.file, r.role, r.description) for r in reloaded] == [
        ("spec.md", "document", "the API contract")
    ]


def test_the_same_bytes_added_twice_stay_one_reference(store, tmp_path) -> None:
    """Re-adding must not leave two copies explaining the same picture differently."""
    first = _file(tmp_path, "a.png", b"identical")
    second = _file(tmp_path, "b.png", b"identical")

    store.add(str(first), description="original note")
    store.add(str(second), description="better note")

    refs = store.load()
    assert len(refs) == 1
    assert refs[0].description == "better note"


def test_a_missing_local_file_is_refused(store) -> None:
    with pytest.raises(ReferenceError):
        store.add("/nonexistent/nope.png")


def test_files_dropped_in_by_hand_are_still_found(store, tmp_path) -> None:
    """The directory worked before the manifest did; it must keep working."""
    store.root.mkdir(parents=True)
    (store.root / "dropped.png").write_bytes(b"png")

    refs = store.load()

    assert [r.file for r in refs] == ["dropped.png"]
    assert refs[0].description == ""


def test_derived_material_never_outranks_what_it_came_from(store, tmp_path) -> None:
    """A contact sheet must not displace the artwork it was cut from.

    This is the bug the old filename heuristic was patching: a goal check asked
    for one reference, sorted alphabetically, got `nightmare-audio-spectrogram`
    and judged whether the game resembled a frequency plot.
    """
    store.add(str(_file(tmp_path, "zzz-artwork.png", b"art")), description="the target")
    store.add(
        str(_file(tmp_path, "aaa-contact-sheet.png", b"sheet")),
        description="frames from the video",
        derived_from="motion.mp4",
    )

    assert [r.file for r in store.load()] == ["zzz-artwork.png", "aaa-contact-sheet.png"]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.png", "visual"),
        ("a.mp4", "motion"),
        ("a.wav", "audio"),
        ("a.md", "document"),
        ("a.json", "example"),
        ("a.bin", "other"),
        # Both a still and a motion format; supplied as a reference it is
        # essentially always for the movement.
        ("a.gif", "motion"),
    ],
)
def test_roles_are_inferred_from_the_file_type(name: str, expected: str) -> None:
    assert infer_role(name) == expected


def test_a_corrupt_manifest_does_not_lose_the_files(store, tmp_path) -> None:
    store.add(str(_file(tmp_path, "keep.png", b"png")))
    (store.root / MANIFEST_NAME).write_text("{not json", encoding="utf-8")

    assert [r.file for r in store.load()] == ["keep.png"]


def test_descriptions_reach_the_goal_check(store, tmp_path) -> None:
    """The whole point: the operator's sentence travels with the picture."""
    from forge.memory.context import reference_images_described

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store.add(str(_file(tmp_path, "ref.png", b"png")), description="palette only")

    described = reference_images_described(workspace)

    assert described == [(store.root / "ref.png", "palette only")]


def test_non_visual_references_are_not_sent_to_the_vision_model(store, tmp_path) -> None:
    from forge.memory.context import reference_images_described

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store.add(str(_file(tmp_path, "clip.mp4", b"video")), description="how it moves")

    assert reference_images_described(workspace) == []


def test_a_manifest_does_not_let_diagnostics_displace_the_artwork(store, tmp_path) -> None:
    """The bug the derived-name heuristic exists to stop, via the manifest.

    Declared entries used to bypass `_reference_rank` entirely, so once a
    manifest existed the whole folder -- declared and hand-dropped alike -- was
    emitted alphabetically. Four supporting diagnostics sort ahead of `table.png`
    and, at the default limit, the actual reference never reaches the model.
    """
    from forge.memory.context import reference_images_described

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store.add(str(_file(tmp_path, "table.png", b"the target")), description="the target")
    for name in (
        "aaa-contact-sheet.png",
        "bb-diff.png",
        "cc-thumbnail.png",
        "nightmare-audio-spectrogram.png",
    ):
        (store.root / name).write_bytes(name.encode())

    described = reference_images_described(workspace, limit=4)

    assert described[0] == (store.root / "table.png", "the target")
    assert [p.name for p, _ in described[1:]] == [
        "aaa-contact-sheet.png",
        "bb-diff.png",
        "cc-thumbnail.png",
    ]


def test_undeclared_forge_files_do_not_outrank_references_versioned_in_the_repo(
    store, tmp_path
) -> None:
    """Neither directory is privileged; only declaration and derivation rank.

    Existing of a manifest used to make every file in `.forge/references` sort
    ahead of everything in `docs/references`, including files nobody had
    declared, so a project that versions its references lost them all.
    """
    from forge.memory.context import reference_images_described

    workspace = tmp_path / "workspace"
    versioned = workspace / "docs" / "references"
    versioned.mkdir(parents=True)
    (versioned / "b-target.png").write_bytes(b"versioned target")
    store.add(str(_file(tmp_path, "declared.png", b"declared")), description="the brief")
    for name in ("a-thumbnail.png", "c-scratch.png"):
        (store.root / name).write_bytes(name.encode())

    described = reference_images_described(workspace, limit=3)

    assert [p.name for p, _ in described] == [
        "declared.png",  # declared, so it leads
        "b-target.png",  # then plain alphabetical across both directories
        "c-scratch.png",  # `a-thumbnail.png` is a derived diagnostic, so it sinks
    ]


def test_an_excluded_image_is_not_smuggled_back_in_by_the_directory_scan(store, tmp_path) -> None:
    """A role that is not `visual` means "do not compare against this".

    Skipping such an entry without marking it seen left the directory scan free
    to pick the same file up again -- stripped of the description that said it
    was the design being replaced.
    """
    from forge.memory.context import reference_images_described

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store.add(
        str(_file(tmp_path, "old-design.png", b"png")),
        description="this is what we are REPLACING",
        role="other",
    )

    assert reference_images_described(workspace) == []


def test_adding_a_file_already_dropped_in_by_hand_does_not_duplicate_it(store, tmp_path) -> None:
    """Dedup has to see entries the directory scan synthesised, which carry no hash."""
    store.root.mkdir(parents=True)
    (store.root / "mockup.png").write_bytes(b"the same bytes")

    ref = store.add(str(_file(tmp_path, "mockup.png", b"the same bytes")), description="target layout")

    assert [r.file for r in store.load()] == ["mockup.png"]
    assert ref.description == "target layout"
    assert ref.sha256, "adoption into the manifest is where the hash gets recorded"


def test_re_adding_applies_the_corrected_role(store, tmp_path) -> None:
    """Re-adding is how an operator fixes a reference; a silent no-op lies about it."""
    origin = _file(tmp_path, "diagram.png", b"png")
    store.add(str(origin), description="the layout")

    ref = store.add(str(origin), role="document", derived_from="spec.md")

    assert (ref.role, ref.derived_from) == ("document", "spec.md")
    assert ref.description == "the layout", "an omitted description must not erase the old one"


def test_a_web_page_served_as_an_image_is_refused(store, tmp_path, monkeypatch) -> None:
    """A soft 404 stored once is compared against for the life of the project."""
    import io
    from email.message import Message

    class FakeResponse(io.BytesIO):
        headers = Message()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    response = FakeResponse(b"<html>Sign in to continue</html>")
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: response)

    with pytest.raises(ReferenceError):
        store.add("https://example.test/mockup.png", description="target layout")

    assert not (store.root / "mockup.png").exists()


# --------------------------------------------------------------------------
# The init prompt flow
# --------------------------------------------------------------------------


def test_the_prompt_collects_several_references_with_descriptions(monkeypatch) -> None:
    from forge import cli

    answers = iter([
        "https://example.test/table.png",
        "composition and palette, modernise the art",
        "/local/clip.mp4",
        "how the ball moves",
        "",  # blank line ends the loop
    ])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    assert cli._ask_references() == [
        ("https://example.test/table.png", "composition and palette, modernise the art"),
        ("/local/clip.mp4", "how the ball moves"),
    ]


def test_the_prompt_survives_being_interrupted(monkeypatch) -> None:
    """Ctrl-C while listing references keeps the goal already typed."""
    from forge import cli

    answers = iter(["https://example.test/a.png", "a note"])

    def fake_input(*a):
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr("builtins.input", fake_input)

    assert cli._ask_references() == [("https://example.test/a.png", "a note")]


def test_interrupting_at_the_description_keeps_the_source(monkeypatch) -> None:
    """Discarding the line just typed reads as the prompt having ignored it."""
    from forge import cli

    answers = iter(["https://example.test/a.png"])

    def fake_input(*a):
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr("builtins.input", fake_input)

    assert cli._ask_references() == [("https://example.test/a.png", "")]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://a.test/x.png", ("https://a.test/x.png", "")),
        ("x.png::the palette", ("x.png", "the palette")),
        ("  x.png  ::  spaced  ", ("x.png", "spaced")),
    ],
)
def test_a_flag_carries_an_optional_inline_description(raw, expected) -> None:
    from forge import cli

    assert cli._split_reference(raw) == expected


def test_scripts_and_ci_are_never_prompted(monkeypatch) -> None:
    """A pipeline that blocks on stdin hangs with no indication why."""
    import argparse

    from forge import cli

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    assert cli._can_prompt(argparse.Namespace(no_input=False))
    assert not cli._can_prompt(argparse.Namespace(no_input=True))

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert not cli._can_prompt(argparse.Namespace(no_input=False))
