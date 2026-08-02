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
