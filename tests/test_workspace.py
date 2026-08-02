"""Workspace: git, atomic patching, sandbox guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.errors import PatchError, SandboxError
from forge.workspace.git import Repo
from forge.workspace.patch import EditPlan, FileEdit, apply_edits
from forge.workspace.sandbox import LocalSandbox, detect_toolchain


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return Repo(tmp_path / "wt").init()


# --------------------------------------------------------------------------
# Patching
# --------------------------------------------------------------------------


def _plan(*edits: FileEdit) -> EditPlan:
    return EditPlan(edits=list(edits), summary="test")


def test_write_creates_nested_files(tmp_path: Path) -> None:
    apply_edits(tmp_path, _plan(FileEdit(path="src/a/b.txt", op="write", content="hi")))
    assert (tmp_path / "src/a/b.txt").read_text() == "hi"


def test_replace_swaps_an_exact_anchor(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("def a():\n    return 1\n")
    apply_edits(tmp_path, _plan(FileEdit(path="f.py", op="replace", anchor="return 1", content="return 2")))
    assert "return 2" in (tmp_path / "f.py").read_text()


def test_replace_tolerates_indentation_drift(tmp_path: Path) -> None:
    """Models reproduce indentation imperfectly far more often than logic."""
    (tmp_path / "f.py").write_text("class A:\n        def go(self):\n            return 1\n")
    apply_edits(
        tmp_path,
        _plan(FileEdit(path="f.py", op="replace", anchor="def go(self):\nreturn 1", content="def go(self):\n        return 2")),
    )
    assert "return 2" in (tmp_path / "f.py").read_text()


def test_missing_anchor_reports_the_file_and_the_anchor(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("x = 1\n")
    with pytest.raises(PatchError) as exc:
        apply_edits(tmp_path, _plan(FileEdit(path="f.py", op="replace", anchor="y = 2", content="z")))
    assert exc.value.context["path"] == "f.py"
    assert "y = 2" in exc.value.context["anchor_preview"]


def test_ambiguous_anchor_selects_the_requested_occurrence(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("v = 0\nv = 0\n")
    apply_edits(tmp_path, _plan(FileEdit(path="f.py", op="replace", anchor="v = 0", content="v = 9", occurrence=2)))
    assert (tmp_path / "f.py").read_text() == "v = 0\nv = 9\n"


def test_an_anchor_matching_twice_with_no_occurrence_is_refused(tmp_path: Path) -> None:
    """Silently editing the first of several matches is a wrong edit that looks
    like a right one.

    The guard for this existed but could never fire: it tested `occurrence < 1`
    against a schema with `minimum: 1` and a parser that coerced everything
    absent to 1. So "the first one" and "I did not consider that there might be
    more than one" were indistinguishable, and the second was treated as the
    first.
    """
    (tmp_path / "f.py").write_text("v = 0\nv = 0\n")
    with pytest.raises(PatchError) as exc:
        apply_edits(tmp_path, _plan(FileEdit(path="f.py", op="replace", anchor="v = 0", content="v = 9")))

    assert exc.value.context["matches"] == 2
    assert "occurrence" in str(exc.value), "it must say how to fix it"
    assert (tmp_path / "f.py").read_text() == "v = 0\nv = 0\n", "and change nothing"


def test_a_unique_anchor_still_needs_no_occurrence(tmp_path: Path) -> None:
    """The refusal is narrow: the common case must stay effortless."""
    (tmp_path / "f.py").write_text("v = 0\nw = 1\n")
    apply_edits(tmp_path, _plan(FileEdit(path="f.py", op="replace", anchor="v = 0", content="v = 9")))
    assert (tmp_path / "f.py").read_text() == "v = 9\nw = 1\n"


def test_a_plan_is_all_or_nothing(tmp_path: Path) -> None:
    """The critical property: a bad edit late in a plan changes nothing at all."""
    (tmp_path / "existing.txt").write_text("original")
    with pytest.raises(PatchError):
        apply_edits(
            tmp_path,
            _plan(
                FileEdit(path="new.txt", op="write", content="created"),
                FileEdit(path="existing.txt", op="write", content="changed"),
                FileEdit(path="missing.txt", op="replace", anchor="nope", content="x"),
            ),
        )
    assert not (tmp_path / "new.txt").exists()
    assert (tmp_path / "existing.txt").read_text() == "original"


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", ".git/config", ".forge/ledger.db"])
def test_paths_outside_the_workspace_are_refused(tmp_path: Path, path: str) -> None:
    with pytest.raises(PatchError):
        apply_edits(tmp_path, _plan(FileEdit(path=path, op="write", content="x")))


def test_dry_run_reports_without_writing(tmp_path: Path) -> None:
    result = apply_edits(tmp_path, _plan(FileEdit(path="a.txt", op="write", content="x")), dry_run=True)
    assert result.written == ["a.txt"]
    assert not (tmp_path / "a.txt").exists()


# --------------------------------------------------------------------------
# Git
# --------------------------------------------------------------------------


def test_init_is_idempotent(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "r").init()
    head = repo.head()
    assert Repo(tmp_path / "r").init().head() == head


def test_commit_records_the_node_in_a_trailer(repo: Repo) -> None:
    (repo.path / "a.txt").write_text("hello")
    sha = repo.commit("feat: add a", node_id="node_123")
    assert sha
    assert repo.log(limit=1)[0].node_id == "node_123"


def test_commit_returns_none_when_nothing_changed(repo: Repo) -> None:
    assert repo.commit("empty") is None


def test_reset_hard_discards_a_failed_attempt(repo: Repo) -> None:
    """This is what makes retrying a node safe."""
    (repo.path / "good.txt").write_text("committed")
    baseline = repo.commit("feat: good")

    (repo.path / "good.txt").write_text("half-written garbage")
    (repo.path / "stray.txt").write_text("left behind")
    assert repo.is_dirty()

    repo.reset_hard(baseline)
    assert not repo.is_dirty()
    assert (repo.path / "good.txt").read_text() == "committed"
    assert not (repo.path / "stray.txt").exists()


def test_conflicting_merge_leaves_a_clean_tree(repo: Repo) -> None:
    (repo.path / "f.txt").write_text("base\n")
    repo.commit("base")
    repo.checkout("side", create=True)
    (repo.path / "f.txt").write_text("side\n")
    repo.commit("side change")
    repo.checkout("main")
    (repo.path / "f.txt").write_text("main\n")
    repo.commit("main change")

    assert repo.merge("side") is False
    assert not repo.is_dirty(), "an aborted merge must not leave conflict markers"


def test_runtime_state_stays_ignored_when_project_replaces_gitignore(tmp_path: Path) -> None:
    """A scaffold owns .gitignore; Forge's internal exclusions must not depend on it."""
    repo = Repo(tmp_path / "r").init()
    (repo.path / ".gitignore").write_text("dist/\n", encoding="utf-8")
    (repo.path / ".forge").mkdir()
    (repo.path / ".forge" / "deps-stamp.json").write_text("{}")
    (repo.path / "node_modules").mkdir()
    (repo.path / "node_modules" / "x.js").write_text("x")

    status = repo.status()
    paths = [path for _state, path in status]
    assert ".forge/" not in paths
    assert "node_modules/" not in paths


def test_operational_metadata_can_be_neutralised_before_a_merge(repo: Repo) -> None:
    (repo.path / ".forge").mkdir()
    stamp = repo.path / ".forge" / "deps-stamp.json"
    stamp.write_text('{"fingerprint":"main"}')
    # Reproduce a legacy repository from before Forge's private exclude existed.
    repo._git("add", "-f", ".forge/deps-stamp.json")
    repo.commit("track legacy stamp", add_all=False)
    repo.checkout("side", create=True)
    stamp.write_text('{"fingerprint":"side"}')
    repo.commit("side runtime state")

    assert repo.match_paths("main", [".forge/deps-stamp.json"])
    repo.commit("neutralise runtime state")
    assert ".forge/deps-stamp.json" not in repo.changed_files("main")


def test_worktrees_isolate_parallel_work(repo: Repo, tmp_path: Path) -> None:
    (repo.path / "shared.txt").write_text("v1")
    repo.commit("base")

    side = repo.add_worktree(tmp_path / "side", "feature/x")
    (side.path / "shared.txt").write_text("v2")
    side.commit("change on the branch")

    assert (repo.path / "shared.txt").read_text() == "v1"
    repo.remove_worktree(tmp_path / "side")


# --------------------------------------------------------------------------
# Sandbox
# --------------------------------------------------------------------------


def _sandbox(tmp_path: Path) -> LocalSandbox:
    from forge.config import SandboxConfig

    return LocalSandbox(SandboxConfig(command_timeout=10), tmp_path)


def test_commands_run_and_report_output(tmp_path: Path) -> None:
    result = _sandbox(tmp_path).exec(["echo", "hello"])
    assert result.ok and "hello" in result.stdout


def test_denylist_refuses_destructive_commands(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="denylist"):
        _sandbox(tmp_path).exec("rm -rf / --no-preserve-root", shell=True)


def test_timeouts_kill_the_process_group(tmp_path: Path) -> None:
    from forge.errors import CommandTimeout

    with pytest.raises(CommandTimeout):
        _sandbox(tmp_path).exec(["sleep", "30"], timeout=0.5)


def test_reads_are_confined_to_the_sandbox_root(tmp_path: Path) -> None:
    with pytest.raises(SandboxError):
        _sandbox(tmp_path).path_for("../../etc/passwd")


def test_oversized_output_keeps_both_ends(tmp_path: Path) -> None:
    from forge.util.proc import clamp_output

    text = "START" + ("x" * 100_000) + "END"
    clamped, truncated = clamp_output(text, limit=1000)
    assert truncated and clamped.startswith("START") and clamped.endswith("END")


def test_toolchain_detection_finds_node_scripts(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"scripts": {"build": "vite build", "test": "vitest run", "dev": "vite"}}'
    )
    (tmp_path / "package-lock.json").write_text("{}")
    detected = detect_toolchain(_sandbox(tmp_path))

    assert "node" in detected["languages"]
    assert detected["commands"]["build"] == "npm run build"
    assert detected["commands"]["unit"] == "npm run test"
    assert detected["commands"]["serve"] == "npm run dev"


def test_toolchain_detection_prefers_the_declared_package_manager(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"scripts": {"build": "x"}}')
    (tmp_path / "pnpm-lock.yaml").write_text("")
    assert detect_toolchain(_sandbox(tmp_path))["commands"]["build"] == "pnpm run build"


# --------------------------------------------------------------------------
# Resetting the working tree between attempts
# --------------------------------------------------------------------------


def test_reset_preserves_installed_dependencies(tmp_path: Path) -> None:
    """`git clean -fdx` removes ignored files, so it would delete node_modules.

    Every node attempt resets the tree. Wiping node_modules there means a full
    dependency install per node, which is both enormously slow and the thing
    that widened a two-worker clean into a collision over tens of thousands of
    files.
    """
    from forge.workspace.git import Repo

    repo = Repo(tmp_path / "wt").init()
    (repo.path / "node_modules" / "typescript").mkdir(parents=True)
    (repo.path / "node_modules" / "typescript" / "index.js").write_text("x")
    (repo.path / "src.ts").write_text("keep me")
    repo.commit("add source")
    # Created *after* the commit, so it is genuinely untracked junk rather than
    # a tracked file that reset would restore.
    (repo.path / "junk.tmp").write_text("discard me")
    (repo.path / "src.ts").write_text("half-finished edit")

    repo.reset_hard()

    assert (repo.path / "node_modules" / "typescript" / "index.js").exists(), \
        "installed dependencies must survive a reset"
    assert not (repo.path / "junk.tmp").exists(), "untracked junk must still be removed"
    assert (repo.path / "src.ts").read_text() == "keep me", "tracked edits must be discarded"


def test_a_vanished_path_during_clean_is_not_a_failure() -> None:
    """Verbatim stderr from the live collision; every path is already gone."""
    from forge.workspace.git import _only_vanished_paths

    stderr = (
        "warning: failed to remove node_modules/@typescript-eslint/scope-manager/"
        "dist/lib/esnext.intl.d.ts.map: No such file or directory\n"
        "warning: could not lstat node_modules/@typescript-eslint/type-utils\n"
        ": No such file or directory"
    )
    assert _only_vanished_paths(stderr)


def test_a_real_clean_failure_is_still_raised() -> None:
    """Permission errors must not be swallowed along with the benign warnings."""
    from forge.workspace.git import _only_vanished_paths

    assert not _only_vanished_paths("error: unable to unlink 'x': Permission denied")
    assert not _only_vanished_paths("")


def test_concurrent_resets_do_not_collide(tmp_path: Path) -> None:
    """Workers share one working tree, so the reset must be exclusive.

    Two nodes started in the same millisecond and both ran `git clean`; each
    deleted files the other was walking, and both nodes blocked.
    """
    import threading

    from forge.workspace.git import Repo

    repo = Repo(tmp_path / "wt").init()
    for i in range(200):
        (repo.path / f"file{i}.tmp").write_text("x" * 100)
    repo.commit("base")

    errors: list[Exception] = []

    def reset() -> None:
        try:
            for _ in range(5):
                for i in range(200):
                    (repo.path / f"junk{i}.tmp").write_text("y")
                repo.reset_hard()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reset) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent resets collided: {errors[:2]}"


def test_a_write_with_no_content_does_not_blank_a_file(tmp_path: Path) -> None:
    """The pinball run lost nine attempts to a zero-byte module.

    `content` is optional in the edit schema, so a model that names a path and
    then runs out of output budget emits a write with nothing in it. Silently
    truncating the file turns that into four gate failures that all describe
    something other than the actual problem.
    """
    source = tmp_path / "collide.ts"
    source.write_text("export function collide() {}\n")

    with pytest.raises(PatchError, match="blank an existing file"):
        apply_edits(tmp_path, _plan(FileEdit(path="collide.ts", op="write", content="")))

    assert source.read_text() == "export function collide() {}\n"


def test_a_write_with_no_content_may_still_create_a_new_file(tmp_path: Path) -> None:
    """An empty new file is a legitimate thing to create -- __init__.py, .keep."""
    apply_edits(tmp_path, _plan(FileEdit(path="pkg/__init__.py", op="write", content="")))
    assert (tmp_path / "pkg/__init__.py").read_text() == ""


def test_emptying_a_file_on_purpose_still_works_through_delete(tmp_path: Path) -> None:
    (tmp_path / "gone.txt").write_text("bye")
    apply_edits(tmp_path, _plan(FileEdit(path="gone.txt", op="delete")))
    assert not (tmp_path / "gone.txt").exists()


def test_restoring_paths_undoes_only_those_paths(repo: Repo) -> None:
    """The safe form of undo while another worker shares the checkout."""
    (repo.path / "mine.txt").write_text("committed\n")
    (repo.path / "theirs.txt").write_text("committed\n")
    repo.commit("feat: base")

    (repo.path / "mine.txt").write_text("broken\n")
    (repo.path / "mine_new.txt").write_text("half-written\n")
    (repo.path / "theirs.txt").write_text("another worker is mid-edit\n")

    restored = repo.restore_paths(["mine.txt", "mine_new.txt"])

    assert sorted(restored) == ["mine.txt", "mine_new.txt"]
    assert (repo.path / "mine.txt").read_text() == "committed\n"
    assert not (repo.path / "mine_new.txt").exists()
    # The concurrent worker's uncommitted work is not ours to discard.
    assert (repo.path / "theirs.txt").read_text() == "another worker is mid-edit\n"


def test_restoring_a_path_that_was_never_written_is_harmless(repo: Repo) -> None:
    (repo.path / "a.txt").write_text("x\n")
    repo.commit("feat: base")
    assert repo.restore_paths(["never-existed.txt", "", "../escape.txt"]) == []


def test_a_directory_in_the_files_place_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    """The rename, not the write, is what fails -- and it fails in phase two.

    A stray `create_dir` claimed `collide.ts`, so the atomic rename raised
    IsADirectoryError after other files had already landed, breaking the
    all-or-nothing guarantee and discarding 13KB of correct generated code.
    """
    (tmp_path / "collide.ts").mkdir()

    with pytest.raises(PatchError, match="directory already exists"):
        apply_edits(
            tmp_path,
            _plan(
                FileEdit(path="other.ts", op="write", content="export const a = 1\n"),
                FileEdit(path="collide.ts", op="write", content="export const b = 2\n"),
            ),
        )

    assert not (tmp_path / "other.ts").exists(), "phase one must reject before phase two writes"
    assert not list(tmp_path.glob("**/*.forge-tmp")), "no scratch file may be left behind"


def test_a_plan_cannot_both_create_a_directory_and_write_a_file_at_one_path(tmp_path: Path) -> None:
    with pytest.raises(PatchError, match="both creates a directory and writes a file"):
        apply_edits(
            tmp_path,
            _plan(
                FileEdit(path="src/thing.ts", op="create_dir"),
                FileEdit(path="src/thing.ts", op="write", content="export const x = 1\n"),
            ),
        )
    assert not (tmp_path / "src/thing.ts").exists()
