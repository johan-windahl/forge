"""Git as the durability substrate for code.

Forge does not invent a snapshot format. Every unit of work ends as a commit,
every checkpoint is a tag, every rollback is a reset. This buys a great deal:

* **Rollback is exact and cheap.** No copying trees, no "which files did that
  attempt touch?" bookkeeping.
* **The audit trail is legible to a human.** An operator can ``git log`` a
  three-day autonomous run and read what happened, with the node id in every
  commit trailer.
* **Isolation for parallel work.** Worktrees let two nodes edit the same
  repository concurrently without a lock, each on its own branch, merged back
  through the same mechanism a human team would use.

Commit identity is fixed to a Forge author so autonomous commits are trivially
distinguishable from human ones in ``git log --author``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..errors import GitError
from ..obs.log import get_logger
from ..util.proc import ProcResult, run

log = get_logger("workspace.git")

#: Directories kept across a working-tree reset. `git clean -fdx` removes
#: ignored files too, which means it deletes `node_modules` -- so every node
#: attempt would re-run a full dependency install, and the tree walk over tens
#: of thousands of files is what let two workers collide mid-clean. None of
#: these are "local changes" the reset exists to discard: they are derived from
#: a manifest that is itself version-controlled, and Forge fingerprints them
#: separately (see workspace/deps.py).
PRESERVED_ON_RESET = (
    ".forge",
    "node_modules",
    ".venv",
    "venv",
    "target",          # rust
    "vendor",          # go, php
    ".gradle",
    ".pnpm-store",
    ".cache",
)

#: `git clean` warns and exits 1 when a path disappears while it is walking the
#: tree. The paths it is complaining about are already gone, which is the goal.
_VANISHED_MARKERS = ("no such file or directory", "could not lstat", "failed to remove")


#: Wording that means the clean genuinely could not do its job.
_HARD_FAILURE_MARKERS = ("error:", "fatal:", "permission denied", "read-only file system",
                        "directory not empty", "device or resource busy")


def _only_vanished_paths(stderr: str) -> bool:
    """True when the clean only complained about paths that were already gone.

    Not written as "every line is a warning": git wraps a long path onto its
    own line and puts the errno on the next, so the continuation line is a bare
    ``: No such file or directory``. Requiring every line to start with
    ``warning:`` rejected the real output. Asking instead whether anything
    indicates a *hard* failure is both simpler and robust to that wrapping.
    """
    text = (stderr or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _HARD_FAILURE_MARKERS):
        return False
    return any(marker in text for marker in _VANISHED_MARKERS)

AUTHOR_NAME = "Forge"
AUTHOR_EMAIL = "forge@localhost"

#: Files never committed, whatever the project's own .gitignore says.
BASE_IGNORE = """\
.forge/
node_modules/
__pycache__/
*.pyc
.venv/
venv/
dist/
build/
.DS_Store
*.log
"""


@dataclass(slots=True)
class CommitInfo:
    sha: str
    subject: str
    author: str
    timestamp: int
    node_id: str | None = None


class Repo:
    """A git repository, driven by the CLI rather than a binding.

    Shelling out to ``git`` rather than linking libgit2 or pygit2 is a
    deliberate choice: it is the interface with the strongest backwards
    compatibility guarantee in the entire toolchain, it needs no build step, and
    when something goes wrong the operator can reproduce it by pasting the
    command from the log.
    """

    def __init__(self, path: Path | str, *, timeout: float = 120.0) -> None:
        self.path = Path(path).expanduser().resolve()
        self.timeout = timeout
        # Workers share one working tree, so a reset is exclusive. Two nodes
        # started in the same millisecond and both ran `git clean`; each
        # deleted files the other was still walking, and both nodes blocked on
        # the resulting "could not lstat". Reset is the only operation that
        # removes files wholesale, so it is the only one that needs this.
        self._reset_lock = threading.RLock()

    # -- plumbing --------------------------------------------------------

    def _git(self, *args: str, check: bool = True, timeout: float | None = None, stdin: str | None = None) -> ProcResult:
        env = {
            "GIT_AUTHOR_NAME": AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
            "GIT_TERMINAL_PROMPT": "0",  # never block waiting for credentials
            "GIT_CONFIG_NOSYSTEM": "1",  # host config must not change behaviour
            "LC_ALL": "C",  # stable, parseable output
        }
        result = run(
            ["git", *args],
            cwd=self.path,
            env=env,
            timeout=timeout or self.timeout,
            stdin=stdin,
        )
        if check and not result.ok:
            raise GitError(
                f"git {' '.join(args[:3])} failed",
                exit_code=result.returncode,
                stderr=result.stderr.strip()[:800],
            )
        return result

    # -- lifecycle -------------------------------------------------------

    @property
    def exists(self) -> bool:
        return (self.path / ".git").exists()

    def init(self, *, initial_branch: str = "main") -> Repo:
        """Create the repository if absent; idempotent."""
        self.path.mkdir(parents=True, exist_ok=True)
        if not self.exists:
            run(["git", "init", "-b", initial_branch], cwd=self.path, timeout=self.timeout, check=True)
            log.info("initialised repository", path=str(self.path), branch=initial_branch)
        self._git("config", "user.name", AUTHOR_NAME)
        self._git("config", "user.email", AUTHOR_EMAIL)
        # Autonomous merges must never open an editor or wait for input.
        self._git("config", "core.editor", "true")
        self._git("config", "merge.ff", "false")
        self._git("config", "gc.auto", "0")  # no surprise repacks mid-run
        self.ensure_private_excludes()
        gitignore = self.path / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(BASE_IGNORE, encoding="utf-8")
        if not self.has_commits():
            self.commit("chore: initialise repository", allow_empty=True)
        return self

    def ensure_private_excludes(self) -> None:
        """Ignore Forge runtime state even if the project replaces .gitignore."""
        if not self.exists:
            return
        git_dir = Path(self._git("rev-parse", "--git-common-dir").stdout.strip())
        if not git_dir.is_absolute():
            git_dir = (self.path / git_dir).resolve()
        exclude = git_dir / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        additions = [
            pattern for pattern in (".forge/", "node_modules/", ".venv/", "venv/")
            if pattern not in existing.splitlines()
        ]
        if additions:
            separator = "" if not existing or existing.endswith("\n") else "\n"
            exclude.write_text(
                existing + separator + "\n".join(additions) + "\n",
                encoding="utf-8",
            )

    def match_paths(self, ref: str, paths: Sequence[str]) -> list[str]:
        """Make operational paths match ``ref`` before integrating a branch."""
        changed: list[str] = []
        for path in paths:
            if not path or path.startswith(("/", "..")):
                continue
            exists_at_ref = self._git(
                "cat-file", "-e", f"{ref}:{path}", check=False
            ).ok
            if exists_at_ref:
                if self._git("checkout", ref, "--", path, check=False).ok:
                    changed.append(path)
                continue
            if (
                self._git("ls-files", "--error-unmatch", "--", path, check=False).ok
                and self._git("rm", "-f", "--", path, check=False).ok
            ):
                changed.append(path)
        return changed

    def has_commits(self) -> bool:
        return self._git("rev-parse", "--verify", "HEAD", check=False).ok

    # -- state -----------------------------------------------------------

    def head(self) -> str:
        return self._git("rev-parse", "HEAD").stdout.strip()

    def branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def is_dirty(self) -> bool:
        return bool(self._git("status", "--porcelain").stdout.strip())

    def status(self) -> list[tuple[str, str]]:
        entries = []
        for line in self._git("status", "--porcelain").stdout.splitlines():
            if len(line) > 3:
                entries.append((line[:2].strip(), line[3:]))
        return entries

    def changed_files(self, since: str | None = None) -> list[str]:
        if since:
            result = self._git("diff", "--name-only", f"{since}..HEAD")
        else:
            result = self._git("status", "--porcelain")
            return [line[3:] for line in result.stdout.splitlines() if len(line) > 3]
        return [line for line in result.stdout.splitlines() if line]

    def diff(self, ref: str | None = None, *, staged: bool = False, max_bytes: int = 200_000) -> str:
        """A unified diff, truncated so it cannot blow a context window."""
        args = ["diff", "--no-color", "--find-renames"]
        if staged:
            args.append("--cached")
        if ref:
            args.append(ref)
        text = self._git(*args).stdout
        if len(text) > max_bytes:
            text = text[:max_bytes] + f"\n... [diff truncated at {max_bytes} bytes] ...\n"
        return text

    def show(self, ref: str, path: str) -> str:
        return self._git("show", f"{ref}:{path}", check=False).stdout

    def log(self, limit: int = 20, ref: str | None = None) -> list[CommitInfo]:
        args = ["log", f"-{limit}", "--pretty=format:%H%x1f%s%x1f%an%x1f%at%x1f%(trailers:key=Forge-Node,valueonly)"]
        if ref:
            args.append(ref)
        result = self._git(*args, check=False)
        commits: list[CommitInfo] = []
        for line in result.stdout.splitlines():
            parts = line.split("\x1f")
            if len(parts) < 4:
                continue
            commits.append(
                CommitInfo(
                    sha=parts[0],
                    subject=parts[1],
                    author=parts[2],
                    timestamp=int(parts[3] or 0),
                    node_id=(parts[4].strip() or None) if len(parts) > 4 else None,
                )
            )
        return commits

    # -- writing ---------------------------------------------------------

    def add_all(self) -> None:
        self._git("add", "-A")

    def commit(
        self,
        message: str,
        *,
        node_id: str | None = None,
        allow_empty: bool = False,
        add_all: bool = True,
    ) -> str | None:
        """Commit the working tree. Returns the sha, or ``None`` if nothing changed.

        The node id goes into a trailer rather than the subject, so subjects stay
        readable while the mapping from commit to node remains machine-queryable.
        """
        if add_all:
            self.add_all()
        if not allow_empty and not self._git("diff", "--cached", "--quiet", check=False).returncode:
            return None
        body = message.strip()
        if node_id:
            body += f"\n\nForge-Node: {node_id}"
        args = ["commit", "-m", body, "--no-verify"]
        if allow_empty:
            args.append("--allow-empty")
        result = self._git(*args, check=False)
        if not result.ok:
            if "nothing to commit" in result.combined:
                return None
            raise GitError("commit failed", stderr=result.combined[-800:])
        return self.head()

    def tag(self, name: str, message: str = "") -> None:
        self._git("tag", "-f", "-a", name, "-m", message or name)

    def tags(self, pattern: str = "*") -> list[str]:
        return [t for t in self._git("tag", "-l", pattern, check=False).stdout.splitlines() if t]

    def reset_hard(self, ref: str = "HEAD") -> None:
        """Discard all local changes and move HEAD to ``ref``.

        Called before every node attempt. Combined with per-node commits this is
        what makes retries idempotent: attempt N+1 never inherits attempt N's
        half-finished edits.
        """
        with self._reset_lock:
            self._reset_unlocked(ref)

    def _reset_unlocked(self, ref: str) -> None:
        self._git("reset", "--hard", ref)
        args = ["clean", "-fdx"]
        for pattern in PRESERVED_ON_RESET:
            args += ["-e", pattern]
        # `git clean` exits non-zero when a path vanishes mid-walk, which is a
        # warning about a file that is already gone rather than a failure to
        # clean. Treating it as fatal blocked a node outright; the tree is in
        # exactly the desired state either way.
        result = self._git(*args, check=False)
        if not result.ok and not _only_vanished_paths(result.stderr):
            raise GitError(
                "could not reset the working tree",
                command=" ".join(args),
                stderr=result.stderr[:400],
            )

    def restore_paths(self, paths: Sequence[str]) -> list[str]:
        """Undo uncommitted changes to exactly these paths. Returns what moved.

        The narrow cousin of :meth:`reset_hard`, and the only safe form of
        "undo" while other workers are editing the same tree: a wholesale reset
        would delete a concurrent node's in-progress work, whereas this touches
        only the files one attempt is known to have written.

        Tracked paths go back to HEAD; untracked ones are removed, because at
        HEAD they did not exist. Failures are reported, not raised -- this runs
        on the failure path, where making the cleanup itself fatal would replace
        a retryable node with a dead one.
        """
        restored: list[str] = []
        for path in paths:
            if not path or path.startswith(("/", "..")):
                continue
            tracked = self._git("ls-files", "--error-unmatch", "--", path, check=False).ok
            if tracked:
                if self._git("checkout", "HEAD", "--", path, check=False).ok:
                    restored.append(path)
                continue
            target = self.path / path
            try:
                if target.is_file() or target.is_symlink():
                    target.unlink()
                    restored.append(path)
            except OSError:  # pragma: no cover - racing filesystem
                continue
        return restored

    def checkout(self, ref: str, *, create: bool = False) -> None:
        args = ["checkout"]
        if create:
            args.append("-B")
        args.append(ref)
        self._git(*args)

    def branch_exists(self, name: str) -> bool:
        return self._git("rev-parse", "--verify", f"refs/heads/{name}", check=False).ok

    def delete_branch(self, name: str) -> bool:
        """Delete an integrated task branch after its worktree is detached."""
        if not name or name == self.branch():
            return False
        return self._git("branch", "-D", name, check=False).ok

    def merge(self, ref: str, *, message: str = "") -> bool:
        """Merge ``ref`` into the current branch. False means conflicts.

        On conflict the merge is aborted so the tree is never left in a
        conflicted state -- an autonomous worker cannot be trusted to be
        halfway through a merge when its lease expires.
        """
        result = self._git("merge", "--no-edit", "-m", message or f"merge {ref}", ref, check=False)
        if result.ok:
            return True
        conflicts = self._git("diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
        log.warn(
            "merge conflict",
            ref=ref,
            files=conflicts[:10],
            error=result.tail(10)[:800],
        )
        self._git("merge", "--abort", check=False)
        return False

    def revert(self, sha: str) -> bool:
        result = self._git("revert", "--no-edit", sha, check=False)
        if not result.ok:
            self._git("revert", "--abort", check=False)
        return result.ok

    # -- worktrees -------------------------------------------------------

    def add_worktree(self, path: Path, branch: str, *, base: str = "HEAD") -> Repo:
        """Create an isolated checkout for parallel work."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.remove_worktree(path)
        self._git("worktree", "add", "-B", branch, str(path), base)
        log.debug("worktree created", path=str(path), branch=branch)
        return Repo(path, timeout=self.timeout)

    def ensure_worktree(self, path: Path, branch: str, *, base: str = "HEAD") -> Repo:
        """Return a persistent isolated checkout for one durable task.

        Unlike :meth:`add_worktree`, this never resets an existing branch. Failed
        attempts intentionally keep their provisional commits so a later local
        repair can continue from the best known state instead of starting over.
        """
        path = Path(path)
        if (path / ".git").exists():
            return Repo(path, timeout=self.timeout)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.remove_worktree(path)
        if self.branch_exists(branch):
            self._git("worktree", "add", str(path), branch)
        else:
            self._git("worktree", "add", "-b", branch, str(path), base)
        log.debug("persistent worktree ready", path=str(path), branch=branch)
        return Repo(path, timeout=self.timeout)

    def remove_worktree(self, path: Path) -> None:
        self._git("worktree", "remove", "--force", str(path), check=False)
        self._git("worktree", "prune", check=False)

    def list_worktrees(self) -> list[str]:
        out = self._git("worktree", "list", "--porcelain", check=False).stdout
        return [line.split(" ", 1)[1] for line in out.splitlines() if line.startswith("worktree ")]

    # -- inspection helpers used by context assembly ---------------------

    def tracked_files(self, pattern: str | None = None) -> list[str]:
        args = ["ls-files"]
        if pattern:
            args.append(pattern)
        return [f for f in self._git(*args, check=False).stdout.splitlines() if f]

    def file_stats(self) -> dict[str, int]:
        """Line counts per tracked file, for prioritising what to read."""
        stats: dict[str, int] = {}
        for name in self.tracked_files():
            path = self.path / name
            try:
                if path.is_file() and path.stat().st_size < 2_000_000:
                    stats[name] = sum(1 for _ in path.open("rb"))
            except OSError:  # pragma: no cover
                continue
        return stats

    def size_summary(self) -> dict[str, int]:
        stats = self.file_stats()
        return {"files": len(stats), "lines": sum(stats.values())}


def find_repo(start: Path) -> Repo | None:
    """Walk upwards looking for a repository root."""
    current = Path(start).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return Repo(candidate)
    return None


def git_available() -> bool:
    from ..util.proc import which

    return which("git") is not None and os.access("/", os.R_OK)
