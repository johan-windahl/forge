"""How generated code reaches disk.

Forge does not ask models for unified diffs. Diffs require the model to
reproduce line numbers and surrounding context exactly, and a single miscounted
line rejects the whole patch -- an expensive failure mode that grows worse as
files get longer. Instead the model emits an :class:`EditPlan`: a list of
whole-file writes and anchored replacements.

Anchored replacement ("replace this exact snippet with that one") is the sweet
spot. It is robust to line drift, it is verifiable before anything is written,
and when the anchor is ambiguous or missing the failure is specific enough to
repair in one turn: "anchor matched 3 times in src/app.ts".

Every plan is applied atomically: all edits validate against the current tree
first, and only then does anything get written. A half-applied plan is the one
outcome that is genuinely hard to recover from, so it is made impossible.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import PatchError
from ..obs.log import get_logger

log = get_logger("workspace.patch")

#: Refuse to touch anything outside the workspace, and anything that would let a
#: generated edit change how Forge itself behaves.
FORBIDDEN = (".git/", ".forge/")

MAX_FILE_BYTES = 4 * 1024 * 1024


@dataclass(slots=True)
class FileEdit:
    """One change to one file."""

    path: str
    op: str  # write | replace | insert_after | delete | create_dir
    content: str = ""
    anchor: str = ""  # for replace / insert_after
    #: Which match to act on, 1-based. ``None`` means the model did not say,
    #: which is only safe when the anchor matches exactly once -- see the
    #: ambiguity check in `_apply_anchor`.
    occurrence: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "op": self.op,
            "anchor_length": len(self.anchor),
            "content_length": len(self.content),
            "reason": self.reason,
        }


@dataclass(slots=True)
class EditPlan:
    """A set of edits applied as a unit."""

    edits: list[FileEdit] = field(default_factory=list)
    summary: str = ""
    #: Paths the model says it must read before it can write correctly. Honoured
    #: once per round by the coding agent, which then re-asks with them included
    #: and discards the edits made without them -- those were written against a
    #: guessed interface, which is the thing being avoided.
    need_files: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.edits)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EditPlan:
        # `or default` rather than `get(key, default)`: a strict-mode provider
        # sends every optional key explicitly as null, so the key is present and
        # `get` returns None. `int(None)` raised TypeError, which is not a
        # ForgeError, so it surfaced as an internal platform fault on a provider
        # that had done exactly what its schema asked of it.
        edits = [
            FileEdit(
                path=str(item["path"]),
                op=item.get("op") or "write",
                content=item.get("content") or "",
                anchor=item.get("anchor") or "",
                # Left as None when unsaid, rather than coerced to 1. Both
                # forms arrive: a strict-mode provider sends `null` for what it
                # has no opinion about, and `int(None)` used to raise TypeError
                # -- not a ForgeError, so it surfaced as an internal platform
                # fault on a provider that had followed its schema exactly. But
                # coercing to 1 threw away the distinction the ambiguity check
                # needs: "the first one" and "I did not consider that there
                # might be more than one" are not the same claim.
                occurrence=(
                    int(item["occurrence"])
                    if isinstance(item.get("occurrence"), int)
                    else None
                ),
                reason=item.get("reason") or "",
            )
            for item in payload.get("edits", [])
        ]
        wanted = payload.get("need_files") or []
        return cls(
            edits=edits,
            summary=payload.get("summary", ""),
            need_files=[str(p) for p in wanted if isinstance(p, str | int | float)][:32],
        )

    def paths(self) -> list[str]:
        return sorted({e.path for e in self.edits})


@dataclass(slots=True)
class ApplyResult:
    written: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    bytes_written: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "deleted": self.deleted,
            "bytes_written": self.bytes_written,
        }


def _safe_path(root: Path, relative: str) -> Path:
    """Resolve a path inside the workspace, rejecting escapes."""
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise PatchError("edit path must be relative and inside the workspace", path=relative)
    normalised = relative.replace("\\", "/")
    for prefix in FORBIDDEN:
        if normalised.startswith(prefix) or f"/{prefix}" in normalised:
            raise PatchError("edit targets a protected path", path=relative)
    target = (root / normalised).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise PatchError("edit path escapes the workspace", path=relative) from exc
    return target


def _apply_anchor(text: str, edit: FileEdit) -> str:
    count = text.count(edit.anchor)
    if count == 0:
        # Retry with whitespace-normalised matching before giving up. Models
        # reproduce indentation imperfectly far more often than they get the
        # actual code wrong.
        normalised = _find_normalised(text, edit.anchor)
        if normalised is None:
            raise PatchError(
                "anchor not found in file",
                path=edit.path,
                anchor_preview=edit.anchor[:160],
            )
        start, end = normalised
        replacement = edit.content if edit.op == "replace" else text[start:end] + edit.content
        return text[:start] + replacement + text[end:]
    # An anchor matching several places, with nothing saying which, is a
    # guess about which one the model meant -- and the guess is silent, so a
    # wrong edit to the first of three `return 1` lines looks exactly like a
    # right one. The old form (`edit.occurrence < 1`) could never fire: the
    # schema sets `minimum: 1` and the parser coerced everything else to 1.
    if count > 1 and edit.occurrence is None:
        raise PatchError(
            "anchor matches more than once; set `occurrence`, or extend the "
            "anchor with surrounding lines until it is unique",
            path=edit.path,
            matches=count,
        )

    index = -1
    for _ in range(max(1, edit.occurrence or 1)):
        index = text.find(edit.anchor, index + 1)
        if index < 0:
            raise PatchError(
                "anchor occurrence out of range",
                path=edit.path,
                requested=edit.occurrence,
                found=count,
            )
    end = index + len(edit.anchor)
    if edit.op == "replace":
        return text[:index] + edit.content + text[end:]
    return text[:end] + edit.content + text[end:]


def _find_normalised(text: str, anchor: str) -> tuple[int, int] | None:
    """Locate an anchor ignoring per-line leading/trailing whitespace."""
    anchor_lines = [line.strip() for line in anchor.strip().splitlines() if line.strip()]
    if not anchor_lines:
        return None
    lines = text.splitlines(keepends=True)
    stripped = [line.strip() for line in lines]
    for start in range(len(lines) - len(anchor_lines) + 1):
        window = [s for s in stripped[start : start + len(anchor_lines)]]
        if window == anchor_lines:
            offset = sum(len(line) for line in lines[:start])
            length = sum(len(line) for line in lines[start : start + len(anchor_lines)])
            return offset, offset + length
    return None


def _has_content(target: Path) -> bool:
    """True when the path is a file that currently holds something."""
    try:
        return target.is_file() and target.stat().st_size > 0
    except OSError:  # pragma: no cover - racing filesystem
        return False


def apply_edits(root: Path, plan: EditPlan, *, dry_run: bool = False) -> ApplyResult:
    """Validate then apply an entire plan, or change nothing.

    The two-phase structure is the point. Phase one computes the final content
    of every touched file in memory and raises on the first problem; phase two
    writes. A model that gets edit 4 of 6 wrong therefore leaves the tree
    exactly as it found it, and the repair prompt describes one clean failure
    rather than a partially mutated codebase.
    """
    root = Path(root)
    staged: dict[Path, str | None] = {}  # None means delete
    directories: list[Path] = []

    for edit in plan.edits:
        target = _safe_path(root, edit.path)
        match edit.op:
            case "create_dir":
                directories.append(target)
                continue
            case "delete":
                if target not in staged and not target.exists():
                    raise PatchError("cannot delete a file that does not exist", path=edit.path)
                staged[target] = None
                continue
            case "write":
                if len(edit.content.encode("utf-8")) > MAX_FILE_BYTES:
                    raise PatchError("file too large", path=edit.path)
                # A write with no content blanks the file. `content` is optional
                # in the schema, so a model that names a path and then runs out
                # of output budget mid-plan emits exactly this, and the result
                # is a zero-byte source file that every later gate reports as
                # something else entirely -- "is not a module", not "is empty".
                # The pinball run lost an attempt to one such file: the round
                # read the blank module, failed to infer what belonged in it,
                # and left it blank. Emptying a file on purpose is what
                # `delete` is for.
                #
                # A directory in the file's place has to be caught here too. The
                # write itself succeeds -- it is the atomic rename onto the
                # target that fails, with `IsADirectoryError` from deep inside
                # phase two, after other files have already landed. That broke
                # the all-or-nothing guarantee this function exists to provide,
                # and it threw away 13KB of correct generated code because a
                # stray `create_dir` had claimed the path.
                if target.is_dir():
                    raise PatchError(
                        "a directory already exists at this path, so a file cannot "
                        "be written there; delete it first or choose another path",
                        path=edit.path,
                    )
                if not edit.content.strip() and _has_content(target):
                    raise PatchError(
                        "a write with empty content would blank an existing file; "
                        "send the full new content, or use op 'delete' to remove it",
                        path=edit.path,
                    )
                staged[target] = edit.content
                continue
            case "replace" | "insert_after":
                if target in staged:
                    current = staged[target]
                    if current is None:
                        raise PatchError("edit follows a delete in the same plan", path=edit.path)
                else:
                    if not target.is_file():
                        raise PatchError("cannot edit a file that does not exist", path=edit.path)
                    current = target.read_text(encoding="utf-8", errors="replace")
                if not edit.anchor:
                    raise PatchError(f"{edit.op} requires an anchor", path=edit.path)
                staged[target] = _apply_anchor(current, edit)
                continue
            case _:
                raise PatchError(f"unknown edit operation {edit.op!r}", path=edit.path)

    # A plan that creates a directory *and* writes a file at the same path
    # cannot be satisfied, and finding out during phase two would leave the tree
    # half-changed. The two operations are validated separately above, so the
    # contradiction is only visible once both lists exist.
    conflicts = sorted(set(directories) & {p for p, c in staged.items() if c is not None})
    if conflicts:
        raise PatchError(
            "the plan both creates a directory and writes a file at this path",
            path=str(conflicts[0].relative_to(root)),
        )

    result = ApplyResult()
    if dry_run:
        result.written = [str(p.relative_to(root)) for p, c in staged.items() if c is not None]
        result.deleted = [str(p.relative_to(root)) for p, c in staged.items() if c is None]
        return result

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    for target, content in staged.items():
        relative = str(target.relative_to(root))
        if content is None:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            result.deleted.append(relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        # Write-then-rename: a crash mid-write leaves the old file intact.
        tmp = target.with_name(target.name + ".forge-tmp")
        tmp.write_bytes(data)
        try:
            tmp.replace(target)
        except OSError as exc:
            # Never leave the scratch file behind. It sits next to real source,
            # git reports it as untracked, and the next `git clean` is the only
            # thing that removes it -- one was found holding the only copy of
            # 13KB of correct generated code after a failed rename.
            tmp.unlink(missing_ok=True)
            raise PatchError(f"could not write the file: {exc}", path=relative) from exc
        result.written.append(relative)
        result.bytes_written += len(data)

    log.debug(
        "edit plan applied",
        written=len(result.written),
        deleted=len(result.deleted),
        bytes=result.bytes_written,
    )
    return result


#: JSON Schema for an edit plan, shared by every agent that writes code.
EDIT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "One line describing the change, imperative mood",
        },
        "need_files": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Paths you must read before you can write correctly. Use this "
                "instead of guessing at an interface. When you set this, send an "
                "empty `edits` list: you will be asked again with these files "
                "included. Leave this empty if you have what you need."
            ),
        },
        "edits": {
            "type": "array",
            # May be empty, but only when `need_files` is set. Requiring an edit
            # here forced a model that merely wanted to read a file to invent a
            # throwaway one, and its own `reason` said so: "Schema requires at
            # least one edit, but need_files is set so this is discarded." One of
            # those throwaways was applied and committed as a node's deliverable.
            # The caller rejects a plan that is empty of both.
            "minItems": 0,
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the project root"},
                    "op": {
                        "type": "string",
                        "enum": ["write", "replace", "insert_after", "delete", "create_dir"],
                        "description": "write replaces the whole file; replace swaps an exact anchor",
                    },
                    "content": {"type": "string", "description": "New content, or replacement text"},
                    "anchor": {
                        "type": "string",
                        "description": "Exact existing text to match, for replace/insert_after",
                    },
                    "occurrence": {"type": "integer", "minimum": 1, "description": "Which match, 1-based"},
                    "reason": {"type": "string", "description": "Why this edit is needed"},
                },
                "required": ["path", "op"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "edits"],
    "additionalProperties": False,
}
