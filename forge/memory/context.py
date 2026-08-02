"""Context assembly: deciding what the model gets to see.

This is where token efficiency is won or lost. The naive approach -- append
everything and hope -- costs money on every call, degrades quality once the
prompt exceeds what the model attends to well, and grows without bound over a
multi-day run.

Forge assembles every prompt from *sections*, each with a priority and a token
ceiling. Sections are filled in priority order until the budget is spent, and
each one knows how to shrink itself rather than simply being dropped. The result
is a prompt that is the same shape every time, so the model learns where to look,
and whose expensive parts degrade gracefully instead of falling off a cliff.

Two properties are engineered deliberately:

**Stable prefix.** Sections that do not change within a node -- role, project
digest, conventions -- are emitted first and marked as a cache breakpoint. On
providers with prompt caching this makes repeat calls within a node cost a
fraction of their nominal input tokens, which is the single largest cloud saving
available.

**Evidence last.** The task and the most recent failure output go at the end,
nearest the generation point, because that is where models weight most heavily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models.types import Message, estimate_tokens
from ..obs.log import get_logger
from ..workspace.references import ReferenceStore
from .records import MemoryKind, MemoryRecord

log = get_logger("memory.context")


@dataclass(slots=True)
class Section:
    """One labelled block of prompt content."""

    name: str
    content: str
    #: Lower numbers are filled first and are the last to be trimmed.
    priority: int = 100
    #: Hard ceiling for this section, in estimated tokens. 0 means unbounded.
    max_tokens: int = 0
    #: Sections marked stable form the cacheable prefix.
    stable: bool = False
    #: If set, the section is trimmed by keeping this many lines from each end.
    head_lines: int = 0
    tail_lines: int = 0

    def tokens(self) -> int:
        return estimate_tokens(self.content)

    def trimmed(self, budget: int) -> str:
        """Shrink to fit ``budget`` tokens, keeping the informative parts.

        The elision marker counts against the budget. Forgetting that is how a
        packer quietly overshoots its ceiling by exactly the amount it spends
        announcing that it trimmed.
        """
        if budget <= 0:
            return ""
        if self.tokens() <= budget:
            return self.content

        if self.head_lines or self.tail_lines:
            lines = self.content.splitlines()
            head = lines[: self.head_lines] if self.head_lines else []
            tail = lines[-self.tail_lines :] if self.tail_lines else []
            elided = len(lines) - len(head) - len(tail)
            if elided > 0:
                candidate = "\n".join([*head, f"... [{elided} lines omitted] ...", *tail])
                if estimate_tokens(candidate) <= budget:
                    return candidate

        marker = "\n... [truncated to fit context budget] ..."
        room = budget - estimate_tokens(marker)
        if room <= 0:
            return ""
        keep = int(len(self.content) * room / max(1, self.tokens()))

        # A section that asked for a tail must not be handed a head. The
        # head/tail candidate above is only attempted at the *requested* sizes;
        # when those do not fit -- which is exactly when the section is huge and
        # trimming matters most -- this is where it lands. Falling through to a
        # plain prefix here silently inverted the caller's request: a validation
        # section asking for the last 80 lines of vitest output got the npm
        # banner and lost the failing assertion.
        if self.tail_lines and not self.head_lines:
            cut = self.content[-keep:] if keep else ""
            if "\n" in cut:
                cut = cut.split("\n", 1)[1]
            return marker + "\n" + cut
        if self.head_lines and self.tail_lines:
            half = max(1, keep // 2)
            head_text = self.content[:half]
            tail_text = self.content[-half:]
            if "\n" in head_text:
                head_text = head_text.rsplit("\n", 1)[0]
            if "\n" in tail_text:
                tail_text = tail_text.split("\n", 1)[1]
            return head_text + marker + "\n" + tail_text

        cut = self.content[:keep]
        if "\n" in cut:
            cut = cut.rsplit("\n", 1)[0]
        return cut + marker


class ContextBuilder:
    """Assembles a message list under a token budget.

    Usage is intentionally declarative: callers add named sections and the
    builder decides what survives. Agents therefore describe *what is relevant*
    and never do arithmetic about context windows.
    """

    def __init__(self, budget_tokens: int = 24_000) -> None:
        self.budget = budget_tokens
        self._sections: list[Section] = []
        self._images: list[Any] = []

    # -- composition -----------------------------------------------------

    def add(
        self,
        name: str,
        content: str,
        *,
        priority: int = 100,
        max_tokens: int = 0,
        stable: bool = False,
        head_lines: int = 0,
        tail_lines: int = 0,
    ) -> ContextBuilder:
        if content and content.strip():
            self._sections.append(
                Section(
                    name=name,
                    content=content.strip(),
                    priority=priority,
                    max_tokens=max_tokens,
                    stable=stable,
                    head_lines=head_lines,
                    tail_lines=tail_lines,
                )
            )
        return self

    def add_image(self, image: Any) -> ContextBuilder:
        self._images.append(image)
        return self

    def add_records(
        self,
        name: str,
        records: list[MemoryRecord],
        *,
        priority: int = 100,
        max_tokens: int = 0,
        stable: bool = False,
        verbose: bool = False,
    ) -> ContextBuilder:
        if not records:
            return self
        body = "\n\n".join(record.render(verbose=verbose) for record in records)
        return self.add(name, body, priority=priority, max_tokens=max_tokens, stable=stable)

    def add_files(
        self,
        name: str,
        files: dict[str, str],
        *,
        priority: int = 60,
        max_tokens: int = 0,
        stable: bool = False,
    ) -> ContextBuilder:
        """Include file contents with line numbers.

        Line numbers are worth their tokens: they let the model point at
        locations precisely in its output, and they make anchored edits far more
        reliable because the model can see whether an anchor is unique.
        """
        if not files:
            return self
        blocks = []
        for path, content in files.items():
            numbered = "\n".join(
                f"{i:>5} | {line}" for i, line in enumerate(content.splitlines(), start=1)
            )
            blocks.append(f"--- {path} ---\n{numbered}")
        return self.add(name, "\n\n".join(blocks), priority=priority, max_tokens=max_tokens, stable=stable)

    # -- rendering -------------------------------------------------------

    def _allocate(self) -> list[tuple[Section, str]]:
        ordered = sorted(self._sections, key=lambda s: (s.priority, s.name))
        remaining = self.budget
        rendered: list[tuple[Section, str]] = []
        for section in ordered:
            if remaining <= 0:
                log.debug("context budget exhausted, dropping section", section=section.name)
                continue
            ceiling = section.max_tokens or remaining
            allowance = min(ceiling, remaining)
            text = section.trimmed(allowance)
            if not text:
                continue
            used = estimate_tokens(text)
            remaining -= used
            rendered.append((section, text))
        return rendered

    def build(self, *, system_prompt: str, task: str) -> list[Message]:
        """Produce the final message list.

        Layout, in order: role instructions, stable context (cacheable), volatile
        context, then the task. The single cache breakpoint sits at the end of
        the stable block.
        """
        rendered = self._allocate()
        stable = [(s, t) for s, t in rendered if s.stable]
        volatile = [(s, t) for s, t in rendered if not s.stable]

        messages: list[Message] = [Message("system", system_prompt.strip())]

        if stable:
            body = "\n\n".join(f"## {s.name}\n{t}" for s, t in stable)
            messages.append(Message("system", body, cache_breakpoint=True))

        if volatile:
            body = "\n\n".join(f"## {s.name}\n{t}" for s, t in volatile)
            messages.append(Message("system", body))

        messages.append(Message("user", task.strip(), images=list(self._images)))
        return messages

    def report(self) -> dict[str, Any]:
        """What went in and what it cost. Logged with every agent call."""
        rendered = self._allocate()
        return {
            "budget": self.budget,
            "used": sum(estimate_tokens(t) for _, t in rendered),
            "sections": [
                {
                    "name": s.name,
                    "tokens": estimate_tokens(t),
                    "trimmed": estimate_tokens(t) < s.tokens(),
                    "stable": s.stable,
                }
                for s, t in rendered
            ],
            "dropped": [
                s.name for s in self._sections if all(s.name != r.name for r, _ in rendered)
            ],
        }


# --------------------------------------------------------------------------
# Standard section priorities
# --------------------------------------------------------------------------

# Lower survives longer. The ordering encodes a claim about what an agent can
# least afford to lose: it can write mediocre code without the style guide, but
# it cannot write *correct* code without the interfaces it must satisfy or the
# error it is supposed to fix.
P_GOAL = 10
P_ACCEPTANCE = 15
P_ARCHITECTURE = 20
P_INTERFACES = 25
P_FAILURE = 30  # the error being fixed, if any
P_TASK_FILES = 40  # files the node will edit
P_CONVENTIONS = 50
P_MEMORY = 60  # retrieved records
P_RELATED_FILES = 70
P_LESSONS = 80
P_HISTORY = 90  # recent activity summary
P_TREE = 95  # file listing


def read_files(root: Path, paths: list[str], *, max_bytes_each: int = 60_000) -> dict[str, str]:
    """Read files for inclusion in context, skipping what cannot help.

    Binary files and generated lockfiles are excluded: they consume enormous
    context and no model has ever needed to read one to complete a task.
    """
    skip_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".woff", ".woff2", ".ttf",
                     ".mp4", ".webm", ".wasm", ".zip", ".gz", ".pdf"}
    skip_names = {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "Cargo.lock", "poetry.lock"}
    out: dict[str, str] = {}
    try:
        base = root.resolve()
    except OSError:
        return out
    for relative in paths:
        path = root / relative
        if path.name in skip_names or path.suffix.lower() in skip_suffixes:
            continue
        try:
            # Every path here came from a model, one way or another: planner
            # JSON via `spec["paths"]`, or a file request. `_grant_files`
            # checks its own, but this is the chokepoint they all pass through,
            # and a planned path of "../../.ssh/id_rsa" was read and placed in
            # the prompt. Resolve first: `is_file()` is happy to follow a
            # symlink out of the workspace.
            if not path.resolve().is_relative_to(base):
                continue
            if not path.is_file():
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:2048]:
            continue
        out[relative] = data[:max_bytes_each].decode("utf-8", "replace")
    return out


#: Names that describe a *derived diagnostic* rather than a picture of the
#: thing to build. An operator dropping supporting material into the reference
#: folder is normal; letting one of those files outrank the actual artwork is
#: not. Observed for real: five references were supplied for a pinball table
#: and the goal check received `nightmare-audio-spectrogram.png`, because it
#: asked for exactly one and sorted alphabetically. It then judged whether the
#: game resembled a frequency plot.
_DERIVED_REFERENCE_HINTS = (
    "spectrogram",
    "waveform",
    "audio",
    "contact-sheet",
    "contactsheet",
    "poster",
    "thumbnail",
    "thumb",
    "diff",
)


def _reference_rank(path: Path) -> tuple[int, str]:
    """Sort key: likely primary references first, then stable alphabetical."""
    name = path.name.lower()
    derived = any(hint in name for hint in _DERIVED_REFERENCE_HINTS)
    return (1 if derived else 0, name)


def reference_images(root: Path, *, limit: int = 6) -> list[Path]:
    """Return durable human-supplied visual references, best first.

    References can be versioned in ``docs/references`` or kept as operator-owned
    state in ``.forge/references`` beside the workspace. The latter lets a human
    steer a live run without dirtying an implementation node's Git tree. Keep
    discovery narrow: arbitrary project images are product assets, not
    necessarily examples to imitate.

    Ordering matters more than it looks. Callers truncate, so whatever sorts
    first is what a vision model actually gets to compare against. Plain
    alphabetical order meant a supporting artifact could silently displace the
    artwork it was derived from, and supplying *more* reference material made
    the comparison worse. Derived diagnostics therefore sort last, and the
    default limit is high enough to carry a normal reference set intact.
    """
    return [path for path, _ in reference_images_described(root, limit=limit)]


def reference_images_described(root: Path, *, limit: int = 6) -> list[tuple[Path, str]]:
    """Reference images paired with the operator's description of each.

    The description is the half that used to be missing. The same screenshot can
    mean "match this composition", "match this palette but not the layout", or
    "this is what we are replacing", and a model shown the bare file has to
    guess which -- differently each time it is asked.

    A declared manifest wins over the filename heuristic below, because a
    declaration cannot be wrong about its own role and a substring match can.
    """
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    forge_dir = root.parent / ".forge"

    described: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    store = ReferenceStore(forge_dir)
    if store.manifest_path.is_file():
        for ref in store.load():
            path = store.path_of(ref)
            if ref.role == "visual" and path.is_file() and path.suffix.lower() in suffixes:
                described.append((path, ref.description))
                seen.add(path)

    found: list[Path] = []
    for directory in (root / "docs" / "references", forge_dir / "references"):
        if not directory.is_dir():
            continue
        found.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes and path not in seen
        )
    described.extend((path, "") for path in sorted(found, key=_reference_rank))
    return described[:limit]


def file_tree(root: Path, *, limit: int = 400, exclude: set[str] | None = None) -> str:
    """A compact tree listing, for orientation rather than detail."""
    exclude = exclude or {".git", "node_modules", "__pycache__", ".venv", "dist", "build", ".forge"}
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in exclude for part in rel.parts):
            continue
        if path.is_dir():
            continue
        lines.append(rel.as_posix())
        if len(lines) >= limit:
            lines.append(f"... [listing truncated at {limit} files] ...")
            break
    return "\n".join(lines)


def summarize_records_for_digest(records: list[MemoryRecord]) -> str:
    """One-line-per-record digest used in the stable prefix."""
    order = [
        MemoryKind.REQUIREMENT,
        MemoryKind.DECISION,
        MemoryKind.INTERFACE,
        MemoryKind.CONVENTION,
        MemoryKind.ASSUMPTION,
    ]
    by_kind: dict[str, list[MemoryRecord]] = {}
    for record in records:
        by_kind.setdefault(record.kind, []).append(record)
    parts: list[str] = []
    for kind in order:
        items = by_kind.get(kind, [])
        if not items:
            continue
        parts.append(f"{kind.title()}s:")
        parts += [f"  - {r.title}" for r in items[:20]]
    return "\n".join(parts)
