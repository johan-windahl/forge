"""Operator-supplied reference material: what "done" is supposed to look like.

A goal sentence says what to build. It cannot say what good looks like, and for
anything with a visual, audible or structural target that is most of the
requirement. References close that gap: pictures of the thing, a video of it
moving, a spec it has to satisfy, an example output it should resemble.

Two properties matter more than they look:

*Descriptions.* A bare file is ambiguous. The same screenshot can be supplied as
"match this composition", "match this palette, ignore the layout", or "this is
what we are replacing". Without the operator's sentence, a model has to guess
which, and it guesses differently every time it is asked.

*Stability.* A reference is only useful for comparison if it is byte-identical
today and next week. Remote material is therefore fetched **once**, at add time,
and stored locally with its hash. A run never reaches the network for a
reference: a URL that changes underneath a project would silently move the
target between milestones, and a URL that 404s mid-run would make a comparison
fail for a reason that has nothing to do with the work.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..errors import ForgeError
from ..util.clock import Clock, default_clock, iso
from ..util.hashing import file_hash

MANIFEST_NAME = "references.json"
REFERENCES_DIRNAME = "references"

#: Refuse anything larger. A reference is material to compare against, not a
#: dataset; a 2 GB download here is a mistake, and finding out after it lands is
#: worse than a clear error.
MAX_BYTES = 200 * 1024 * 1024

FETCH_TIMEOUT = 60.0

#: JSON, not TOML: the core takes no third-party dependencies and the standard
#: library reads TOML but cannot write it. Hand-rolling TOML emission to keep a
#: file format symmetrical is a worse trade than a manifest a human can still
#: read and edit.
MANIFEST_VERSION = 1

_ROLE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "visual": (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"),
    "motion": (".mp4", ".mov", ".webm", ".mkv", ".avi", ".gif"),
    "audio": (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"),
    "document": (".md", ".txt", ".pdf", ".rst", ".html", ".htm"),
    "example": (".json", ".yaml", ".yml", ".toml", ".csv", ".xml"),
}


class ReferenceError(ForgeError):
    """A reference could not be acquired."""


@dataclass(slots=True)
class Reference:
    """One piece of durable reference material, as stored."""

    file: str
    description: str = ""
    #: visual | motion | audio | document | example | other
    role: str = "other"
    #: Where it came from: a URL, or the absolute path it was copied from.
    source: str = ""
    sha256: str = ""
    bytes: int = 0
    added_at: str = ""
    #: Set on material Forge produced from another reference (a contact sheet
    #: from a video). Derived items never outrank what they came from.
    derived_from: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_derived(self) -> bool:
        return bool(self.derived_from)

    def label(self) -> str:
        """One line for a prompt: the filename plus why it is here."""
        return f"{self.file} -- {self.description}" if self.description else self.file


def infer_role(name: str) -> str:
    """Guess a role from a filename, for when the operator did not say."""
    suffix = Path(name).suffix.lower()
    # Order matters: .gif is both a still and a motion format, and as a
    # reference it is almost always being supplied for the motion.
    for role in ("motion", "visual", "audio", "document", "example"):
        if suffix in _ROLE_SUFFIXES[role]:
            return role
    return "other"


def is_url(source: str) -> bool:
    return urllib.parse.urlparse(source).scheme in {"http", "https"}


def _safe_name(source: str) -> str:
    """A filesystem-safe basename derived from a URL or path."""
    if is_url(source):
        raw = Path(urllib.parse.unquote(urllib.parse.urlparse(source).path)).name
    else:
        raw = Path(source).name
    cleaned = "".join(c if c.isalnum() or c in "._-" else "-" for c in raw).strip("-._")
    return cleaned or "reference"


class ReferenceStore:
    """The reference directory plus its manifest.

    Lives under ``.forge/references`` so a human can steer a live run by adding
    material without touching an implementation node's Git tree.
    """

    def __init__(self, forge_dir: Path, *, clock: Clock | None = None) -> None:
        self.root = Path(forge_dir) / REFERENCES_DIRNAME
        self._clock = clock or default_clock()

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    # -- reading ---------------------------------------------------------

    def load(self) -> list[Reference]:
        """Every declared reference, primary material before derived.

        Falls back to listing the directory when there is no manifest, so a
        directory populated by hand before this existed still works. Those
        entries carry no description, which is exactly the gap the manifest
        closes.
        """
        declared = self._load_manifest()
        known = {ref.file for ref in declared}
        for path in sorted(self.root.glob("*")):
            if path.is_file() and path.name != MANIFEST_NAME and path.name not in known:
                declared.append(
                    Reference(file=path.name, role=infer_role(path.name), bytes=_size(path))
                )
        declared.sort(key=lambda ref: (ref.is_derived, ref.file))
        return declared

    def declared(self) -> list[Reference]:
        """Only what the manifest actually says, without the directory scan.

        `load()` deliberately blends in hand-dropped files so nothing is lost.
        Callers that treat a declaration as authoritative -- "this role was
        chosen by a human" -- need the narrower set, because a synthesised entry
        carries a *guessed* role and no description at all.
        """
        refs = self._load_manifest()
        refs.sort(key=lambda ref: (ref.is_derived, ref.file))
        return refs

    def by_role(self, *roles: str) -> list[Reference]:
        wanted = set(roles)
        return [ref for ref in self.load() if ref.role in wanted]

    def path_of(self, ref: Reference) -> Path:
        return self.root / ref.file

    def _load_manifest(self) -> list[Reference]:
        if not self.manifest_path.is_file():
            return []
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        out: list[Reference] = []
        for entry in data.get("references", []):
            if not isinstance(entry, dict) or not entry.get("file"):
                continue
            known = {f.name for f in fields(Reference)}
            out.append(Reference(**{k: v for k, v in entry.items() if k in known}))
        return out

    # -- writing ---------------------------------------------------------

    def add(
        self,
        source: str,
        *,
        description: str = "",
        role: str = "",
        derived_from: str = "",
    ) -> Reference:
        """Acquire one reference and record it. Returns what was stored.

        Idempotent by content: re-adding the same bytes updates the description
        rather than leaving two copies with different explanations of the same
        picture.
        """
        source = source.strip()
        if not source:
            raise ReferenceError("no reference source given")
        self.root.mkdir(parents=True, exist_ok=True)

        target = self._free_path(_safe_name(source))
        if is_url(source):
            _download(source, target)
        else:
            origin = Path(source).expanduser()
            if not origin.is_file():
                raise ReferenceError("reference not found", path=str(origin))
            if _size(origin) > MAX_BYTES:
                raise ReferenceError(
                    "reference is too large", path=str(origin), limit_bytes=MAX_BYTES
                )
            shutil.copy2(origin, target)
            source = str(origin.resolve())

        digest = file_hash(target)
        existing = self._match_digest(digest, skip=target.name)
        if existing is not None:
            target.unlink(missing_ok=True)
            # Re-adding is how an operator corrects a reference, so everything
            # they supplied this time has to land. Carrying only the description
            # made `--role` silently inert on material already in the store,
            # while still reporting success.
            existing.description = description.strip() or existing.description
            existing.role = role or existing.role
            existing.derived_from = derived_from or existing.derived_from
            existing.source = existing.source or source
            existing.sha256 = existing.sha256 or digest
            self._save([existing if r.file == existing.file else r for r in self.load()])
            return existing

        ref = Reference(
            file=target.name,
            description=description.strip(),
            role=role or infer_role(target.name),
            source=source,
            sha256=digest,
            bytes=_size(target),
            added_at=iso(self._clock.now()),
            derived_from=derived_from,
        )
        self._save([r for r in self.load() if r.file != ref.file] + [ref])
        return ref

    def _match_digest(self, digest: str, *, skip: str) -> Reference | None:
        """Find already-stored bytes, hashing on demand where needed.

        Entries synthesised by the directory scan have no recorded hash, so a
        plain manifest comparison misses them and `forge reference add` on a file
        the operator had already dropped in produced a second copy -- one with
        their description, one without, both sent to the same comparison.
        """
        for ref in self.load():
            if ref.file == skip:
                continue
            known = ref.sha256 or _hash_or_empty(self.path_of(ref))
            if known and known == digest:
                return ref
        return None

    def _free_path(self, name: str) -> Path:
        target = self.root / name
        if not target.exists():
            return target
        stem, suffix = target.stem, target.suffix
        for n in range(2, 1000):
            candidate = self.root / f"{stem}-{n}{suffix}"
            if not candidate.exists():
                return candidate
        raise ReferenceError("too many references with the same name", name=name)

    def _save(self, refs: list[Reference]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # Adopting a hand-dropped file into the manifest is the moment to record
        # what it is. Writing the synthesised entry through unchanged would make
        # the missing hash permanent, and dedup depends on it.
        for ref in refs:
            if not ref.sha256:
                ref.sha256 = _hash_or_empty(self.path_of(ref))
            if not ref.bytes:
                ref.bytes = _size(self.path_of(ref))
        refs.sort(key=lambda ref: (ref.is_derived, ref.file))
        payload = {
            "version": MANIFEST_VERSION,
            "references": [asdict(ref) for ref in refs],
        }
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.manifest_path)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _hash_or_empty(path: Path) -> str:
    try:
        return file_hash(path)
    except OSError:
        return ""


#: Content types that mean "you did not get the file you asked for" when the
#: thing requested was a picture, a clip or a sound: a soft 404, a login wall or
#: a consent interstitial, all served with status 200.
_MARKUP_TYPES = ("text/html", "application/xhtml")


def _download(url: str, target: Path) -> None:
    """Fetch a reference once, refusing anything implausibly large.

    Streamed and size-checked while reading rather than trusting
    ``Content-Length``: a server that lies about the length would otherwise
    write it to disk anyway.

    The content type is checked too, because a reference is fetched once and
    then compared against for the life of the project. An error page served with
    status 200 under an ``.png`` URL would be stored, typed `visual` from its
    suffix, base64-encoded as an image and sent to the vision model on every
    goal check -- failing, if at all, as an unrelated provider error days later.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "forge/reference-fetch"})
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            wanted = infer_role(target.name)
            if wanted in {"visual", "motion", "audio"} and content_type.startswith(_MARKUP_TYPES):
                raise ReferenceError(
                    "reference is a web page, not the file it claims to be",
                    url=url,
                    content_type=content_type,
                )
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_BYTES:
                raise ReferenceError(
                    "reference is too large", url=url, bytes=int(declared), limit_bytes=MAX_BYTES
                )
            written = 0
            with target.open("wb") as out:
                while chunk := response.read(1 << 16):
                    written += len(chunk)
                    if written > MAX_BYTES:
                        out.close()
                        target.unlink(missing_ok=True)
                        raise ReferenceError(
                            "reference is too large", url=url, limit_bytes=MAX_BYTES
                        )
                    out.write(chunk)
    except urllib.error.HTTPError as exc:
        target.unlink(missing_ok=True)
        raise ReferenceError("could not fetch reference", url=url, status=exc.code) from exc
    except ReferenceError:
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise ReferenceError("could not fetch reference", url=url, error=str(exc)) from exc
