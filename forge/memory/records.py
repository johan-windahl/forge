"""The vocabulary of project memory.

Records are *typed* rather than free-form notes, and the types were chosen by
asking what a returning engineer needs to know that the code does not already
say:

``ASSUMPTION``
    Something Forge decided on the human's behalf because asking would have
    stalled the run. These are the platform's substitute for a requirements
    conversation. They carry a confidence and are revisited: an assumption
    contradicted by later evidence gets superseded, and the contradiction is the
    single most valuable input to the next planning pass.

``DECISION``
    An architectural choice with its alternatives and rationale -- an ADR. The
    reason it must be a record and not a comment is that the *rejected* options
    matter: without them a later agent re-proposes them every time.

``INTERFACE``
    The contract between modules. Lets an agent implement against a boundary
    without reading the module behind it, which is the single biggest lever on
    prompt size in a large codebase.

``CONVENTION``
    How this project does things. Prevents the drift where every file is written
    in a slightly different style because a different context produced it.

``FACT``
    Observed properties of the world: the build takes 40 seconds, the dev server
    binds 5173, this dependency needs a native toolchain.

``LESSON``
    Transferable knowledge about *how to work*, not about this project. Lessons
    are the only kind promoted to the global library and reused across projects.

``FINDING``
    An open problem from a review or gate that has not been resolved yet.

``DIGEST``
    A rollup that replaces a large set of older records. Compaction, made
    explicit and auditable rather than hidden inside a summarisation call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..util.clock import iso
from ..util.ids import new_id


class MemoryKind(StrEnum):
    ASSUMPTION = "assumption"
    DECISION = "decision"
    INTERFACE = "interface"
    CONVENTION = "convention"
    FACT = "fact"
    LESSON = "lesson"
    FINDING = "finding"
    DIGEST = "digest"
    REQUIREMENT = "requirement"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"  # findings that have been fixed
    REJECTED = "rejected"


#: Weights used when ranking retrieval hits. Decisions and interfaces are
#: load-bearing: getting them wrong produces code that does not fit the system,
#: which is far more expensive than a stylistic miss.
KIND_WEIGHTS: dict[str, float] = {
    MemoryKind.REQUIREMENT: 1.6,
    MemoryKind.DECISION: 1.5,
    MemoryKind.INTERFACE: 1.4,
    MemoryKind.ASSUMPTION: 1.2,
    MemoryKind.CONVENTION: 1.1,
    MemoryKind.LESSON: 1.1,
    MemoryKind.FINDING: 1.0,
    MemoryKind.FACT: 0.9,
    MemoryKind.DIGEST: 0.8,
}


@dataclass(slots=True)
class MemoryRecord:
    """One durable thing Forge knows."""

    kind: str
    title: str
    body: str
    id: str = field(default_factory=lambda: new_id("mem"))
    tags: list[str] = field(default_factory=list)
    #: How much to trust this. Assumptions start low; facts observed from a
    #: command's exit code start at 1.0.
    confidence: float = 0.7
    status: str = MemoryStatus.ACTIVE
    #: Where it came from: node id, gate name, or "human".
    source: str = "system"
    created_at: float = 0.0
    updated_at: float = 0.0
    superseded_by: str | None = None
    #: Free-form structured payload, e.g. the alternatives of a decision.
    data: dict[str, Any] = field(default_factory=dict)
    #: Paths this record is about, used to boost retrieval when editing them.
    paths: list[str] = field(default_factory=list)

    @property
    def weight(self) -> float:
        return KIND_WEIGHTS.get(self.kind, 1.0) * (0.5 + 0.5 * self.confidence)

    def searchable(self) -> str:
        return " ".join([self.title, self.body, " ".join(self.tags), " ".join(self.paths)])

    def render(self, *, verbose: bool = False) -> str:
        """The form that goes into a prompt. Terse by design."""
        head = f"[{self.kind}] {self.title}"
        if self.kind == MemoryKind.ASSUMPTION:
            head += f" (confidence {self.confidence:.0%})"
        body = self.body.strip()
        if not verbose and len(body) > 600:
            body = body[:600].rsplit(" ", 1)[0] + " …"
        lines = [head, body]
        if verbose and self.data.get("alternatives"):
            lines.append("Rejected: " + "; ".join(str(a) for a in self.data["alternatives"]))
        return "\n".join(line for line in lines if line)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "tags": self.tags,
            "confidence": self.confidence,
            "status": self.status,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "superseded_by": self.superseded_by,
            "data": self.data,
            "paths": self.paths,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        return cls(
            id=data.get("id") or new_id("mem"),
            kind=data["kind"],
            title=data["title"],
            body=data.get("body", ""),
            tags=list(data.get("tags", [])),
            confidence=float(data.get("confidence", 0.7)),
            status=data.get("status", MemoryStatus.ACTIVE),
            source=data.get("source", "system"),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
            superseded_by=data.get("superseded_by"),
            data=dict(data.get("data", {})),
            paths=list(data.get("paths", [])),
        )

    def as_markdown(self) -> str:
        """Human-facing rendering for the project docs Forge maintains."""
        lines = [f"### {self.title}", ""]
        if self.kind == MemoryKind.DECISION:
            lines.append(f"*Decided {iso(self.created_at)} by {self.source}*")
            lines.append("")
        lines.append(self.body.strip())
        if self.data.get("alternatives"):
            lines += ["", "**Alternatives considered**", ""]
            lines += [f"- {alt}" for alt in self.data["alternatives"]]
        if self.data.get("consequences"):
            lines += ["", "**Consequences**", "", str(self.data["consequences"])]
        if self.status == MemoryStatus.SUPERSEDED:
            lines += ["", f"> Superseded by `{self.superseded_by}`."]
        return "\n".join(lines)
