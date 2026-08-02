"""Lexical retrieval (BM25) over project memory and code.

Forge retrieves from two corpora: structured memory records and source files.
Both are small (thousands of documents, not millions) and both are dominated by
*identifiers* -- symbol names, file paths, error strings. For that regime a
well-tuned lexical index beats a small embedding model on precision, needs no
GPU, no model download, and no index rebuild latency. Embeddings can be layered
on later through the same :class:`Index` interface; see
``docs/design-decisions.md#retrieval``.

The tokenizer is the part that matters: it splits ``camelCase``,
``snake_case``, ``kebab-case`` and dotted paths into both the whole token and
its parts, so a query for ``renderPlayer`` matches ``render_player`` and
``PlayerRenderer``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "if", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "then", "there", "these", "this", "to", "was", "were", "will", "with", "we", "you", "your"]
)


def tokenize(text: str) -> list[str]:
    """Split text into search tokens, expanding compound identifiers."""
    tokens: list[str] = []
    for raw in _WORD.findall(text):
        low = raw.lower()
        if low not in _STOPWORDS and len(low) > 1:
            tokens.append(low)
        if "_" in raw:
            tokens.extend(p.lower() for p in raw.split("_") if len(p) > 1)
        else:
            parts = _CAMEL.findall(raw)
            if len(parts) > 1:
                tokens.extend(p.lower() for p in parts if len(p) > 1)
    return tokens


@dataclass(slots=True)
class Document:
    """One retrievable unit. ``weight`` biases documents Forge trusts more."""

    id: str
    text: str
    weight: float = 1.0
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Hit:
    doc: Document
    score: float


class Index:
    """An in-memory BM25 index, rebuilt cheaply from the memory store.

    Rebuilding from scratch is O(corpus) and takes single-digit milliseconds at
    Forge's corpus sizes, so there is no incremental-update machinery to get
    wrong. The index is a pure function of the store's contents, which means it
    can never drift out of sync with the ledger.
    """

    K1 = 1.4
    B = 0.72

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._len: dict[str, int] = {}
        self._avg_len: float = 0.0

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, doc: Document) -> None:
        if doc.id in self._docs:
            self.remove(doc.id)
        tokens = tokenize(doc.text)
        tf = Counter(tokens)
        self._docs[doc.id] = doc
        self._tf[doc.id] = tf
        self._len[doc.id] = len(tokens)
        for term in tf:
            self._df[term] += 1
        self._recompute_avg()

    def add_all(self, docs: list[Document]) -> None:
        for doc in docs:
            self.add(doc)

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            return
        for term in self._tf[doc_id]:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        del self._docs[doc_id], self._tf[doc_id], self._len[doc_id]
        self._recompute_avg()

    def _recompute_avg(self) -> None:
        self._avg_len = (sum(self._len.values()) / len(self._len)) if self._len else 0.0

    def search(self, query: str, limit: int = 10, *, min_score: float = 0.0) -> list[Hit]:
        """Return the top ``limit`` documents scored by BM25 times doc weight."""
        terms = tokenize(query)
        if not terms or not self._docs:
            return []
        n = len(self._docs)
        scores: dict[str, float] = {}
        for term in set(terms):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, tf in self._tf.items():
                freq = tf.get(term)
                if not freq:
                    continue
                norm = 1 - self.B + self.B * (self._len[doc_id] / self._avg_len or 1.0)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (freq * (self.K1 + 1)) / (freq + self.K1 * norm)
        hits = [
            Hit(self._docs[doc_id], score * self._docs[doc_id].weight)
            for doc_id, score in scores.items()
        ]
        hits = [h for h in hits if h.score > min_score]
        hits.sort(key=lambda h: (-h.score, h.doc.id))
        return hits[:limit]
