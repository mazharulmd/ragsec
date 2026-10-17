"""Data models shared across the framework."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Provenance:
    """Where a document came from and how much we trust it.

    Attributes:
        source_id: Identifier of the originating source (e.g. ``"internal-wiki"``).
        author: Optional author or uploader identity.
        ingest_time: Unix timestamp of ingestion.
        trust_score: Prior trust in ``[0, 1]`` assigned to the source.
        signature: Optional hex HMAC over the versioned document ID, source ID, and text.
        extra: Free-form metadata.
    """

    source_id: str = "unknown"
    author: str | None = None
    ingest_time: float = field(default_factory=time.time)
    trust_score: float = 0.5
    signature: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A raw document prior to chunking."""

    text: str
    doc_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    provenance: Provenance = field(default_factory=Provenance)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.text)

    @property
    def is_poison(self) -> bool:
        """Ground-truth label used only for evaluation; never consulted by defenses."""
        return bool(self.metadata.get("poison", False))


@dataclass
class Chunk:
    """A retrievable unit of text derived from a :class:`Document`."""

    text: str
    doc_id: str
    chunk_id: str
    position: int
    provenance: Provenance
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_poison(self) -> bool:
        """Ground-truth label used only for evaluation; never consulted by defenses."""
        return bool(self.metadata.get("poison", False))


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, annotated by defense layers."""

    chunk: Chunk
    score: float
    rank: int
    index: int
    filter_scores: dict[str, float] = field(default_factory=dict)
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)

    def flag(self, reason: str) -> None:
        self.flagged = True
        if reason not in self.flag_reasons:
            self.flag_reasons.append(reason)


@dataclass
class IngestResult:
    """Summary of an ingestion call."""

    accepted_documents: int
    rejected_documents: int
    chunks_added: int


@dataclass
class RAGResponse:
    """Result of a pipeline query."""

    query: str
    answer: str
    retrieved: list[RetrievedChunk]
    used: list[RetrievedChunk]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def filtered(self) -> list[RetrievedChunk]:
        """Chunks that were retrieved but removed by a defense layer."""
        used_ids = {rc.chunk.chunk_id for rc in self.used}
        return [rc for rc in self.retrieved if rc.chunk.chunk_id not in used_ids]
