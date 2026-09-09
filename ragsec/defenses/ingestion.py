"""Layer 1: ingestion-time source verification.

Controls *who* may write to the knowledge base and *what* they may write: source allowlists,
per-source trust scores, HMAC signatures over content, duplicate suppression and per-source
quotas that blunt bulk poisoning (e.g. context flooding).
"""

from __future__ import annotations

import hashlib
import hmac
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from ragsec.core.defense import DefenseLayer
from ragsec.core.models import Document


def sign_document(text: str, key: bytes) -> str:
    """HMAC-SHA256 signature (hex) over the document text."""
    return hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class IngestRejection:
    doc_id: str
    source_id: str
    reason: str


class SourceVerificationLayer(DefenseLayer):
    name = "source_verification"

    def __init__(
        self,
        allowed_sources: Iterable[str] | None = None,
        source_trust: dict[str, float] | None = None,
        min_trust: float = 0.0,
        hmac_key: bytes | None = None,
        require_signature: bool = False,
        dedupe: bool = True,
        max_documents_per_source: int | None = None,
    ) -> None:
        self.allowed_sources = set(allowed_sources) if allowed_sources is not None else None
        self.source_trust = dict(source_trust or {})
        self.min_trust = min_trust
        self.hmac_key = hmac_key
        self.require_signature = require_signature
        self.dedupe = dedupe
        self.max_documents_per_source = max_documents_per_source
        self.rejections: list[IngestRejection] = []
        self._seen_hashes: set[str] = set()
        self._per_source: Counter[str] = Counter()

    def on_ingest(self, documents: list[Document]) -> list[Document]:
        accepted: list[Document] = []
        for doc in documents:
            source = doc.provenance.source_id
            if source in self.source_trust:
                doc.provenance.trust_score = self.source_trust[source]
            reason = self._reject_reason(doc)
            if reason is not None:
                self.rejections.append(IngestRejection(doc.doc_id, source, reason))
                continue
            self._seen_hashes.add(doc.content_hash)
            self._per_source[source] += 1
            accepted.append(doc)
        return accepted

    def _reject_reason(self, doc: Document) -> str | None:
        source = doc.provenance.source_id
        if self.allowed_sources is not None and source not in self.allowed_sources:
            return "source not in allowlist"
        if doc.provenance.trust_score < self.min_trust:
            return "untrusted source"
        signature = doc.provenance.signature
        if signature is None:
            if self.require_signature:
                return "missing signature"
        elif self.hmac_key is not None and not hmac.compare_digest(
            signature, sign_document(doc.text, self.hmac_key)
        ):
            return "invalid signature"
        if self.dedupe and doc.content_hash in self._seen_hashes:
            return "duplicate content"
        if (
            self.max_documents_per_source is not None
            and self._per_source[source] >= self.max_documents_per_source
        ):
            return "source quota exceeded"
        return None
