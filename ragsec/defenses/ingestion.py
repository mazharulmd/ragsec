"""Layer 1: ingestion-time source verification.

Controls *who* may write to the knowledge base and *what* they may write: source allowlists,
per-source trust scores, HMAC signatures over document identity and content, duplicate
suppression and per-source quotas that blunt bulk poisoning (e.g. context flooding).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, replace

from ragsec.core.defense import DefenseLayer
from ragsec.core.models import Document, content_hash
from ragsec.defenses.independence import VERIFIED_SIGNER_KEY

_SIGNATURE_RE = re.compile(r"[0-9a-f]{64}\Z")


def document_signing_payload(doc: Document) -> bytes:
    """Return a canonical, versioned representation of the signed document fields."""
    envelope = {
        "version": "ragsec-document-v1",
        "doc_id": doc.doc_id,
        "source_id": doc.provenance.source_id,
        "text": doc.text,
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sign_document(doc: Document, key: bytes) -> str:
    """Return an HMAC-SHA256 hex digest over document identity, source, and text."""
    return hmac.new(key, document_signing_payload(doc), hashlib.sha256).hexdigest()


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
        default_trust: float = 0.5,
        min_trust: float = 0.0,
        hmac_key: bytes | None = None,
        require_signature: bool = False,
        dedupe: bool = True,
        max_documents_per_source: int | None = None,
        signer_id: str = "ingestion-key",
        hmac_keys: dict[str, bytes] | None = None,
    ) -> None:
        self.allowed_sources = set(allowed_sources) if allowed_sources is not None else None
        self.source_trust = dict(source_trust or {})
        self.default_trust = default_trust
        self.min_trust = min_trust
        if not 0.0 <= default_trust <= 1.0:
            raise ValueError("default_trust must be in [0, 1]")
        if not 0.0 <= min_trust <= 1.0:
            raise ValueError("min_trust must be in [0, 1]")
        if any(not 0.0 <= score <= 1.0 for score in self.source_trust.values()):
            raise ValueError("source_trust scores must be in [0, 1]")
        self.hmac_key = hmac_key
        self.require_signature = require_signature
        #: Accepted publisher keys, as signer name to key. A real deployment authorises several
        #: independent publishers, so verification has to say *which* key vouched for a
        #: document, not merely that some key did: that identity is what Layer 3 later treats
        #: as a unit of corroboration.
        self.keyring: dict[str, bytes] = dict(hmac_keys or {})
        if hmac_key is not None:
            self.keyring.setdefault(signer_id, hmac_key)
        if require_signature and not self.keyring:
            raise ValueError("hmac_key is required when require_signature is enabled")
        self.dedupe = dedupe
        self.max_documents_per_source = max_documents_per_source
        #: Name recorded against documents whose signature this layer actually verified.
        #: Aggregation uses it to tell evidence whose origin was proved from evidence whose
        #: origin was merely asserted, which an adversary can forge for free.
        self.signer_id = signer_id
        self.rejections: list[IngestRejection] = []
        self._seen_hashes: set[str] = set()
        self._per_source: Counter[str] = Counter()

    def on_ingest(self, documents: list[Document]) -> list[Document]:
        accepted: list[Document] = []
        for doc in documents:
            source = doc.provenance.source_id
            # Trust is policy metadata. Never accept the caller's score as authority, and do
            # not mutate their object while assigning the trusted score.
            trusted_doc = replace(
                doc,
                provenance=deepcopy(doc.provenance),
                metadata=dict(doc.metadata),
            )
            trusted_doc.provenance.trust_score = self.source_trust.get(
                source, self.default_trust
            )
            reason = self._reject_reason(trusted_doc)
            if reason is not None:
                self.rejections.append(IngestRejection(doc.doc_id, source, reason))
                continue
            verified_by = self._verifying_signer(trusted_doc)
            if verified_by is not None:
                trusted_doc.provenance.extra[VERIFIED_SIGNER_KEY] = verified_by
            else:
                # Never let a caller assert that their document was verified.
                trusted_doc.provenance.extra.pop(VERIFIED_SIGNER_KEY, None)
            self._seen_hashes.add(content_hash(trusted_doc.text))
            self._per_source[source] += 1
            accepted.append(trusted_doc)
        return accepted

    def _verifying_signer(self, doc: Document) -> str | None:
        """Name the configured key whose signature on ``doc`` holds, or ``None``.

        An unchecked signature is not evidence of anything, so a document carrying a
        signature this layer holds no key for verifies as nothing.
        """
        signature = doc.provenance.signature
        if not signature or not isinstance(signature, str):
            return None
        for name, key in self.keyring.items():
            if hmac.compare_digest(signature, sign_document(doc, key)):
                return name
        return None

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
        elif self.keyring:
            if not isinstance(signature, str) or not _SIGNATURE_RE.fullmatch(signature):
                return "invalid signature"
            if self._verifying_signer(doc) is None:
                return "invalid signature"
        if self.dedupe and content_hash(doc.text) in self._seen_hashes:
            return "duplicate content"
        if (
            self.max_documents_per_source is not None
            and self._per_source[source] >= self.max_documents_per_source
        ):
            return "source quota exceeded"
        return None
