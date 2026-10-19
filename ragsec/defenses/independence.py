"""Provenance independence: deciding when two passages are really two pieces of evidence.

Evidence aggregation in retrieval-augmented generation treats agreement between retrieved
passages as corroboration, and corroboration as a reason for confidence. That inference is only
sound when the agreeing passages are independent. A document identifier does not establish
independence: an adversary who can write to the knowledge base chooses how many identifiers
their claim arrives under, so counting identifiers lets them manufacture agreement at will. Our
measurements show the consequence, with answer aggregation raising attack success rather than
lowering it when an adversary supplies several differently worded copies of one claim.

This module derives an *independence class* for a passage from the provenance attributes an
adversary cannot multiply for free. Two passages in the same class are treated as one voice
that spoke twice, not two voices that agree.

The signals, in decreasing order of how hard they are to forge:

``signer``
    The identity whose signature over the document was verified at ingestion. An adversary
    without the key cannot produce a second one, so this is the only component that is
    cryptographically bound rather than asserted.
``source_id``
    The connector or publisher that submitted the document. Forgeable by a defense-aware
    adversary, as our provenance-spoofing results show, but still separates an adversary who
    has compromised one connector from the rest of the corpus.
``author``
    The uploader identity within a source, where a deployment records one.
``epoch``
    A coarse ingestion-time bucket. Documents arriving from one source inside one window are
    plausibly one upload. Off by default: in a corpus loaded in a single batch it collapses
    everything into one class and discards real independence.

The weaker components are not security controls on their own. They are included because
independence discounting has to degrade gracefully: a deployment with no signing still gets
the volume advantage removed, and a deployment with signing gets a class structure that an
adversary cannot forge.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragsec.core.models import Chunk

#: Provenance attributes combined into an independence class, strongest first.
DEFAULT_FIELDS: tuple[str, ...] = ("signer", "source_id", "author")

#: Width of an ingestion-time bucket, in seconds, when ``epoch`` is in use.
DEFAULT_EPOCH_SECONDS: float = 3600.0

#: Recorded by Layer 1 when it verifies a signature, so that aggregation can tell a passage
#: whose origin was proved from one whose origin was merely claimed.
VERIFIED_SIGNER_KEY = "verified_signer"

UNSIGNED = "unsigned"
#: A signature is present but was never checked, because no verifying layer was configured.
UNVERIFIED = "unverified-signature"


def signer_of(chunk: Chunk) -> str:
    """Identity that provably vouched for ``chunk``, or a marker for the absence of one."""
    provenance = chunk.provenance
    verified = provenance.extra.get(VERIFIED_SIGNER_KEY)
    if verified:
        return f"signer:{verified}"
    if provenance.signature:
        return UNVERIFIED
    return UNSIGNED


def independence_key(
    chunk: Chunk,
    fields: Sequence[str] = DEFAULT_FIELDS,
    epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
) -> tuple[str, ...]:
    """Return the independence class of ``chunk`` as a tuple of provenance components.

    Passages sharing a key are treated as one source of evidence regardless of how many
    document identifiers they carry.
    """
    provenance = chunk.provenance
    parts: list[str] = []
    for field in fields:
        if field == "signer":
            parts.append(signer_of(chunk))
        elif field == "source_id":
            parts.append(provenance.source_id)
        elif field == "author":
            parts.append(provenance.author or "")
        elif field == "doc_id":
            # Reproduces the identifier counting this module exists to replace; useful as the
            # control arm of an ablation.
            parts.append(chunk.doc_id)
        elif field == "epoch":
            parts.append(str(int(provenance.ingest_time // epoch_seconds)))
        else:
            raise ValueError(
                f"Unknown independence field {field!r}; choose from "
                "'signer', 'source_id', 'author', 'doc_id', 'epoch'"
            )
    return tuple(parts)
