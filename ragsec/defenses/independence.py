"""Provenance principals: deciding when two passages come from the same party.

Evidence aggregation in retrieval-augmented generation treats agreement between retrieved
passages as corroboration, and corroboration as a reason for confidence. That inference is only
sound when the agreeing passages come from parties that an adversary cannot multiply. A
document identifier does not establish this: an adversary who can write to the knowledge base
chooses how many identifiers their claim arrives under, so counting identifiers lets them
manufacture agreement at will. Our measurements show the consequence, with answer aggregation
raising attack success rather than lowering it when an adversary supplies several differently
worded copies of one claim.

This module derives a *principal* for a passage, and aggregation caps how much weight any one
principal can contribute. Two passages with the same principal are treated as one voice that
spoke twice, not two voices that agree.

Binding is the whole point
--------------------------
A principal is only useful if the adversary cannot mint new ones. That requires the attribute
to be assigned by the server, not read from the upload:

``signer`` (bound, the default)
    The identity whose signature over the document Layer 1 verified, recorded by Layer 1
    itself and never read from caller-supplied metadata. An adversary without a key cannot
    produce a second one; an adversary with one key is confined to that one identity no matter
    what else it writes.

Everything else a document carries is chosen by whoever submitted it:

``source_id`` (unbound under a compromised publisher)
    Covered by the signing envelope, so a third party cannot rewrite it. The holder of a valid
    key can still choose any value it likes before signing, so it does not constrain the
    adversary this defense is aimed at.
``author`` (unbound)
    Not covered by the signing envelope at all. A key holder may vary it freely and still
    verify. An earlier version of this module included ``author`` in the default key, which let
    one compromised key mint unlimited principals; see ``test_independence.py``.
``doc_id``
    Reproduces the identifier counting this module exists to replace. Useful only as the
    control arm of an ablation.
``epoch``
    A coarse ingestion-time bucket. In a corpus loaded in one batch it collapses everything
    into a single principal.

The unbound attributes remain selectable so that an ablation can measure what they are worth,
but they must be requested deliberately: :func:`independence_key` refuses them unless the
caller passes ``allow_unbound=True``, because a deployment that mixes them into the principal
is counting metadata rather than parties.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragsec.core.models import Chunk

#: Attributes the server assigns from its own verification, which an adversary cannot choose.
BOUND_FIELDS: frozenset[str] = frozenset({"signer"})

#: Attributes the submitter controls. Selectable for ablations, never part of the default.
UNBOUND_FIELDS: frozenset[str] = frozenset({"source_id", "author", "doc_id", "epoch"})

#: Provenance attributes combined into a principal. Only bound attributes by default.
DEFAULT_FIELDS: tuple[str, ...] = ("signer",)

#: Width of an ingestion-time bucket, in seconds, when ``epoch`` is in use.
DEFAULT_EPOCH_SECONDS: float = 3600.0

#: Recorded by Layer 1 when it verifies a signature, so that aggregation can tell a passage
#: whose origin was proved from one whose origin was merely claimed.
VERIFIED_SIGNER_KEY = "verified_signer"

UNSIGNED = "unsigned"
#: A signature is present but was never checked, because no verifying layer was configured.
UNVERIFIED = "unverified-signature"


def signer_of(chunk: Chunk) -> str:
    """Identity that provably vouched for ``chunk``, or a marker for the absence of one.

    All unsigned passages share one value, and so form one principal. That is deliberate: a
    deployment that cannot prove who wrote a document has no basis for treating two of them as
    two parties.
    """
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
    *,
    allow_unbound: bool = False,
) -> tuple[str, ...]:
    """Return the principal of ``chunk`` as a tuple of provenance components.

    Passages sharing a key are treated as one party regardless of how many document
    identifiers, source labels or author names they carry.

    Raises:
        ValueError: if ``fields`` names an attribute the submitter controls and
            ``allow_unbound`` is not set, or names an unknown attribute.
    """
    unbound = [f for f in fields if f in UNBOUND_FIELDS]
    if unbound and not allow_unbound:
        raise ValueError(
            f"{unbound} are chosen by whoever uploads the document, so including them lets a "
            "single key holder mint principals at will. Pass allow_unbound=True only to "
            f"measure that effect; the defensible choice is {sorted(BOUND_FIELDS)}."
        )

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
            parts.append(chunk.doc_id)
        elif field == "epoch":
            parts.append(str(int(provenance.ingest_time // epoch_seconds)))
        else:
            raise ValueError(
                f"Unknown principal field {field!r}; choose from "
                "'signer', 'source_id', 'author', 'doc_id', 'epoch'"
            )
    return tuple(parts)
