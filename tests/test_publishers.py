"""Tests for multi-publisher signing and the compromised-publisher threat model.

Signed ingestion answers the question "did an authorised key vouch for this document?". It
cannot answer "is that key's holder honest?". These tests pin both halves: verification must
name *which* publisher vouched, and a publisher that turns adversarial must still get its
documents admitted, because that is the premise the independence mechanism is there to meet.
"""

from __future__ import annotations

import pytest

from ragsec.core.models import Document, Provenance
from ragsec.defenses.independence import VERIFIED_SIGNER_KEY
from ragsec.defenses.ingestion import SourceVerificationLayer, sign_document
from ragsec.eval.runner import (
    PUBLISHER_ZERO,
    ExperimentConfig,
    ExperimentRunner,
    sign_documents,
)


def _doc(doc_id: str, source_id: str = "corpus", text: str | None = None) -> Document:
    return Document(
        doc_id=doc_id,
        text=text if text is not None else f"text of {doc_id}",
        provenance=Provenance(source_id=source_id),
    )


KEYS = {"publisher-a": b"k" * 32, "publisher-b": b"j" * 32}


# ---------------------------------------------------------------------------
# Keyring verification
# ---------------------------------------------------------------------------


def test_verification_records_which_publisher_vouched() -> None:
    layer = SourceVerificationLayer(hmac_keys=KEYS, require_signature=True)
    doc = sign_documents([_doc("d1")], KEYS["publisher-b"])[0]

    accepted = layer.on_ingest([doc])

    assert len(accepted) == 1
    # Not merely "a key verified this": aggregation needs the publisher's identity, since that
    # identity is the unit of corroboration.
    assert accepted[0].provenance.extra[VERIFIED_SIGNER_KEY] == "publisher-b"


def test_each_publisher_is_named_separately() -> None:
    layer = SourceVerificationLayer(hmac_keys=KEYS, require_signature=True)
    docs = [
        sign_documents([_doc("d1")], KEYS["publisher-a"])[0],
        sign_documents([_doc("d2")], KEYS["publisher-b"])[0],
    ]

    signers = {
        doc.provenance.extra[VERIFIED_SIGNER_KEY] for doc in layer.on_ingest(docs)
    }

    assert signers == {"publisher-a", "publisher-b"}


def test_a_signature_from_an_unknown_key_is_rejected() -> None:
    layer = SourceVerificationLayer(hmac_keys=KEYS, require_signature=True)
    doc = sign_documents([_doc("d1")], b"not-an-authorised-key".ljust(32, b"0"))[0]

    assert layer.on_ingest([doc]) == []
    assert [r.reason for r in layer.rejections] == ["invalid signature"]


def test_a_single_key_still_configures_a_keyring() -> None:
    layer = SourceVerificationLayer(
        hmac_key=b"k" * 32, signer_id="solo", require_signature=True
    )
    doc = sign_documents([_doc("d1")], b"k" * 32)[0]

    accepted = layer.on_ingest([doc])

    assert accepted[0].provenance.extra[VERIFIED_SIGNER_KEY] == "solo"


def test_require_signature_without_any_key_is_refused() -> None:
    with pytest.raises(ValueError, match="hmac_key is required"):
        SourceVerificationLayer(require_signature=True)


def test_a_tampered_document_loses_its_signer() -> None:
    layer = SourceVerificationLayer(hmac_keys=KEYS)
    signed = sign_documents([_doc("d1")], KEYS["publisher-a"])[0]
    tampered = Document(
        doc_id=signed.doc_id,
        text="rewritten after signing",
        provenance=signed.provenance,
    )

    assert layer.on_ingest([tampered]) == []


def test_an_unsigned_document_is_not_attributed_to_a_publisher() -> None:
    layer = SourceVerificationLayer(hmac_keys=KEYS)

    accepted = layer.on_ingest([_doc("d1")])

    assert VERIFIED_SIGNER_KEY not in accepted[0].provenance.extra


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


def test_compromised_publisher_requires_signing() -> None:
    with pytest.raises(ValueError, match="compromised_publisher requires sign_corpus"):
        ExperimentConfig(sign_corpus=False, compromised_publisher=True)


@pytest.mark.parametrize("publishers", [0, -1, 1.5])
def test_publishers_must_be_a_positive_integer(publishers: object) -> None:
    with pytest.raises(ValueError, match="publishers must be a positive integer"):
        ExperimentConfig(publishers=publishers)  # type: ignore[arg-type]


def test_the_corpus_is_spread_across_publishers() -> None:
    config = ExperimentConfig(sign_corpus=True, publishers=4)
    runner = ExperimentRunner(config)

    corpus = runner._corpus()
    layer = SourceVerificationLayer(hmac_keys=runner._signing_keys, require_signature=True)
    signers = {
        doc.provenance.extra[VERIFIED_SIGNER_KEY] for doc in layer.on_ingest(corpus)
    }

    # A claim several corpus documents support should normally be attested by several keys,
    # which is what makes "independent" mean anything stronger than "separately uploaded".
    assert len(signers) > 1
    assert signers <= set(runner._signing_keys)


def test_publisher_keys_are_distinct_and_not_persisted() -> None:
    config = ExperimentConfig(sign_corpus=True, publishers=3)
    runner = ExperimentRunner(config)

    assert len(set(runner._signing_keys.values())) == 3
    # Keys are runtime-only; a serialised config must never carry key material.
    assert "signing_key" not in config.to_dict()
    assert not any(
        isinstance(v, bytes) for v in config.to_dict().values()
    )


def test_poison_from_a_compromised_publisher_passes_signed_ingestion() -> None:
    """The premise of the threat model: signatures cannot exclude an authorised adversary."""
    runner = ExperimentRunner(ExperimentConfig(sign_corpus=True, publishers=4))
    layer = SourceVerificationLayer(hmac_keys=runner._signing_keys, require_signature=True)
    poison = sign_documents(
        [_doc("evil-1", text="fabricated claim one")],
        runner._signing_keys[PUBLISHER_ZERO],
    )

    accepted = layer.on_ingest(poison)

    assert len(accepted) == 1
    assert accepted[0].provenance.extra[VERIFIED_SIGNER_KEY] == PUBLISHER_ZERO


def test_unsigned_poison_is_still_excluded_when_signatures_are_required() -> None:
    runner = ExperimentRunner(ExperimentConfig(sign_corpus=True, publishers=4))
    layer = SourceVerificationLayer(hmac_keys=runner._signing_keys, require_signature=True)

    assert layer.on_ingest([_doc("evil-1", text="fabricated claim one")]) == []
    assert [r.reason for r in layer.rejections] == ["missing signature"]


def test_a_forged_signer_claim_does_not_survive_ingestion() -> None:
    """An uploader may write anything into provenance; only verification may be believed."""
    layer = SourceVerificationLayer(hmac_keys=KEYS)
    doc = _doc("d1")
    doc.provenance.extra[VERIFIED_SIGNER_KEY] = "publisher-a"

    accepted = layer.on_ingest([doc])

    assert VERIFIED_SIGNER_KEY not in accepted[0].provenance.extra


def test_signing_payload_covers_identity_source_and_text() -> None:
    """Independence rests on the signature binding the fields the class is derived from."""
    key = KEYS["publisher-a"]
    base = _doc("d1", source_id="corpus")
    signature = sign_document(base, key)

    assert sign_document(_doc("d2", source_id="corpus"), key) != signature
    assert sign_document(_doc("d1", source_id="elsewhere"), key) != signature
    assert sign_document(_doc("d1", text="other text"), key) != signature
