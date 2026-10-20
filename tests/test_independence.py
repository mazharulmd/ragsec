"""Tests for provenance independence discounting in evidence aggregation.

The property under test is the one the mechanism exists for: an adversary who writes several
differently worded copies of a claim must not gain weight in proportion to how many documents
they uploaded. Counting document identifiers gives them exactly that; counting independence
classes does not.
"""

from __future__ import annotations

import pytest

from ragsec.core.models import Chunk, Provenance, RetrievedChunk
from ragsec.defenses.aggregation import AnswerGroup, IsolatedAnswer, RobustAggregationLayer
from ragsec.defenses.independence import (
    UNSIGNED,
    UNVERIFIED,
    VERIFIED_SIGNER_KEY,
    independence_key,
    signer_of,
)


def _chunk(
    doc_id: str,
    source_id: str = "corpus",
    author: str | None = None,
    signature: str | None = None,
    verified_signer: str | None = None,
    ingest_time: float = 1000.0,
) -> Chunk:
    extra = {VERIFIED_SIGNER_KEY: verified_signer} if verified_signer else {}
    return Chunk(
        text=f"text of {doc_id}",
        doc_id=doc_id,
        chunk_id=f"{doc_id}:0",
        position=0,
        provenance=Provenance(
            source_id=source_id,
            author=author,
            signature=signature,
            ingest_time=ingest_time,
            extra=extra,
        ),
    )


def _answer(chunk: Chunk, weight: float) -> IsolatedAnswer:
    return IsolatedAnswer(
        text="an answer",
        chunk=RetrievedChunk(chunk=chunk, score=0.5, rank=0, index=0),
        tokens={"an", "answer"},
        novel_tokens={"answer"},
        relevance=0.5,
        weight=weight,
    )


# ---------------------------------------------------------------------------
# Independence keys
# ---------------------------------------------------------------------------


def test_distinct_documents_from_one_source_share_a_class() -> None:
    """The adversary's lever: many identifiers, one actor."""
    a = independence_key(_chunk("doc-1", source_id="wiki"))
    b = independence_key(_chunk("doc-2", source_id="wiki"))
    assert a == b


def test_a_differing_source_label_alone_does_not_make_a_second_principal() -> None:
    """A source label is chosen by whoever uploads, so it cannot stand for a second party."""
    a = independence_key(_chunk("doc-1", source_id="wiki"))
    b = independence_key(_chunk("doc-2", source_id="archive"))
    assert a == b


def test_a_verified_signature_separates_proved_from_claimed_origin() -> None:
    """Two passages claiming one source are only independent if the signer differs."""
    claimed = independence_key(_chunk("doc-1", source_id="wiki"))
    proved = independence_key(_chunk("doc-2", source_id="wiki", verified_signer="key-a"))
    other = independence_key(_chunk("doc-3", source_id="wiki", verified_signer="key-b"))
    assert claimed != proved
    assert proved != other


def test_signer_states_are_distinguished() -> None:
    assert signer_of(_chunk("d")) == UNSIGNED
    assert signer_of(_chunk("d", signature="abc")) == UNVERIFIED
    assert signer_of(_chunk("d", signature="abc", verified_signer="k")) == "signer:k"


def test_doc_id_field_reproduces_identifier_counting() -> None:
    """The control arm of the ablation must behave like the mechanism it replaces."""
    a = independence_key(_chunk("doc-1"), fields=("doc_id",), allow_unbound=True)
    b = independence_key(_chunk("doc-2"), fields=("doc_id",), allow_unbound=True)
    assert a != b


def test_epoch_buckets_group_one_upload_window() -> None:
    same = independence_key(
        _chunk("d1", ingest_time=10.0), fields=("epoch",), epoch_seconds=100.0,
        allow_unbound=True,
    )
    near = independence_key(
        _chunk("d2", ingest_time=90.0), fields=("epoch",), epoch_seconds=100.0,
        allow_unbound=True,
    )
    later = independence_key(
        _chunk("d3", ingest_time=250.0), fields=("epoch",), epoch_seconds=100.0,
        allow_unbound=True,
    )
    assert same == near
    assert same != later


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown principal field"):
        independence_key(_chunk("d"), fields=("nonsense",))


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------


def test_identifier_counting_rewards_volume() -> None:
    """Baseline behaviour, and the reason the mechanism is needed."""
    one = AnswerGroup([_answer(_chunk("d1", source_id="attacker"), 1.0)])
    three = AnswerGroup([
        _answer(_chunk(f"d{i}", source_id="attacker"), 1.0) for i in range(3)
    ])
    assert three.total_weight(0.25) > 3 * one.total_weight(0.25) * 0.9


def test_discounting_removes_the_volume_advantage() -> None:
    """Three documents from one actor must not outweigh one document from that actor."""
    one = AnswerGroup([_answer(_chunk("d1", source_id="attacker"), 1.0)])
    three = AnswerGroup([
        _answer(_chunk(f"d{i}", source_id="attacker"), 1.0) for i in range(3)
    ])
    assert three.total_weight(0.25, independence_discounting=True) == pytest.approx(
        one.total_weight(0.25, independence_discounting=True)
    )


def test_genuinely_independent_signers_still_earn_the_bonus() -> None:
    """Discounting must not punish real corroboration, or it is just a weaker aggregator."""
    correlated = AnswerGroup([
        _answer(_chunk(f"d{i}", verified_signer="key-a"), 1.0) for i in range(3)
    ])
    independent = AnswerGroup([
        _answer(_chunk(f"d{i}", verified_signer=f"key-{i}"), 1.0) for i in range(3)
    ])
    assert independent.total_weight(0.25, independence_discounting=True) > \
        correlated.total_weight(0.25, independence_discounting=True)


def test_within_class_discount_gives_partial_credit() -> None:
    group = AnswerGroup([
        _answer(_chunk(f"d{i}", source_id="one-source"), 1.0) for i in range(3)
    ])
    strict = group.total_weight(0.25, independence_discounting=True, within_class_discount=0.0)
    partial = group.total_weight(0.25, independence_discounting=True, within_class_discount=0.5)
    full = group.total_weight(0.25, independence_discounting=True, within_class_discount=1.0)
    assert strict < partial < full


def test_strongest_member_of_a_class_is_the_one_that_counts() -> None:
    group = AnswerGroup([
        _answer(_chunk("d1", source_id="s"), 0.2),
        _answer(_chunk("d2", source_id="s"), 0.9),
    ])
    assert group.total_weight(0.25, independence_discounting=True) == pytest.approx(0.9)


def test_discounting_off_reproduces_the_original_weighting_exactly() -> None:
    group = AnswerGroup([
        _answer(_chunk("d1", source_id="a"), 0.4),
        _answer(_chunk("d2", source_id="b"), 0.6),
    ])
    expected = (0.4 + 0.6) * (1.0 + 0.25 * 1)
    assert group.total_weight(0.25) == pytest.approx(expected)


def test_classes_partition_every_member() -> None:
    members = [
        _answer(_chunk("d1", verified_signer="key-a"), 1.0),
        _answer(_chunk("d2", verified_signer="key-a"), 1.0),
        _answer(_chunk("d3", verified_signer="key-b"), 1.0),
    ]
    grouped = AnswerGroup(members).classes()
    assert len(grouped) == 2
    assert sum(len(v) for v in grouped.values()) == len(members)


# ---------------------------------------------------------------------------
# Layer wiring
# ---------------------------------------------------------------------------


def test_layer_defaults_to_the_unmodified_behaviour() -> None:
    layer = RobustAggregationLayer()
    assert layer.independence_discounting is False


def test_layer_rejects_an_out_of_range_discount() -> None:
    with pytest.raises(ValueError, match="within_class_discount"):
        RobustAggregationLayer(independence_discounting=True, within_class_discount=1.5)


def test_layer_passes_its_settings_through_to_weighting() -> None:
    """A fabricated majority must not beat a single genuine passage once discounting is on."""
    layer = RobustAggregationLayer(independence_discounting=True)
    adversary = AnswerGroup([
        _answer(_chunk(f"bad{i}", source_id="spoofed"), 0.5) for i in range(3)
    ])
    genuine = AnswerGroup([_answer(_chunk("good", source_id="spoofed"), 0.6)])
    weigh = lambda g: g.total_weight(  # noqa: E731 - local alias keeps the assertion readable
        layer.redundancy_bonus,
        layer.independence_discounting,
        layer.within_class_discount,
        layer.independence_fields,
        layer.epoch_seconds,
    )
    assert weigh(genuine) > weigh(adversary)


# ---------------------------------------------------------------------------
# What the principal must resist
# ---------------------------------------------------------------------------


def test_a_key_holder_cannot_mint_principals_from_the_author_field() -> None:
    """The flaw this module was revised to close.

    The signing envelope covers version, document id, source and text. It does not cover the
    author, so a holder of one valid key could vary that field and still verify. While the
    principal included the author, five such documents counted as five agreeing parties, and
    the cap the rule exists to apply did not apply at all.
    """
    keys = {
        independence_key(_chunk(f"evil-{i}", verified_signer="key-a", author=f"made-up-{i}"))
        for i in range(5)
    }

    assert keys == {("signer:key-a",)}


def test_a_key_holder_cannot_mint_principals_from_the_source_label() -> None:
    """A compromised signer chooses the source label before signing, so it binds nothing."""
    keys = {
        independence_key(_chunk(f"evil-{i}", verified_signer="key-a", source_id=f"src-{i}"))
        for i in range(5)
    }

    assert keys == {("signer:key-a",)}


def test_a_key_holder_cannot_mint_principals_from_document_identifiers() -> None:
    keys = {
        independence_key(_chunk(f"evil-{i}", verified_signer="key-a")) for i in range(20)
    }

    assert len(keys) == 1


@pytest.mark.parametrize("field", ["source_id", "author", "doc_id", "epoch"])
def test_attributes_the_uploader_chooses_are_refused_by_default(field: str) -> None:
    with pytest.raises(ValueError, match="chosen by whoever uploads"):
        independence_key(_chunk("d"), fields=(field,))


def test_an_ablation_may_still_request_them_deliberately() -> None:
    key = independence_key(_chunk("d", author="someone"), fields=("author",),
                           allow_unbound=True)

    assert key == ("someone",)


def test_unsigned_documents_form_a_single_principal() -> None:
    """Without proof of origin there is no basis for calling two uploads two parties."""
    keys = {independence_key(_chunk(f"d{i}", source_id=f"src-{i}")) for i in range(5)}

    assert keys == {(UNSIGNED,)}


def test_the_layer_refuses_an_unbound_principal() -> None:
    with pytest.raises(ValueError, match="chosen by whoever uploads"):
        RobustAggregationLayer(
            independence_discounting=True, independence_fields=("signer", "author")
        )


def test_the_layer_accepts_one_for_an_ablation() -> None:
    layer = RobustAggregationLayer(
        independence_discounting=True,
        independence_fields=("signer", "author"),
        allow_unbound_fields=True,
    )

    assert layer.allow_unbound_fields is True
