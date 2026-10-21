"""Each defense layer must mitigate its target attacks without breaking clean answers."""

from __future__ import annotations

from collections import Counter

import pytest

from ragsec.attacks import (
    Attack,
    AttackTarget,
    ContextFloodingAttack,
    PoisonedRAGAttack,
    PromptInjectionAttack,
    TriggerBackdoorAttack,
)
from ragsec.core import (
    Chunk,
    DefenseLayer,
    Document,
    HashingEmbedder,
    MockLLM,
    Provenance,
    RAGPipeline,
    RetrievedChunk,
)
from ragsec.defenses import (
    ABSTAIN_ANSWER,
    FilterContext,
    InstructionPatternFilter,
    MonitoringLayer,
    NaturalnessFilter,
    QueryEchoFilter,
    RetrievalFilterLayer,
    RobustAggregationLayer,
    SimilarityOutlierFilter,
    SourceVerificationLayer,
    js_divergence,
    sign_document,
)
from ragsec.defenses.independence import VERIFIED_SIGNER_KEY

CLEAN = [
    "Paris is the capital and largest city of France.",
    "Tokyo is the capital of Japan and one of the most populous cities in the world.",
    "The mitochondrion is the powerhouse of the cell, producing ATP through respiration.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "Mount Everest is the highest mountain above sea level, located in the Himalayas.",
    "The Amazon is the largest river by discharge volume of water in the world.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Great Wall of China was built over centuries to protect against invasions.",
    "France's capital city is Paris, located on the Seine river.",
]
QUERY = "What is the capital of France?"
TARGET = AttackTarget(query=QUERY, target_answer="Berlin")
SUPPRESS = AttackTarget(query=QUERY, target_answer="Paris")


def clean_docs(source: str = "internal-wiki", trust: float = 0.5) -> list[Document]:
    return [
        Document(
            text=t,
            provenance=Provenance(source_id=source, trust_score=trust),
            metadata={"poison": False},
        )
        for t in CLEAN
    ]


def build(defenses: list[DefenseLayer]) -> RAGPipeline:
    pipeline = RAGPipeline.from_defaults(
        embedder=HashingEmbedder(dim=512), llm=MockLLM(), top_k=3, defenses=defenses
    )
    pipeline.ingest(clean_docs())
    return pipeline


def make_rc(text: str, score: float = 0.0) -> RetrievedChunk:
    chunk = Chunk(text=text, doc_id="d", chunk_id="d:0", position=0, provenance=Provenance())
    return RetrievedChunk(chunk=chunk, score=score, rank=0, index=0)


ATTACK_CASES: list[tuple[Attack, AttackTarget, int]] = [
    (PoisonedRAGAttack(), TARGET, 1),
    (PromptInjectionAttack(), TARGET, 4),
    (TriggerBackdoorAttack(), TARGET, 1),
    (ContextFloodingAttack(), SUPPRESS, 5),
]
ATTACK_IDS = [case[0].name for case in ATTACK_CASES]


# Layer 1


def test_source_verification_trust_registry_and_dedupe() -> None:
    layer = SourceVerificationLayer(
        source_trust={"internal-wiki": 0.9, "community-upload": 0.2}, min_trust=0.5
    )
    docs = clean_docs() + PoisonedRAGAttack().generate_poison([TARGET], 2)
    docs.append(Document(text=CLEAN[0], provenance=Provenance(source_id="internal-wiki")))
    accepted = layer.on_ingest(docs)
    assert len(accepted) == len(CLEAN)
    assert all(d.provenance.source_id == "internal-wiki" for d in accepted)
    assert all(d.provenance.trust_score == 0.9 for d in accepted)
    reasons = Counter(r.reason for r in layer.rejections)
    assert reasons == {"untrusted source": 2, "duplicate content": 1}


def test_ingestion_trust_comes_from_policy_and_isolated_from_input_mutation() -> None:
    document = Document(
        text="trusted evidence",
        provenance=Provenance(source_id="internal-wiki", trust_score=1.0),
    )
    layer = SourceVerificationLayer(source_trust={"internal-wiki": 0.8})
    accepted = layer.on_ingest([document])

    assert document.provenance.trust_score == 1.0
    assert accepted[0] is not document
    assert accepted[0].provenance.trust_score == 0.8

    document.provenance.trust_score = 0.0
    assert accepted[0].provenance.trust_score == 0.8


def test_unmapped_source_uses_server_default_trust() -> None:
    document = Document(
        text="unmapped evidence",
        provenance=Provenance(source_id="upload", trust_score=1.0),
    )
    accepted = SourceVerificationLayer(default_trust=0.2).on_ingest([document])
    assert accepted[0].provenance.trust_score == 0.2


def test_source_verification_recomputes_hash_for_dedupe() -> None:
    layer = SourceVerificationLayer()
    original = Document(text="same content")
    forged_hash = Document(text="same content", content_hash="not-the-content-hash")

    assert layer.on_ingest([original, forged_hash]) == [original]
    assert layer.rejections[0].reason == "duplicate content"


def test_source_verification_allowlist_and_quota() -> None:
    layer = SourceVerificationLayer(
        allowed_sources={"community-upload"}, max_documents_per_source=2
    )
    docs = ContextFloodingAttack().generate_poison([SUPPRESS], 5) + clean_docs()[:1]
    accepted = layer.on_ingest(docs)
    assert len(accepted) == 2
    reasons = Counter(r.reason for r in layer.rejections)
    assert reasons["source quota exceeded"] == 3
    assert reasons["source not in allowlist"] == 1


def test_source_verification_signatures() -> None:
    key = b"secret-key"
    layer = SourceVerificationLayer(hmac_key=key, require_signature=True)
    good = Document(text="signed")
    good.provenance.signature = sign_document(good, key)
    bad = Document(text="other")
    bad.provenance.signature = sign_document(bad, key)
    bad.text = "tampered"
    missing = Document(text="unsigned")
    accepted = layer.on_ingest([good, bad, missing])
    assert [d.doc_id for d in accepted] == [good.doc_id]
    assert accepted[0].text == good.text
    assert {r.reason for r in layer.rejections} == {"invalid signature", "missing signature"}


def test_verified_signature_is_recorded_on_the_accepted_document() -> None:
    """Aggregation needs to know whose key vouched for a document, not merely that one did."""
    key = b"secret-key"
    layer = SourceVerificationLayer(hmac_key=key, signer_id="corpus-key")
    signed = Document(text="signed")
    signed.provenance.signature = sign_document(signed, key)
    unsigned = Document(text="plain")

    accepted = {d.text: d for d in layer.on_ingest([signed, unsigned])}
    assert accepted["signed"].provenance.extra[VERIFIED_SIGNER_KEY] == "corpus-key"
    assert VERIFIED_SIGNER_KEY not in accepted["plain"].provenance.extra


def test_a_claimed_verification_marker_is_not_trusted() -> None:
    """An uploader must not be able to assert that their own document was verified."""
    layer = SourceVerificationLayer()
    forged = Document(text="claims to be verified")
    forged.provenance.extra[VERIFIED_SIGNER_KEY] = "corpus-key"
    accepted = layer.on_ingest([forged])
    assert VERIFIED_SIGNER_KEY not in accepted[0].provenance.extra


def test_source_verification_rejects_malformed_signature_without_raising() -> None:
    layer = SourceVerificationLayer(hmac_key=b"secret-key", require_signature=True)
    malformed = Document(text="signed")
    malformed.provenance.signature = "é" * 64
    assert layer.on_ingest([malformed]) == []
    assert layer.rejections[0].reason == "invalid signature"


def test_source_verification_requires_key_for_required_signatures() -> None:
    with pytest.raises(ValueError, match="hmac_key"):
        SourceVerificationLayer(require_signature=True)


# Layer 2 filters


def test_query_echo_filter() -> None:
    ctx = FilterContext.build(QUERY)
    flt = QueryEchoFilter()
    assert flt.evaluate(make_rc(PoisonedRAGAttack().craft(TARGET, 0)), ctx).flagged
    assert flt.evaluate(make_rc(ContextFloodingAttack().craft(SUPPRESS, 0)), ctx).flagged
    assert not any(flt.evaluate(make_rc(t), ctx).flagged for t in CLEAN)
    # Declarative overlap without the interrogative is not an echo.
    japan_ctx = FilterContext.build("What is the capital of Japan?")
    result = flt.evaluate(make_rc(CLEAN[1]), japan_ctx)
    assert result.score >= 0.75
    assert not result.flagged


def test_instruction_filter_including_obfuscation() -> None:
    ctx = FilterContext.build(QUERY)
    flt = InstructionPatternFilter()
    for variant in range(4):
        assert flt.evaluate(make_rc(PromptInjectionAttack().craft(TARGET, variant)), ctx).flagged
    obfuscated = PromptInjectionAttack(obfuscate=True).craft(TARGET, 0)
    result = flt.evaluate(make_rc(obfuscated), ctx)
    assert result.flagged
    assert "invisible" in result.reason
    assert not any(flt.evaluate(make_rc(t), ctx).flagged for t in CLEAN)


def test_naturalness_filter_flags_trigger_tokens() -> None:
    ctx = FilterContext.build(QUERY)
    flt = NaturalnessFilter()
    result = flt.evaluate(make_rc(TriggerBackdoorAttack().craft(TARGET, 0)), ctx)
    assert result.flagged
    assert "gibberish" in result.reason
    assert not any(flt.evaluate(make_rc(t), ctx).flagged for t in CLEAN)


def test_similarity_outlier_filter_flags_query_duplicates() -> None:
    ctx = FilterContext.build(QUERY)
    flt = SimilarityOutlierFilter(max_query_similarity=0.85)
    assert flt.evaluate(make_rc("x", score=0.95), ctx).flagged
    assert not flt.evaluate(make_rc("x", score=0.5), ctx).flagged


@pytest.mark.parametrize(("attack", "target", "n"), ATTACK_CASES, ids=ATTACK_IDS)
def test_retrieval_filter_layer_mitigates(attack: Attack, target: AttackTarget, n: int) -> None:
    pipeline = build([RetrievalFilterLayer()])
    pipeline.ingest(attack.generate_poison([target], n))
    response = pipeline.query(attack.poison_query(target))
    assert any(rc.chunk.is_poison for rc in response.retrieved), "poison must be a candidate"
    assert not any(rc.chunk.is_poison for rc in response.used)
    assert all(rc.flagged for rc in response.filtered if rc.chunk.is_poison)
    assert "Paris" in response.answer
    assert not attack.is_success(target, response.answer)


def test_retrieval_filter_layer_annotate_mode_keeps_chunks() -> None:
    pipeline = build([RetrievalFilterLayer(mode="annotate")])
    pipeline.ingest(PoisonedRAGAttack().generate_poison([TARGET], 1))
    response = pipeline.query(QUERY)
    assert response.used[0].chunk.is_poison
    assert response.used[0].flagged


# Layer 3


def test_robust_aggregation_preserves_clean_answer() -> None:
    pipeline = build([RobustAggregationLayer()])
    response = pipeline.query(QUERY)
    assert "Paris" in response.answer
    assert response.metadata["generator"] == "robust_aggregation"


def test_robust_aggregation_scores_retrieved_passage_not_answer_word_overlap() -> None:
    from ragsec.core.llm import BaseLLM

    class ConciseAnswerLLM(BaseLLM):
        name = "concise-test"

        def generate(self, prompt: str, *, system: str | None = None) -> str:
            return "1953"

    pipeline = RAGPipeline.from_defaults(
        embedder=HashingEmbedder(dim=512),
        llm=ConciseAnswerLLM(),
        top_k=1,
        defenses=[RobustAggregationLayer()],
    )
    pipeline.ingest(
        [
            Document(
                text="The city was founded in 1953.",
                provenance=Provenance(source_id="encyclopedia", trust_score=0.5),
            )
        ]
    )

    response = pipeline.query("In what year was the city founded?")

    assert response.answer == "1953"


def test_robust_aggregation_blocks_minority_poison() -> None:
    layer = RobustAggregationLayer()
    pipeline = build([layer])
    attack = PoisonedRAGAttack()
    pipeline.ingest(attack.generate_poison([TARGET], 1))
    response = pipeline.query(QUERY)
    assert response.used[0].chunk.is_poison, "undefended retrieval still ranks poison first"
    assert not attack.is_success(TARGET, response.answer)
    assert response.answer == ABSTAIN_ANSWER


def test_robust_aggregation_abstains_on_conflicting_injection_answer() -> None:
    pipeline = build([RobustAggregationLayer()])
    attack = PromptInjectionAttack()
    pipeline.ingest(attack.generate_poison([TARGET], 1))
    response = pipeline.query(QUERY)
    assert not attack.is_success(TARGET, response.answer)
    assert response.answer == ABSTAIN_ANSWER


# Layer 4


def test_js_divergence_bounds() -> None:
    assert js_divergence(Counter(a=1), Counter(a=1)) == 0.0
    assert js_divergence(Counter(a=1), Counter(b=1)) == pytest.approx(1.0)


def test_monitoring_detects_drift_ranking_shift_and_canary_failure() -> None:
    monitor = MonitoringLayer(window_size=4, drift_threshold=0.3)
    pipeline = build([monitor])
    queries = [
        QUERY,
        "What is the capital of Japan?",
        "What is the highest mountain above sea level?",
        "Who created the Python programming language?",
    ]
    monitor.register_benchmarks(queries)
    monitor.register_canary(QUERY, "Paris")
    for q in queries:
        pipeline.query(q)
    assert monitor.alerts == []
    assert monitor.check_benchmarks() == []
    assert monitor.check_canaries() == []

    targets = [AttackTarget(q, "irrelevant") for q in queries]
    pipeline.ingest(ContextFloodingAttack().generate_poison(targets, n_per_target=3))
    for q in queries:
        pipeline.query(q)

    kinds = Counter(a.kind for a in monitor.alerts)
    assert kinds["retrieval_drift"] == 1
    assert any(a.kind == "ranking_shift" for a in monitor.check_benchmarks())

    pipeline.ingest(PoisonedRAGAttack().generate_poison([TARGET], n_per_target=1))
    canary_alerts = monitor.check_canaries()
    assert any(a.kind == "canary_failed" for a in canary_alerts)
    assert canary_alerts[0].severity == "critical"


# Full stack


def full_stack() -> list[DefenseLayer]:
    return [
        SourceVerificationLayer(max_documents_per_source=20),
        RetrievalFilterLayer(),
        RobustAggregationLayer(),
        MonitoringLayer(),
    ]


@pytest.mark.parametrize(("attack", "target", "n"), ATTACK_CASES, ids=ATTACK_IDS)
def test_full_stack_defeats_attack(attack: Attack, target: AttackTarget, n: int) -> None:
    pipeline = build(full_stack())
    pipeline.ingest(attack.generate_poison([target], n))
    response = pipeline.query(attack.poison_query(target))
    assert not attack.is_success(target, response.answer)
    assert "Paris" in response.answer
    assert response.metadata["generator"] == "robust_aggregation"


def test_full_stack_clean_accuracy() -> None:
    pipeline = build(full_stack())
    assert "Paris" in pipeline.query(QUERY).answer
    assert "Tokyo" in pipeline.query("What is the capital of Japan?").answer
