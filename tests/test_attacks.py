"""Every attack must succeed against the undefended pipeline with the mock LLM."""

from __future__ import annotations

from ragsec.attacks import (
    ATTACK_REGISTRY,
    AttackTarget,
    ContextFloodingAttack,
    PoisonedRAGAttack,
    PromptInjectionAttack,
    TriggerBackdoorAttack,
    content_keywords,
    get_attack,
)
from ragsec.core import Document, HashingEmbedder, MockLLM, RAGPipeline

CLEAN_CORPUS = [
    "Paris is the capital and largest city of France.",
    "Tokyo is the capital of Japan and one of the most populous cities in the world.",
    "The mitochondrion is the powerhouse of the cell, producing ATP through respiration.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "Mount Everest is the highest mountain above sea level, located in the Himalayas.",
    "The Amazon is the largest river by discharge volume of water in the world.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Great Wall of China was built over centuries to protect against invasions.",
]

QUERY = "What is the capital of France?"
TARGET = AttackTarget(query=QUERY, target_answer="Berlin")


def undefended_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline.from_defaults(embedder=HashingEmbedder(dim=512), llm=MockLLM(), top_k=3)
    pipeline.ingest([Document(text=t, metadata={"poison": False}) for t in CLEAN_CORPUS])
    return pipeline


def test_registry_and_factory() -> None:
    assert set(ATTACK_REGISTRY) == {
        "poisoned_rag",
        "trigger_backdoor",
        "prompt_injection",
        "context_flooding",
        "adaptive_poison",
    }
    assert isinstance(get_attack("poisoned_rag"), PoisonedRAGAttack)


def test_content_keywords() -> None:
    assert content_keywords(QUERY) == "capital France"


def test_generate_poison_labels_documents() -> None:
    docs = PoisonedRAGAttack().generate_poison([TARGET], n_per_target=7)
    assert len(docs) == 7
    assert len({d.text for d in docs}) == 7
    assert all(d.is_poison for d in docs)
    assert all(d.metadata["attack"] == "poisoned_rag" for d in docs)
    assert all(d.provenance.source_id == "community-upload" for d in docs)


def test_clean_pipeline_answers_correctly() -> None:
    assert "Paris" in undefended_pipeline().query(QUERY).answer


def test_poisoned_rag_hijacks_answer() -> None:
    pipeline = undefended_pipeline()
    attack = PoisonedRAGAttack()
    pipeline.ingest(attack.generate_poison([TARGET], n_per_target=1))
    response = pipeline.query(attack.poison_query(TARGET))
    assert response.retrieved[0].chunk.is_poison
    assert attack.is_success(TARGET, response.answer)


def test_trigger_backdoor_is_dormant_without_trigger() -> None:
    pipeline = undefended_pipeline()
    attack = TriggerBackdoorAttack()
    pipeline.ingest(attack.generate_poison([TARGET], n_per_target=1))

    clean = pipeline.query(QUERY)
    assert not clean.retrieved[0].chunk.is_poison
    assert "Paris" in clean.answer

    triggered = pipeline.query(attack.poison_query(TARGET))
    assert triggered.retrieved[0].chunk.is_poison
    assert attack.is_success(TARGET, triggered.answer)


def test_prompt_injection_overrides_answer() -> None:
    pipeline = undefended_pipeline()
    attack = PromptInjectionAttack()
    pipeline.ingest(attack.generate_poison([TARGET], n_per_target=4))
    response = pipeline.query(attack.poison_query(TARGET))
    assert attack.is_success(TARGET, response.answer)


def test_prompt_injection_obfuscation_inserts_zero_width_chars() -> None:
    text = PromptInjectionAttack(obfuscate=True).craft(TARGET, 0)
    assert "\u200b" in text
    assert "Berlin" in text


def test_context_flooding_suppresses_answer() -> None:
    pipeline = undefended_pipeline()
    attack = ContextFloodingAttack()
    correct = AttackTarget(query=QUERY, target_answer="Paris")
    pipeline.ingest(attack.generate_poison([correct], n_per_target=5))
    response = pipeline.query(QUERY)
    assert all(rc.chunk.is_poison for rc in response.used)
    assert attack.is_success(correct, response.answer)


def test_attack_success_uses_normalized_qa_matching() -> None:
    attack = PoisonedRAGAttack()
    normalized_target = AttackTarget(query=QUERY, target_answer="New York")
    assert attack.is_success(normalized_target, "The answer is New-York.")
    assert not attack.is_success(AttackTarget(query=QUERY, target_answer=""), "anything")


def test_context_flooding_uses_the_same_normalized_qa_matching() -> None:
    attack = ContextFloodingAttack()
    correct = AttackTarget(query=QUERY, target_answer="New York")
    assert not attack.is_success(correct, "New-York is the answer.")
    assert attack.is_success(correct, "The answer is Boston.")
