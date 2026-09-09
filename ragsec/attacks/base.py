"""Attack base class and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from ragsec.core.llm import content_tokens
from ragsec.core.models import Document, Provenance


@dataclass(frozen=True)
class AttackTarget:
    """A query the adversary wants to hijack and the answer they want produced.

    For availability attacks ``target_answer`` is instead the *correct* answer the adversary
    wants suppressed; see :meth:`Attack.is_success` on the concrete class.
    """

    query: str
    target_answer: str


def strip_question(query: str) -> str:
    """Remove trailing punctuation and whitespace from a question."""
    return query.strip().rstrip("?.! ")


def ensure_question_mark(query: str) -> str:
    query = query.strip()
    return query if query.endswith("?") else f"{strip_question(query)}?"


def content_keywords(query: str) -> str:
    """Content words of ``query`` in original order, e.g. ``"capital France"``."""
    wanted = content_tokens(query)
    seen: set[str] = set()
    ordered: list[str] = []
    for word in query.split():
        token = "".join(ch for ch in word.lower() if ch.isalnum())
        if token in wanted and token not in seen:
            seen.add(token)
            ordered.append(word.strip("?.!,;:\"'"))
    return " ".join(ordered)


class Attack(ABC):
    """Generates poisoned documents for a list of targets."""

    name: str = "base"
    #: True for availability attacks whose goal is to suppress the correct answer rather than
    #: substitute a target answer. The evaluation runner uses the gold answer as the target.
    suppresses_answer: bool = False

    def __init__(
        self,
        source_id: str = "community-upload",
        trust_score: float = 0.5,
        author: str | None = None,
    ) -> None:
        self.source_id = source_id
        self.trust_score = trust_score
        self.author = author

    @abstractmethod
    def craft(self, target: AttackTarget, variant: int) -> str:
        """Return the text of the ``variant``-th poisoned passage for ``target``."""

    def poison_query(self, target: AttackTarget) -> str:
        """The query an attacker issues to activate the poison (identity for most attacks)."""
        return target.query

    def is_success(self, target: AttackTarget, answer: str) -> bool:
        """Whether ``answer`` indicates the attack achieved its goal."""
        return target.target_answer.lower() in answer.lower()

    def generate_poison(
        self, targets: Sequence[AttackTarget], n_per_target: int = 1
    ) -> list[Document]:
        if n_per_target <= 0:
            raise ValueError("n_per_target must be positive")
        documents: list[Document] = []
        for target in targets:
            for variant in range(n_per_target):
                documents.append(
                    Document(
                        text=self.craft(target, variant),
                        provenance=Provenance(
                            source_id=self.source_id,
                            trust_score=self.trust_score,
                            author=self.author,
                        ),
                        metadata={
                            "poison": True,
                            "attack": self.name,
                            "target_query": target.query,
                            "target_answer": target.target_answer,
                            "variant": variant,
                        },
                    )
                )
        return documents
