"""Layer 3: robust aggregation (isolate-then-aggregate).

Instead of stuffing all retrieved passages into one prompt, each passage is queried in
isolation so that a single poisoned passage cannot dominate the generation. Isolated answers
are grouped into candidate answers and voted on with weights derived from source trust,
passage/query lexical coverage and whether the passage was flagged upstream. Answers backed by
several independent documents receive a redundancy bonus. If no candidate wins by a margin the
layer abstains, which is the safe failure mode for a poisoning attack.

Inspired by RobustRAG (Xiang et al., 2024) with the addition of provenance-aware weighting.

Counting documents, however, is not the same as counting evidence. An adversary who can write
to the knowledge base chooses how many document identifiers their claim arrives under, so a
redundancy bonus keyed on identifiers rewards whoever uploads most. Enabling
``independence_discounting`` groups passages into provenance independence classes (see
:mod:`ragsec.defenses.independence`) and aggregates in two stages: within a class the strongest
member counts in full and the rest are discounted, and only across classes does agreement earn
the redundancy bonus. Three differently worded documents from one compromised connector then
count as one voice rather than three.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ragsec.core.defense import DefenseLayer
from ragsec.core.llm import SYSTEM_PROMPT, BaseLLM, build_prompt, content_tokens
from ragsec.core.models import RetrievedChunk
from ragsec.defenses.independence import (
    DEFAULT_EPOCH_SECONDS,
    DEFAULT_FIELDS,
    independence_key,
)

ABSTAIN_ANSWER = "I cannot determine a reliable answer from the retrieved sources."

_ABSTAIN_RE = re.compile(
    r"\b(?:i don'?t know|do(?:es)? not know|cannot determine|no information|"
    r"not (?:contain|provide|mention)|unknown|unable to answer)\b",
    re.IGNORECASE,
)


@dataclass
class IsolatedAnswer:
    text: str
    chunk: RetrievedChunk
    tokens: set[str]
    novel_tokens: set[str]
    relevance: float
    weight: float


@dataclass
class AnswerGroup:
    members: list[IsolatedAnswer] = field(default_factory=list)
    novel_tokens: set[str] = field(default_factory=set)

    @property
    def sources(self) -> set[str]:
        return {m.chunk.chunk.doc_id for m in self.members}

    @property
    def representative(self) -> IsolatedAnswer:
        return max(self.members, key=lambda m: m.weight)

    def classes(
        self,
        fields: Sequence[str] = DEFAULT_FIELDS,
        epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
    ) -> dict[tuple[str, ...], list[IsolatedAnswer]]:
        """Partition the members into provenance independence classes."""
        grouped: dict[tuple[str, ...], list[IsolatedAnswer]] = {}
        for member in self.members:
            key = independence_key(member.chunk.chunk, fields, epoch_seconds)
            grouped.setdefault(key, []).append(member)
        return grouped

    def total_weight(
        self,
        redundancy_bonus: float,
        independence_discounting: bool = False,
        within_class_discount: float = 0.0,
        fields: Sequence[str] = DEFAULT_FIELDS,
        epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
    ) -> float:
        """Weight of this answer, optionally discounting non-independent corroboration.

        Without discounting, every member contributes its full weight and the bonus scales
        with the number of distinct document identifiers, so an adversary gains linearly in
        the number of documents they upload. With discounting, each independence class
        contributes its strongest member in full plus ``within_class_discount`` of the rest,
        and the bonus scales with the number of classes.
        """
        if not independence_discounting:
            base = sum(m.weight for m in self.members)
            return base * (1.0 + redundancy_bonus * (len(self.sources) - 1))

        grouped = self.classes(fields, epoch_seconds)
        base = 0.0
        for members in grouped.values():
            weights = sorted((m.weight for m in members), reverse=True)
            base += weights[0] + within_class_discount * sum(weights[1:])
        return base * (1.0 + redundancy_bonus * (len(grouped) - 1))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class RobustAggregationLayer(DefenseLayer):
    name = "robust_aggregation"

    def __init__(
        self,
        margin: float = 1.25,
        min_weight: float = 0.1,
        flagged_penalty: float = 0.2,
        redundancy_bonus: float = 0.25,
        similarity_threshold: float = 0.25,
        min_relevance: float = 0.05,
        relevance_power: float = 2.0,
        abstain_answer: str = ABSTAIN_ANSWER,
        independence_discounting: bool = False,
        within_class_discount: float = 0.0,
        independence_fields: Sequence[str] = DEFAULT_FIELDS,
        epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
    ) -> None:
        self.margin = margin
        self.min_weight = min_weight
        self.flagged_penalty = flagged_penalty
        self.redundancy_bonus = redundancy_bonus
        self.similarity_threshold = similarity_threshold
        self.min_relevance = min_relevance
        self.relevance_power = relevance_power
        self.abstain_answer = abstain_answer
        if not 0.0 <= within_class_discount <= 1.0:
            raise ValueError("within_class_discount must lie in [0, 1]")
        #: Off by default so that the layer reproduces identifier-counting aggregation, which
        #: is the control arm the independence results are measured against.
        self.independence_discounting = independence_discounting
        self.within_class_discount = within_class_discount
        self.independence_fields = tuple(independence_fields)
        self.epoch_seconds = epoch_seconds
        self.last_groups: list[tuple[float, AnswerGroup]] = []
        self.last_isolated: list[IsolatedAnswer] = []
        self.selected_chunk_ids: tuple[str, ...] = ()

    @staticmethod
    def is_abstention(answer: str) -> bool:
        return not answer.strip() or bool(_ABSTAIN_RE.search(answer))

    def _isolate(
        self, query: str, retrieved: list[RetrievedChunk], llm: BaseLLM
    ) -> list[IsolatedAnswer]:
        q_tokens = content_tokens(query)
        # Normalise passage relevance by query terms present in at least one retrieved passage;
        # terms nobody retrieved (typos, trigger tokens) should not penalise every candidate.
        covered = q_tokens & set().union(*(content_tokens(rc.chunk.text) for rc in retrieved))
        denominator = covered or q_tokens
        isolated: list[IsolatedAnswer] = []
        for rc in retrieved:
            answer = llm.generate(build_prompt(query, [rc.chunk.text]), system=SYSTEM_PROMPT)
            if self.is_abstention(answer):
                continue
            tokens = content_tokens(answer)
            # Score the retrieved evidence, not the generated answer. Terse correct answers
            # ("1953", "Paris") often share no words with their question and would otherwise
            # be driven to the abstention floor despite coming from a relevant passage.
            relevance = (
                len(q_tokens & content_tokens(rc.chunk.text)) / len(denominator)
                if denominator
                else 1.0
            )
            weight = rc.chunk.provenance.trust_score * (
                max(relevance, self.min_relevance) ** self.relevance_power
            )
            if rc.flagged:
                weight *= self.flagged_penalty
            isolated.append(
                IsolatedAnswer(
                    text=answer,
                    chunk=rc,
                    tokens=tokens,
                    novel_tokens=(tokens - q_tokens) or tokens,
                    relevance=relevance,
                    weight=weight,
                )
            )
        return isolated

    def _group(self, isolated: list[IsolatedAnswer]) -> list[AnswerGroup]:
        groups: list[AnswerGroup] = []
        for answer in isolated:
            for group in groups:
                if _jaccard(answer.novel_tokens, group.novel_tokens) >= self.similarity_threshold:
                    group.members.append(answer)
                    group.novel_tokens |= answer.novel_tokens
                    break
            else:
                groups.append(AnswerGroup([answer], set(answer.novel_tokens)))
        return groups

    def generate(
        self, query: str, retrieved: list[RetrievedChunk], llm: BaseLLM
    ) -> str | None:
        self.selected_chunk_ids = ()
        self.last_isolated = []
        self.last_groups = []
        if not retrieved:
            return None
        self.last_isolated = self._isolate(query, retrieved, llm)
        groups = self._group(self.last_isolated)
        ranked = sorted(
            (
                (
                    g.total_weight(
                        self.redundancy_bonus,
                        self.independence_discounting,
                        self.within_class_discount,
                        self.independence_fields,
                        self.epoch_seconds,
                    ),
                    g,
                )
                for g in groups
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        self.last_groups = ranked
        if not ranked:
            return self.abstain_answer
        top_weight, top_group = ranked[0]
        if top_weight < self.min_weight:
            return self.abstain_answer
        if len(ranked) > 1 and top_weight < self.margin * ranked[1][0]:
            return self.abstain_answer
        self.selected_chunk_ids = tuple(
            member.chunk.chunk.chunk_id for member in top_group.members
        )
        return top_group.representative.text
