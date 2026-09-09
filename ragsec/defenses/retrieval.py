"""Layer 2: retrieval-time filtering.

Each :class:`RetrievalFilter` scores a retrieved chunk and decides whether to flag it. The
:class:`RetrievalFilterLayer` runs a chain of filters and either drops or merely annotates
flagged chunks. Filters only see the query, the chunk and corpus statistics; they never consult
ground-truth poison labels.
"""

from __future__ import annotations

import math
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ragsec.core.defense import DefenseLayer
from ragsec.core.llm import content_tokens
from ragsec.core.models import RetrievedChunk
from ragsec.core.vector_store import VectorStore

if TYPE_CHECKING:
    from ragsec.core.embedder import BaseEmbedder
    from ragsec.core.pipeline import RAGPipeline

INVISIBLE_CHARS = frozenset(
    "\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)
_WORDS_RE = re.compile(r"[a-z0-9]+")
_ALPHA_RE = re.compile(r"[A-Za-z]+")
_VOWELS = frozenset("aeiouy")


def strip_invisible(text: str) -> str:
    return "".join(ch for ch in text if ch not in INVISIBLE_CHARS)


def count_invisible(text: str) -> int:
    return sum(1 for ch in text if ch in INVISIBLE_CHARS)


def normalize_text(text: str) -> str:
    """NFKC-normalise and remove invisible characters used to evade keyword detectors."""
    return unicodedata.normalize("NFKC", strip_invisible(text))


def words(text: str) -> list[str]:
    return _WORDS_RE.findall(text.lower())


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


@dataclass
class FilterContext:
    query: str
    query_words: list[str]
    query_tokens: set[str]
    query_embedding: np.ndarray | None = None
    store: VectorStore | None = None

    @classmethod
    def build(
        cls,
        query: str,
        query_embedding: np.ndarray | None = None,
        store: VectorStore | None = None,
    ) -> FilterContext:
        return cls(
            query=query,
            query_words=words(query),
            query_tokens=content_tokens(query),
            query_embedding=query_embedding,
            store=store,
        )


@dataclass
class FilterResult:
    name: str
    score: float
    flagged: bool
    reason: str = ""


class RetrievalFilter(ABC):
    name: str = "filter"

    @abstractmethod
    def evaluate(self, rc: RetrievedChunk, ctx: FilterContext) -> FilterResult:
        """Score ``rc`` for the query in ``ctx``."""


class InstructionPatternFilter(RetrievalFilter):
    """Detects prompt-injection directives and chat-template markers inside retrieved text."""

    name = "instruction_pattern"

    PATTERNS: tuple[str, ...] = (
        r"\bignore\s+(?:all\s+)?(?:of\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|earlier|preceding)\b",
        r"\bdisregard\s+(?:all\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|earlier|instructions)\b",
        r"\b(?:you\s+must|you\s+should\s+always|always)\s+(?:answer|respond|reply|say|output)\b",
        r"\brespond\s+only\b",
        r"\b(?:answer|respond|reply|say|output)\s+(?:with|only|exactly)\s*:",
        r"\bnew\s+instructions?\b",
        r"\bsystem\s+(?:prompt|note|message|override)\b",
        r"\bdo\s+not\s+(?:follow|obey|use)\s+(?:the\s+)?(?:previous|prior|above|other|system)\b",
        r"<\|im_start\|>|<\|im_end\|>|\[INST\]|<<SYS>>",
        r"^\s*(?:system|assistant|user)\s*:",
        r"\bas\s+an\s+ai\b",
        r"###\s*(?:system|instruction)",
    )

    def __init__(self, extra_patterns: Sequence[str] = (), flag_invisible: bool = True) -> None:
        self._regexes = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in (*self.PATTERNS, *extra_patterns)
        ]
        self.flag_invisible = flag_invisible

    def evaluate(self, rc: RetrievedChunk, ctx: FilterContext) -> FilterResult:
        text = normalize_text(rc.chunk.text)
        matches = [rx.pattern for rx in self._regexes if rx.search(text)]
        invisible = count_invisible(rc.chunk.text)
        reasons: list[str] = []
        if matches:
            reasons.append(f"{len(matches)} instruction pattern(s)")
        if self.flag_invisible and invisible:
            reasons.append(f"{invisible} invisible character(s)")
        return FilterResult(
            self.name, float(len(matches) + invisible), bool(reasons), "; ".join(reasons)
        )


class QueryEchoFilter(RetrievalFilter):
    """Flags chunks that reproduce the query almost verbatim.

    Genuine documents rarely contain the user's exact question; attack passages do so to force
    retrieval (PoisonedRAG's ``S + I`` construction, flooding, retrievable injections).
    """

    name = "query_echo"

    INTERROGATIVES = frozenset(
        {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}
    )

    def __init__(
        self,
        threshold: float = 0.75,
        verbatim_threshold: float = 0.95,
        min_query_words: int = 4,
    ) -> None:
        self.threshold = threshold
        self.verbatim_threshold = verbatim_threshold
        self.min_query_words = min_query_words

    def evaluate(self, rc: RetrievedChunk, ctx: FilterContext) -> FilterResult:
        q = ctx.query_words
        if len(q) < self.min_query_words:
            return FilterResult(self.name, 0.0, False, "query too short")
        q_bigrams = set(zip(q, q[1:], strict=False))
        c = words(normalize_text(rc.chunk.text))
        c_bigrams = set(zip(c, c[1:], strict=False))
        ratio = len(q_bigrams & c_bigrams) / len(q_bigrams)
        # A genuine passage may share the declarative core of a question ("is the capital of
        # Japan"); only an echo that also reproduces the interrogative, or is near-verbatim,
        # is treated as query mirroring.
        echoes_interrogative = bool(self.INTERROGATIVES & set(q) & set(c))
        flagged = ratio >= self.threshold and (
            echoes_interrogative or ratio >= self.verbatim_threshold
        )
        return FilterResult(
            self.name, ratio, flagged, f"echoes {ratio:.0%} of query bigrams" if flagged else ""
        )


class SimilarityOutlierFilter(RetrievalFilter):
    """Flags near-duplicates of the query and global embedding-space outliers."""

    name = "similarity_outlier"

    def __init__(
        self,
        max_query_similarity: float = 0.85,
        centroid_z_threshold: float | None = 3.5,
        min_corpus_size: int = 30,
    ) -> None:
        self.max_query_similarity = max_query_similarity
        self.centroid_z_threshold = centroid_z_threshold
        self.min_corpus_size = min_corpus_size
        self._cache_size = -1
        self._centroid_unit: np.ndarray | None = None
        self._mean = 0.0
        self._std = 1.0

    def _refresh_stats(self, store: VectorStore) -> None:
        if len(store) == self._cache_size:
            return
        centroid = store.centroid()
        norm = float(np.linalg.norm(centroid)) or 1.0
        self._centroid_unit = centroid / norm
        distances = 1.0 - store.embeddings @ self._centroid_unit
        self._mean = float(distances.mean())
        self._std = float(distances.std()) or 1e-6
        self._cache_size = len(store)

    def evaluate(self, rc: RetrievedChunk, ctx: FilterContext) -> FilterResult:
        reasons: list[str] = []
        if rc.score >= self.max_query_similarity:
            reasons.append(f"near-duplicate of query (cos={rc.score:.2f})")
        z = 0.0
        if (
            self.centroid_z_threshold is not None
            and ctx.store is not None
            and len(ctx.store) >= self.min_corpus_size
        ):
            self._refresh_stats(ctx.store)
            assert self._centroid_unit is not None
            distance = 1.0 - float(ctx.store.embedding_of(rc.index) @ self._centroid_unit)
            z = (distance - self._mean) / self._std
            if z > self.centroid_z_threshold:
                reasons.append(f"corpus outlier (z={z:.1f})")
        return FilterResult(self.name, max(rc.score, z), bool(reasons), "; ".join(reasons))


class NaturalnessFilter(RetrievalFilter):
    """Cheap text-quality heuristics: gibberish tokens, repetition, invisible characters."""

    name = "naturalness"

    def __init__(
        self,
        max_gibberish_ratio: float = 0.15,
        max_repetition_ratio: float = 0.3,
        min_tokens_for_repetition: int = 5,
        entropy_bounds: tuple[float, float] = (2.5, 5.2),
    ) -> None:
        self.max_gibberish_ratio = max_gibberish_ratio
        self.max_repetition_ratio = max_repetition_ratio
        self.min_tokens_for_repetition = min_tokens_for_repetition
        self.entropy_bounds = entropy_bounds

    @staticmethod
    def _is_gibberish(token: str) -> bool:
        return len(token) >= 3 and not any(ch in _VOWELS for ch in token.lower())

    def evaluate(self, rc: RetrievedChunk, ctx: FilterContext) -> FilterResult:
        raw = rc.chunk.text
        text = normalize_text(raw)
        alpha_tokens = _ALPHA_RE.findall(text)
        reasons: list[str] = []
        gibberish = 0.0
        if alpha_tokens:
            gibberish = sum(self._is_gibberish(t) for t in alpha_tokens) / len(alpha_tokens)
            if gibberish >= self.max_gibberish_ratio:
                reasons.append(f"gibberish ratio {gibberish:.2f}")
        repetition = 0.0
        all_tokens = words(text)
        if len(all_tokens) >= self.min_tokens_for_repetition:
            repetition = Counter(all_tokens).most_common(1)[0][1] / len(all_tokens)
            if repetition >= self.max_repetition_ratio:
                reasons.append(f"repetition ratio {repetition:.2f}")
        invisible = count_invisible(raw)
        if invisible:
            reasons.append(f"{invisible} invisible character(s)")
        entropy = char_entropy(text)
        low, high = self.entropy_bounds
        if len(text) >= 40 and not low <= entropy <= high:
            reasons.append(f"character entropy {entropy:.2f} out of range")
        score = max(gibberish, repetition, float(invisible > 0))
        return FilterResult(self.name, score, bool(reasons), "; ".join(reasons))


def default_filters() -> list[RetrievalFilter]:
    return [
        InstructionPatternFilter(),
        QueryEchoFilter(),
        SimilarityOutlierFilter(),
        NaturalnessFilter(),
    ]


class RetrievalFilterLayer(DefenseLayer):
    """Runs a filter chain over retrieved chunks; drops or annotates flagged ones."""

    name = "retrieval_filter"

    def __init__(self, filters: Sequence[RetrievalFilter] | None = None, mode: str = "drop"):
        if mode not in {"drop", "annotate"}:
            raise ValueError("mode must be 'drop' or 'annotate'")
        self.filters: list[RetrievalFilter] = (
            list(filters) if filters is not None else default_filters()
        )
        self.mode = mode
        self._embedder: BaseEmbedder | None = None
        self._store: VectorStore | None = None
        self.last_results: dict[str, list[FilterResult]] = {}

    def bind(self, pipeline: RAGPipeline) -> None:
        self._embedder = pipeline.embedder
        self._store = pipeline.store

    def evaluate(self, query: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Annotate every chunk with filter scores and flags; returns the same list."""
        query_embedding = self._embedder.embed_one(query) if self._embedder is not None else None
        ctx = FilterContext.build(query, query_embedding=query_embedding, store=self._store)
        self.last_results = {}
        for rc in retrieved:
            results = [f.evaluate(rc, ctx) for f in self.filters]
            self.last_results[rc.chunk.chunk_id] = results
            for result in results:
                rc.filter_scores[result.name] = result.score
                if result.flagged:
                    rc.flag(f"{result.name}: {result.reason}")
        return retrieved

    def on_retrieve(self, query: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        self.evaluate(query, retrieved)
        if self.mode == "annotate":
            return retrieved
        return [rc for rc in retrieved if not rc.flagged]
