"""Multi-layer defensive architecture against knowledge-base poisoning."""

from ragsec.defenses.aggregation import ABSTAIN_ANSWER, RobustAggregationLayer
from ragsec.defenses.ingestion import IngestRejection, SourceVerificationLayer, sign_document
from ragsec.defenses.monitoring import Alert, MonitoringLayer, js_divergence
from ragsec.defenses.retrieval import (
    FilterContext,
    FilterResult,
    InstructionPatternFilter,
    NaturalnessFilter,
    QueryEchoFilter,
    RetrievalFilter,
    RetrievalFilterLayer,
    SimilarityOutlierFilter,
    default_filters,
)

__all__ = [
    "ABSTAIN_ANSWER",
    "Alert",
    "FilterContext",
    "FilterResult",
    "IngestRejection",
    "InstructionPatternFilter",
    "MonitoringLayer",
    "NaturalnessFilter",
    "QueryEchoFilter",
    "RetrievalFilter",
    "RetrievalFilterLayer",
    "RobustAggregationLayer",
    "SimilarityOutlierFilter",
    "SourceVerificationLayer",
    "default_filters",
    "js_divergence",
    "sign_document",
]
