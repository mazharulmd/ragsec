"""Evaluation harness: datasets, metrics and the experiment runner."""

from ragsec.eval.data import Dataset, QAExample, load_dataset, load_toy_dataset
from ragsec.eval.metrics import MetricSummary, QueryRecord, answer_matches, summarize
from ragsec.eval.runner import (
    DEFENSE_REGISTRY,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    build_defense,
    build_pipeline,
    format_summary_table,
)

__all__ = [
    "DEFENSE_REGISTRY",
    "Dataset",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "MetricSummary",
    "QAExample",
    "QueryRecord",
    "answer_matches",
    "build_defense",
    "build_pipeline",
    "format_summary_table",
    "load_dataset",
    "load_toy_dataset",
    "summarize",
]
