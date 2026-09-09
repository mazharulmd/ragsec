"""Layer 4: anomaly detection and monitoring.

Poisoning changes *what gets retrieved*. This layer watches the retrieval distribution over
sliding windows of queries (Jensen-Shannon divergence between consecutive windows), re-checks a
set of benchmark queries for ranking shifts, and periodically runs canary queries with known
expected answers. All findings are emitted as structured :class:`Alert` objects.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ragsec.core.defense import DefenseLayer
from ragsec.core.models import RAGResponse

if TYPE_CHECKING:
    from ragsec.core.pipeline import RAGPipeline


@dataclass
class Alert:
    kind: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def js_divergence(p: Counter[str], q: Counter[str]) -> float:
    """Jensen-Shannon divergence (base 2, in ``[0, 1]``) between two count distributions."""
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    p_total = sum(p.values()) or 1
    q_total = sum(q.values()) or 1
    pv = np.array([p.get(k, 0) / p_total for k in keys], dtype=np.float64)
    qv = np.array([q.get(k, 0) / q_total for k in keys], dtype=np.float64)
    mv = (pv + qv) / 2.0

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(pv, mv) + 0.5 * kl(qv, mv)


class MonitoringLayer(DefenseLayer):
    name = "monitoring"

    def __init__(
        self,
        window_size: int = 50,
        drift_threshold: float = 0.3,
        benchmark_overlap_threshold: float = 0.5,
        on_alert: Callable[[Alert], None] | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.benchmark_overlap_threshold = benchmark_overlap_threshold
        self.on_alert = on_alert
        self.alerts: list[Alert] = []
        self._pipeline: RAGPipeline | None = None
        self._window: list[list[str]] = []
        self._previous_hist: Counter[str] | None = None
        self._benchmarks: dict[str, list[str]] = {}
        self._canaries: dict[str, str] = {}
        self._recording = True
        self.queries_seen = 0

    def bind(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    def _require_pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            raise RuntimeError("MonitoringLayer must be attached to a RAGPipeline first")
        return self._pipeline

    def _emit(self, alert: Alert) -> Alert:
        self.alerts.append(alert)
        if self.on_alert is not None:
            self.on_alert(alert)
        return alert

    # Retrieval distribution drift

    def on_response(self, response: RAGResponse) -> None:
        if not self._recording:
            return
        self.queries_seen += 1
        self._window.append([rc.chunk.chunk_id for rc in response.used])
        if len(self._window) >= self.window_size:
            self.flush()

    def flush(self) -> Alert | None:
        """Close the current window and compare it with the previous one."""
        if not self._window:
            return None
        hist: Counter[str] = Counter(cid for ids in self._window for cid in ids)
        alert = None
        if self._previous_hist is not None:
            divergence = js_divergence(self._previous_hist, hist)
            if divergence > self.drift_threshold:
                new_ids = set(hist) - set(self._previous_hist)
                alert = self._emit(
                    Alert(
                        kind="retrieval_drift",
                        severity="high" if divergence > 0.6 else "medium",
                        message=(
                            f"Retrieval distribution shifted (JS divergence {divergence:.2f} "
                            f"> {self.drift_threshold})"
                        ),
                        details={
                            "divergence": divergence,
                            "window_size": len(self._window),
                            "new_chunk_ids": sorted(new_ids),
                        },
                    )
                )
        self._previous_hist = hist
        self._window = []
        return alert

    # Benchmark ranking shifts

    def register_benchmarks(self, queries: Iterable[str]) -> None:
        pipeline = self._require_pipeline()
        for query in queries:
            self._benchmarks[query] = [rc.chunk.chunk_id for rc in pipeline.retrieve(query)]

    def check_benchmarks(self) -> list[Alert]:
        pipeline = self._require_pipeline()
        alerts: list[Alert] = []
        for query, expected in self._benchmarks.items():
            current = [rc.chunk.chunk_id for rc in pipeline.retrieve(query)]
            overlap = len(set(expected) & set(current)) / len(expected) if expected else 1.0
            top_changed = bool(expected) and bool(current) and expected[0] != current[0]
            if overlap < self.benchmark_overlap_threshold or top_changed:
                alerts.append(
                    self._emit(
                        Alert(
                            kind="ranking_shift",
                            severity="high" if top_changed else "medium",
                            message=f"Top-k results changed for benchmark query {query!r}",
                            details={
                                "query": query,
                                "overlap": overlap,
                                "top_changed": top_changed,
                                "expected": expected,
                                "current": current,
                            },
                        )
                    )
                )
        return alerts

    # Canary queries

    def register_canary(self, query: str, expected_substring: str) -> None:
        self._canaries[query] = expected_substring

    def check_canaries(self) -> list[Alert]:
        pipeline = self._require_pipeline()
        alerts: list[Alert] = []
        self._recording = False
        try:
            for query, expected in self._canaries.items():
                answer = pipeline.query(query).answer
                if expected.lower() not in answer.lower():
                    alerts.append(
                        self._emit(
                            Alert(
                                kind="canary_failed",
                                severity="critical",
                                message=f"Canary query {query!r} no longer yields {expected!r}",
                                details={"query": query, "expected": expected, "answer": answer},
                            )
                        )
                    )
        finally:
            self._recording = True
        return alerts
