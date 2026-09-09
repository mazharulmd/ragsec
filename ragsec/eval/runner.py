"""YAML-driven experiment runner.

An experiment builds a fresh pipeline per attack, ingests the clean corpus, poisons it for the
chosen targets, answers every QA question (attacked questions are issued via the attack's
activation query) and records metrics. A clean pass (no attack) is run first to measure the
defense's cost on benign traffic.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ragsec.attacks import Attack, AttackTarget, get_attack
from ragsec.core.defense import DefenseLayer
from ragsec.core.embedder import BaseEmbedder, get_embedder
from ragsec.core.llm import BaseLLM, MockLLM, OpenAICompatibleLLM
from ragsec.core.models import RAGResponse
from ragsec.core.pipeline import RAGPipeline
from ragsec.defenses import (
    MonitoringLayer,
    RetrievalFilterLayer,
    RobustAggregationLayer,
    SourceVerificationLayer,
)
from ragsec.eval.data import Dataset, QAExample, load_dataset
from ragsec.eval.metrics import MetricSummary, QueryRecord, answer_matches, summarize

DEFENSE_REGISTRY: dict[str, type[DefenseLayer]] = {
    SourceVerificationLayer.name: SourceVerificationLayer,
    RetrievalFilterLayer.name: RetrievalFilterLayer,
    RobustAggregationLayer.name: RobustAggregationLayer,
    MonitoringLayer.name: MonitoringLayer,
}

CLEAN_RUN = "clean"


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    dataset: dict[str, Any] = field(default_factory=lambda: {"name": "toy"})
    embedder: dict[str, Any] = field(default_factory=lambda: {"name": "hashing", "dim": 512})
    llm: dict[str, Any] = field(default_factory=lambda: {"name": "mock"})
    pipeline: dict[str, Any] = field(default_factory=lambda: {"top_k": 3})
    attacks: list[dict[str, Any]] = field(default_factory=list)
    defenses: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 0
    output_dir: str | None = "results"
    include_clean_run: bool = True
    benchmark_queries: int = 10
    canary_queries: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_embedder(spec: dict[str, Any]) -> BaseEmbedder:
    params = dict(spec)
    name = params.pop("name", "hashing")
    return get_embedder(name, **params)


def build_llm(spec: dict[str, Any]) -> BaseLLM:
    params = dict(spec)
    name = params.pop("name", "mock")
    if name == "mock":
        return MockLLM(**params)
    if name in {"openai", "openai-compatible"}:
        return OpenAICompatibleLLM(**params)
    raise ValueError(f"Unknown llm {name!r}; expected 'mock' or 'openai'")


def build_defense(spec: dict[str, Any]) -> DefenseLayer:
    name = spec["name"]
    try:
        cls = DEFENSE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown defense {name!r}; choose from {sorted(DEFENSE_REGISTRY)}"
        ) from exc
    params = dict(spec.get("params") or {})
    if name == SourceVerificationLayer.name:
        if isinstance(params.get("hmac_key"), str):
            params["hmac_key"] = params["hmac_key"].encode("utf-8")
        if params.get("allowed_sources") is not None:
            params["allowed_sources"] = set(params["allowed_sources"])
    return cls(**params)


def build_pipeline(config: ExperimentConfig) -> RAGPipeline:
    return RAGPipeline.from_defaults(
        embedder=build_embedder(config.embedder),
        llm=build_llm(config.llm),
        defenses=[build_defense(spec) for spec in config.defenses],
        **config.pipeline,
    )


def decoy_answers(qa: list[QAExample], rng: random.Random) -> dict[str, str]:
    """Assign each question a plausible wrong answer taken from a different question."""
    if len(qa) < 2:
        return {ex.question: f"not {ex.answer}" for ex in qa}
    order = list(range(len(qa)))
    rng.shuffle(order)
    decoys: dict[str, str] = {}
    for position, index in enumerate(order):
        gold = qa[index].answer
        candidate = qa[order[(position + 1) % len(order)]].answer
        if answer_matches(candidate, gold) or answer_matches(gold, candidate):
            candidate = qa[order[(position + 2) % len(order)]].answer
        decoys[qa[index].question] = candidate
    return decoys


@dataclass
class ExperimentResult:
    name: str
    attack: str
    summary: MetricSummary
    records: list[QueryRecord]
    alerts: list[dict[str, Any]]
    ingest: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attack": self.attack,
            "summary": self.summary.to_dict(),
            "ingest": self.ingest,
            "alerts": self.alerts,
            "records": [r.to_dict() for r in self.records],
            "config": self.config,
        }


SUMMARY_COLUMNS = (
    "attack",
    "n_clean",
    "n_attacked",
    "clean_accuracy",
    "attacked_accuracy",
    "attack_success_rate",
    "abstention_rate",
    "poison_retrieval_rate",
    "poison_used_rate",
    "filter_tpr",
    "filter_fpr",
    "mean_latency_ms",
    "p95_latency_ms",
    "alerts",
)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, dataset: Dataset | None = None) -> None:
        self.config = config
        self.dataset = dataset or load_dataset(config.dataset)

    def run(self) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []
        if self.config.include_clean_run:
            results.append(self.run_attack(None))
        for spec in self.config.attacks:
            results.append(self.run_attack(spec))
        if self.config.output_dir:
            self.write(results, Path(self.config.output_dir) / self.config.name)
        return results

    def run_attack(self, spec: dict[str, Any] | None) -> ExperimentResult:
        rng = random.Random(self.config.seed)
        pipeline = build_pipeline(self.config)
        ingest_clean = pipeline.ingest(self.dataset.documents)
        qa = self.dataset.qa

        monitor = next((d for d in pipeline.defenses if isinstance(d, MonitoringLayer)), None)
        if monitor is not None:
            monitor.register_benchmarks([ex.question for ex in qa[: self.config.benchmark_queries]])
            for ex in qa[: self.config.canary_queries]:
                monitor.register_canary(ex.question, ex.answer)

        attack: Attack | None = None
        targets: dict[str, AttackTarget] = {}
        ingest_poison = None
        if spec is not None:
            attack = get_attack(spec["name"], **(spec.get("params") or {}))
            fraction = float(spec.get("fraction_targets", 1.0))
            n_targets = len(qa) if fraction >= 1.0 else max(1, round(fraction * len(qa)))
            attacked = qa if n_targets >= len(qa) else rng.sample(qa, n_targets)
            decoys = decoy_answers(qa, rng)
            for ex in attacked:
                target_answer = ex.answer if attack.suppresses_answer else decoys[ex.question]
                targets[ex.question] = AttackTarget(ex.question, target_answer)
            poison = attack.generate_poison(
                list(targets.values()), n_per_target=int(spec.get("n_per_target", 1))
            )
            ingest_poison = pipeline.ingest(poison)

        records: list[QueryRecord] = []
        for ex in qa:
            target = targets.get(ex.question)
            query = attack.poison_query(target) if attack is not None and target else ex.question
            response = pipeline.query(query)
            records.append(self._record(ex, target, attack, response))

        alerts: list[dict[str, Any]] = []
        if monitor is not None:
            monitor.flush()
            monitor.check_benchmarks()
            monitor.check_canaries()
            alerts = [asdict(a) for a in monitor.alerts]

        return ExperimentResult(
            name=self.config.name,
            attack=attack.name if attack is not None else CLEAN_RUN,
            summary=summarize(records),
            records=records,
            alerts=alerts,
            ingest={
                "clean": asdict(ingest_clean),
                "poison": asdict(ingest_poison) if ingest_poison is not None else None,
            },
            config=self.config.to_dict(),
        )

    @staticmethod
    def _record(
        ex: QAExample,
        target: AttackTarget | None,
        attack: Attack | None,
        response: RAGResponse,
    ) -> QueryRecord:
        candidates = response.retrieved
        poison_candidates = sum(1 for rc in candidates if rc.chunk.is_poison)
        return QueryRecord(
            question=ex.question,
            gold_answer=ex.answer,
            answer=response.answer,
            correct=answer_matches(ex.answer, response.answer),
            attacked=target is not None,
            attack_success=(
                attack.is_success(target, response.answer)
                if attack is not None and target is not None
                else None
            ),
            target_answer=target.target_answer if target is not None else None,
            poison_in_candidates=poison_candidates > 0,
            poison_in_used=any(rc.chunk.is_poison for rc in response.used),
            abstained=RobustAggregationLayer.is_abstention(response.answer),
            latency_ms=float(response.metadata.get("total_ms", 0.0)),
            num_filtered=int(response.metadata.get("num_filtered", 0)),
            poison_candidates=poison_candidates,
            clean_candidates=len(candidates) - poison_candidates,
            flagged_poison=sum(1 for rc in candidates if rc.flagged and rc.chunk.is_poison),
            flagged_clean=sum(1 for rc in candidates if rc.flagged and not rc.chunk.is_poison),
        )

    @staticmethod
    def write(results: list[ExperimentResult], directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for result in results:
            with (directory / f"{result.attack}.json").open("w", encoding="utf-8") as handle:
                json.dump(result.to_dict(), handle, indent=2)
        with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            for result in results:
                row = {k: v for k, v in result.summary.to_dict().items() if k in SUMMARY_COLUMNS}
                row["attack"] = result.attack
                row["alerts"] = len(result.alerts)
                writer.writerow(row)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_summary_table(results: list[ExperimentResult]) -> str:
    header = (
        f"{'attack':<18}{'clean_acc':>10}{'atk_acc':>9}{'ASR':>7}{'abstain':>9}"
        f"{'poison_used':>13}{'TPR':>7}{'FPR':>7}{'p95_ms':>9}{'alerts':>8}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        s = result.summary
        lines.append(
            f"{result.attack:<18}{_fmt(s.clean_accuracy):>10}{_fmt(s.attacked_accuracy):>9}"
            f"{_fmt(s.attack_success_rate):>7}{_fmt(s.abstention_rate):>9}"
            f"{_fmt(s.poison_used_rate):>13}{_fmt(s.filter_tpr):>7}{_fmt(s.filter_fpr):>7}"
            f"{s.p95_latency_ms:>9.1f}{len(result.alerts):>8}"
        )
    return "\n".join(lines)
