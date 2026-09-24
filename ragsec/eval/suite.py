"""The experiment suite reported in the paper.

Each study below answers one question and writes one JSONL file under ``results/paper``:

===========================  ==================================================================
``main``                     How much does each layer, and the full stack, reduce attack success
                             on each dataset? Repeated over seeds so that rates carry intervals.
``evasion``                  Which evasion technique is responsible for the adaptive attacker's
                             success? One technique is disabled per cell.
``budget``                   How does attack success scale with the adversary's upload budget?
``threshold``                What does moving the retrieval filter's operating point cost in
                             false positives and clean accuracy?
``backfill``                 How much over-retrieval is needed before filtering actually helps,
                             and what does it cost in latency?
``backfill_strict``          The same sweep with the filter at an aggressive threshold, which is
                             the only regime where backfill can matter.
``provenance``               Do allowlists and trust scores survive a provenance-spoofing
                             adversary, and does HMAC signing recover?
``generator``                How much of the measured risk comes from an injection-susceptible
                             generator rather than from retrieval?
===========================  ==================================================================

Cells are keyed and appended incrementally, so an interrupted sweep resumes where it stopped
and never recomputes a finished cell.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragsec.eval.metrics import rate_with_ci
from ragsec.eval.runner import CLEAN_RUN, ExperimentConfig, ExperimentRunner

# ---------------------------------------------------------------------------
# Shared experiment ingredients
# ---------------------------------------------------------------------------

#: Key held by the ingestion pipeline of the simulated deployment. The adversary does not have
#: it; every other property of a trusted document (source name, trust score) they can forge.
SIGNING_KEY = "ragsec-corpus-signing-key-v1"

DATASETS: dict[str, dict[str, Any]] = {
    "squad": {"name": "hf", "path": "rajpurkar/squad", "limit": 300},
    "hotpotqa": {"name": "hotpotqa", "limit": 200},
    "toy": {"name": "toy"},
}

#: Source identifier each dataset publishes under, needed for allowlists and trust maps.
DATASET_SOURCE: dict[str, str] = {
    "squad": "huggingface",
    "hotpotqa": "hotpotqa",
    "toy": "curated-encyclopedia",
}

ATTACKS: list[dict[str, Any]] = [
    {"name": "poisoned_rag", "n_per_target": 1},
    {"name": "trigger_backdoor", "n_per_target": 1},
    {"name": "prompt_injection", "n_per_target": 1},
    {"name": "context_flooding", "n_per_target": 5},
    {"name": "adaptive_poison", "n_per_target": 3},
]

ATTACK_LABELS = {
    "poisoned_rag": "PoisonedRAG",
    "trigger_backdoor": "Trigger backdoor",
    "prompt_injection": "Prompt injection",
    "context_flooding": "Context flooding",
    "adaptive_poison": "Adaptive poisoning",
    CLEAN_RUN: "Clean",
}

CONFIG_LABELS = {
    "none": "No defense",
    "L1": "L1 source verification",
    "L2": "L2 retrieval filtering",
    "L3": "L3 robust aggregation",
    "L4": "L4 monitoring",
    "full": "Full defense",
    "full_signed": "Full defense + signed ingestion",
}

EVASION_LABELS = {
    "paraphrase": "Query paraphrase",
    "no_directives": "Assertive (non-imperative) payload",
    "natural_padding": "Natural prose padding",
    "dilute_similarity": "Similarity dilution",
    "spoof_source": "Provenance spoofing",
    "diversify": "Corroboration across documents",
}


def defense_stack(name: str, dataset: str) -> list[dict[str, Any]]:
    """Build the defense list for a named configuration on a given dataset."""
    source = DATASET_SOURCE[dataset]
    l1_params: dict[str, Any] = {
        "allowed_sources": [source],
        "source_trust": {source: 0.8},
        "min_trust": 0.6,
        "max_documents_per_source": 100_000,
    }
    l1 = {"name": "source_verification", "params": l1_params}
    l1_signed = {
        "name": "source_verification",
        "params": {**l1_params, "hmac_key": SIGNING_KEY, "require_signature": True},
    }
    l2 = {"name": "retrieval_filter", "params": {"mode": "drop"}}
    l3 = {"name": "robust_aggregation", "params": {}}
    l4 = {"name": "monitoring", "params": {"window_size": 50, "drift_threshold": 0.3}}
    return {
        "none": [],
        "L1": [l1],
        "L2": [l2],
        "L3": [l3],
        "L4": [l4],
        "full": [l1, l2, l3, l4],
        "full_signed": [l1_signed, l2, l3, l4],
    }[name]


#: Layer 1 only enforces signatures when the legitimate corpus is actually signed.
SIGNED_CONFIGS = {"full_signed"}

#: Per-source quota used in the ingestion ladder, as a multiple of the legitimate corpus size.
#: A source is permitted to grow the corpus it already contributed by 50%; anything much
#: tighter would block ordinary curation, and anything looser stops bounding bulk insertion.
QUOTA_FACTOR = 1.5

_CORPUS_SIZES: dict[str, int] = {}


def corpus_size(dataset: str) -> int:
    """Number of legitimate documents in ``dataset`` (memoised across cells)."""
    if dataset not in _CORPUS_SIZES:
        from ragsec.eval.data import load_dataset

        _CORPUS_SIZES[dataset] = len(load_dataset(DATASETS[dataset]).documents)
    return _CORPUS_SIZES[dataset]


def base_config(
    dataset: str,
    config: str,
    seed: int,
    *,
    attacks: list[dict[str, Any]],
    top_k: int = 3,
    candidate_multiplier: int = 10,
    include_clean_run: bool = True,
    follow_injections: bool = True,
    embedder: str = "sentence-transformers",
) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"{dataset}-{config}-s{seed}",
        dataset=DATASETS[dataset],
        embedder={"name": embedder},
        llm={"name": "mock", "follow_injections": follow_injections},
        pipeline={"top_k": top_k, "candidate_multiplier": candidate_multiplier},
        attacks=attacks,
        defenses=defense_stack(config, dataset),
        seed=seed,
        output_dir=None,
        include_clean_run=include_clean_run,
        sign_corpus=SIGNING_KEY if config in SIGNED_CONFIGS else None,
    )


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """One experiment configuration plus the metadata used to key and label it."""

    study: str
    key: dict[str, Any]
    config: ExperimentConfig
    #: Extra fields copied verbatim into every emitted row.
    extra: dict[str, Any] = field(default_factory=dict)


def _row(cell: Cell, result: Any, wall_s: float) -> dict[str, Any]:
    summary = result.summary.to_dict()
    attacked = [r for r in result.records if r.attacked]
    asr_flags = [bool(r.attack_success) for r in attacked if r.attack_success is not None]
    asr, asr_low, asr_high = rate_with_ci(asr_flags)
    cond_flags = [
        bool(r.attack_success) for r in attacked
        if r.attack_success is not None and r.baseline_correct
    ]
    _, cond_low, cond_high = rate_with_ci(cond_flags)
    acc_flags = [r.correct for r in result.records if not r.attacked] or [
        r.correct for r in attacked
    ]
    _, acc_low, acc_high = rate_with_ci(acc_flags)
    return {
        "study": cell.study,
        **cell.key,
        **cell.extra,
        "attack": result.attack,
        "n_alerts": len(result.alerts),
        "alert_kinds": sorted({a["kind"] for a in result.alerts}),
        "asr_ci_low": asr_low if asr_flags else None,
        "asr_ci_high": asr_high if asr_flags else None,
        "casr_ci_low": cond_low if cond_flags else None,
        "casr_ci_high": cond_high if cond_flags else None,
        "acc_ci_low": acc_low,
        "acc_ci_high": acc_high,
        "poison_docs_accepted": (result.ingest.get("poison") or {}).get(
            "accepted_documents", 0
        ),
        "poison_docs_rejected": (result.ingest.get("poison") or {}).get(
            "rejected_documents", 0
        ),
        "clean_docs_accepted": result.ingest["clean"]["accepted_documents"],
        "wall_s": round(wall_s, 2),
        **summary,
    }


def run_cell(cell: Cell) -> list[dict[str, Any]]:
    start = time.perf_counter()
    results = ExperimentRunner(cell.config).run()
    wall = time.perf_counter() - start
    return [_row(cell, result, wall / max(1, len(results))) for result in results]


def cell_id(cell: Cell) -> str:
    return json.dumps({"study": cell.study, **cell.key, **cell.extra}, sort_keys=True)


def run_study(
    cells: list[Cell],
    out_path: Path,
    resume: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run every cell, appending rows to ``out_path`` (JSONL) as they complete."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    rows: list[dict[str, Any]] = []
    if resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row)
                    done.add(row["_cell"])
    with out_path.open("a", encoding="utf-8") as handle:
        for index, cell in enumerate(cells, 1):
            identifier = cell_id(cell)
            if identifier in done:
                continue
            started = time.perf_counter()
            for row in run_cell(cell):
                row["_cell"] = identifier
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            done.add(identifier)
            if verbose:
                print(
                    f"[{index:>4}/{len(cells)}] {cell.study:<11} "
                    f"{identifier[:110]:<112} {time.perf_counter() - started:6.1f}s",
                    flush=True,
                )
    return rows


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------


def study_main(datasets: list[str], seeds: list[int]) -> Iterator[Cell]:
    """Every defense configuration against every attack, on every dataset."""
    for dataset in datasets:
        for config in ("none", "L1", "L2", "L3", "L4", "full", "full_signed"):
            for seed in seeds:
                yield Cell(
                    study="main",
                    key={"dataset": dataset, "config": config, "seed": seed},
                    config=base_config(dataset, config, seed, attacks=ATTACKS),
                )


def study_evasion(datasets: list[str], seeds: list[int]) -> Iterator[Cell]:
    """Leave-one-out over the adaptive attacker's evasion techniques."""
    for dataset in datasets:
        for config in ("full", "full_signed"):
            for disabled in [None, *EVASION_LABELS]:
                params = {disabled: False} if disabled else {}
                for seed in seeds:
                    yield Cell(
                        study="evasion",
                        key={
                            "dataset": dataset,
                            "config": config,
                            "disabled": disabled or "none",
                            "seed": seed,
                        },
                        config=base_config(
                            dataset,
                            config,
                            seed,
                            include_clean_run=False,
                            attacks=[
                                {
                                    "name": "adaptive_poison",
                                    "n_per_target": 3,
                                    "params": params,
                                }
                            ],
                        ),
                    )


def study_budget(datasets: list[str], seeds: list[int]) -> Iterator[Cell]:
    """Attack success as a function of the number of poisoned documents per target."""
    for dataset in datasets:
        for config in ("none", "full", "full_signed"):
            for n in (1, 2, 3, 5, 10, 20):
                for seed in seeds:
                    yield Cell(
                        study="budget",
                        key={
                            "dataset": dataset,
                            "config": config,
                            "n_per_target": n,
                            "seed": seed,
                        },
                        config=base_config(
                            dataset,
                            config,
                            seed,
                            include_clean_run=False,
                            attacks=[
                                {"name": "poisoned_rag", "n_per_target": n},
                                {"name": "adaptive_poison", "n_per_target": n},
                            ],
                        ),
                    )


def study_threshold(datasets: list[str], seeds: list[int]) -> Iterator[Cell]:
    """Sweep the retrieval filter's operating point on the calibrated suspicion scale."""
    for dataset in datasets:
        for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
            for seed in seeds:
                config = base_config(dataset, "L2", seed, attacks=ATTACKS)
                config.defenses = [
                    {
                        "name": "retrieval_filter",
                        "params": {"mode": "drop", "threshold": threshold},
                    }
                ]
                yield Cell(
                    study="threshold",
                    key={"dataset": dataset, "threshold": threshold, "seed": seed},
                    config=config,
                )


def study_backfill(datasets: list[str], seeds: list[int]) -> Iterator[Cell]:
    """Security, utility and latency as a function of the over-retrieval multiplier."""
    for dataset in datasets:
        for multiplier in (1, 2, 3, 5, 10, 20):
            for seed in seeds:
                yield Cell(
                    study="backfill",
                    key={
                        "dataset": dataset,
                        "candidate_multiplier": multiplier,
                        "seed": seed,
                    },
                    config=base_config(
                        dataset,
                        "full",
                        seed,
                        attacks=ATTACKS,
                        candidate_multiplier=multiplier,
                    ),
                )


#: Suspicion threshold used by the strict-filtering backfill study. At the default operating
#: point the filter discards well under 1% of candidates, so over-retrieval has nothing to
#: replace; this threshold is taken from the threshold sweep as one that actually removes a
#: substantial share of the candidate set.
STRICT_THRESHOLD = 0.6


def study_backfill_strict(datasets: list[str], seeds: list[int]) -> Iterator[Cell]:
    """Over-retrieval sweep with the retrieval filter at an aggressive operating point.

    The companion ``backfill`` study sweeps the candidate multiplier with the filter at its
    default threshold, where it turns out to change nothing because almost no candidate is
    dropped. This study repeats the sweep at a threshold that discards a large fraction of
    candidates, which is the regime in which backfill is supposed to matter -- and therefore
    the only regime in which the claim that filtering needs over-retrieval can be tested.
    """
    for dataset in datasets:
        for multiplier in (1, 2, 3, 5, 10, 20):
            for seed in seeds:
                config = base_config(
                    dataset, "full", seed, attacks=ATTACKS,
                    candidate_multiplier=multiplier,
                )
                config.defenses = [
                    {
                        "name": "retrieval_filter",
                        "params": {"mode": "drop", "threshold": STRICT_THRESHOLD},
                    }
                ]
                yield Cell(
                    study="backfill_strict",
                    key={
                        "dataset": dataset,
                        "candidate_multiplier": multiplier,
                        "seed": seed,
                    },
                    config=config,
                )


def study_provenance(datasets: list[str], seeds: list[int]) -> Iterator[Cell]:
    """Ingestion controls of increasing strength against a provenance-spoofing adversary."""
    variants: dict[str, dict[str, Any]] = {
        "open": {},
        "trust_only": {"min_trust": 0.6},
        "allowlist": {"min_trust": 0.6, "allowlist": True},
        "quota": {"min_trust": 0.6, "allowlist": True, "quota_factor": QUOTA_FACTOR},
        "signed": {"min_trust": 0.6, "allowlist": True, "signed": True},
    }
    for dataset in datasets:
        source = DATASET_SOURCE[dataset]
        # The quota has to be expressed relative to the corpus it governs. A fixed document
        # count that happens to sit below the size of the legitimate corpus does not measure a
        # defense -- it silently truncates the knowledge base and reports the resulting failure
        # to answer as security.
        quota = round(QUOTA_FACTOR * corpus_size(dataset))
        for variant, options in variants.items():
            for seed in seeds:
                params: dict[str, Any] = {
                    "source_trust": {source: 0.8},
                    "min_trust": options.get("min_trust", 0.0),
                    "max_documents_per_source": (
                        quota if options.get("quota_factor") else 100_000
                    ),
                }
                if options.get("allowlist"):
                    params["allowed_sources"] = [source]
                if options.get("signed"):
                    params["hmac_key"] = SIGNING_KEY
                    params["require_signature"] = True
                config = base_config(
                    dataset,
                    "L1",
                    seed,
                    include_clean_run=False,
                    attacks=[
                        {"name": "poisoned_rag", "n_per_target": 3},
                        {"name": "adaptive_poison", "n_per_target": 3},
                    ],
                )
                config.defenses = [{"name": "source_verification", "params": params}]
                config.sign_corpus = SIGNING_KEY if options.get("signed") else None
                yield Cell(
                    study="provenance",
                    key={"dataset": dataset, "variant": variant, "seed": seed},
                    config=config,
                )


def study_generator(datasets: list[str], seeds: list[int]) -> Iterator[Cell]:
    """Separate the risk contributed by an injection-susceptible generator from retrieval."""
    for dataset in datasets:
        for config in ("none", "full"):
            for follows in (True, False):
                for seed in seeds:
                    yield Cell(
                        study="generator",
                        key={
                            "dataset": dataset,
                            "config": config,
                            "follow_injections": follows,
                            "seed": seed,
                        },
                        config=base_config(
                            dataset,
                            config,
                            seed,
                            attacks=ATTACKS,
                            follow_injections=follows,
                        ),
                    )


STUDIES = {
    "main": study_main,
    "evasion": study_evasion,
    "budget": study_budget,
    "threshold": study_threshold,
    "backfill": study_backfill,
    "backfill_strict": study_backfill_strict,
    "provenance": study_provenance,
    "generator": study_generator,
}


def run_suite(
    studies: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    output_dir: str | Path = "results/paper",
    resume: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Run the requested studies and return ``{study: rows}``."""
    studies = studies or list(STUDIES)
    datasets = datasets or ["squad", "hotpotqa", "toy"]
    seeds = seeds or [0, 1, 2]
    directory = Path(output_dir)
    out: dict[str, list[dict[str, Any]]] = {}
    for study in studies:
        if study not in STUDIES:
            raise ValueError(f"Unknown study {study!r}; choose from {sorted(STUDIES)}")
        cells = list(STUDIES[study](datasets, seeds))
        print(f"\n=== {study}: {len(cells)} cells ===", flush=True)
        out[study] = run_study(cells, directory / f"{study}.jsonl", resume=resume)
    return out
