"""The experiment suite reported in the paper.

Each study below answers one question and writes one JSONL file under ``results/paper-pinned``.
The main study also saves per-question outcomes in a revision-keyed sidecar directory.

===========================  ==================================================================
``main``                     How much does each layer, and the full stack, reduce attack success
                             on each dataset? Repeated over seeds to measure seed variability.
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

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ragsec.core.embedder import DEFAULT_ST_MODEL_REVISION
from ragsec.eval.data import HOTPOTQA_REVISION, SQUAD_REVISION
from ragsec.eval.metrics import rate_with_ci
from ragsec.eval.runner import CLEAN_RUN, ExperimentConfig, ExperimentRunner

# ---------------------------------------------------------------------------
# Shared experiment ingredients
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    "squad": {
        "name": "hf",
        "path": "rajpurkar/squad",
        "split": "validation",
        "revision": SQUAD_REVISION,
        "limit": 300,
    },
    "hotpotqa": {
        "name": "hotpotqa",
        "config": "distractor",
        "split": "validation",
        "revision": HOTPOTQA_REVISION,
        "limit": 200,
    },
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

#: Models for the real-generator confirmation study. Every other study uses the mock
#: generator, which is what makes the 612-cell factorial affordable and deterministic; this
#: study exists to show the headline result is not an artefact of that instrument.
CONFIRMATION_MODELS: list[str] = ["gpt-4o-mini", "gpt-4o"]

#: Queries per corpus in the confirmation study. Small on purpose: the point is to confirm
#: the direction and rough magnitude of the main result on a real generator, not to repeat
#: the full factorial at API prices.
CONFIRMATION_QUERIES = 100

#: Confirmation ablations expose each component's clean-utility cost on real generators.
CONFIRMATION_CONFIGS = ("none", "L1", "L2", "L3", "L4", "full", "full_signed")

#: Where completions are cached, so an interrupted run resumes without paying twice.
CONFIRMATION_CACHE = "results/paper/llm-cache.sqlite"

#: Endpoint for the confirmation study. Any OpenAI-compatible ``/chat/completions`` service
#: works, so the study can run against a hosted provider with a free tier, or against a local
#: server such as Ollama, without touching the code. Only the base URL, the model names and
#: the environment variable holding the key change.
CONFIRMATION_BASE_URL = "https://api.openai.com/v1"
CONFIRMATION_KEY_ENV = "OPENAI_API_KEY"

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
    "full_signed": "Full defense + HMAC-authenticated ingestion",
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
        "params": {**l1_params, "require_signature": True},
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
    embedder_spec: dict[str, Any] = {"name": embedder}
    if embedder in {"sentence-transformers", "st"}:
        embedder_spec["revision"] = DEFAULT_ST_MODEL_REVISION
    return ExperimentConfig(
        name=f"{dataset}-{config}-s{seed}",
        dataset=DATASETS[dataset],
        embedder=embedder_spec,
        llm={"name": "mock", "follow_injections": follow_injections},
        pipeline={"top_k": top_k, "candidate_multiplier": candidate_multiplier},
        attacks=attacks,
        defenses=defense_stack(config, dataset),
        seed=seed,
        output_dir=None,
        include_clean_run=include_clean_run,
        sign_corpus=config in SIGNED_CONFIGS,
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
    dataset_spec = cell.config.dataset
    embedder_spec = result.config.get("resolved_embedder", cell.config.embedder)
    dataset_name = dataset_spec.get("name", "toy")
    dataset_source = dataset_spec.get("path") or (
        "hotpotqa/hotpot_qa" if dataset_name in {"hotpot", "hotpotqa"} else dataset_name
    )
    return {
        "study": cell.study,
        **cell.key,
        **cell.extra,
        "dataset_source": dataset_source,
        "dataset_config": dataset_spec.get("config"),
        "dataset_revision": dataset_spec.get("revision"),
        "dataset_split": dataset_spec.get("split"),
        "embedding_backend": embedder_spec.get("backend", embedder_spec.get("name")),
        "embedding_model": embedder_spec.get("model_name"),
        "embedding_revision": embedder_spec.get("revision"),
        "config_fingerprint": config_fingerprint(cell.config),
        "code_fingerprint": code_fingerprint(),
        "runtime_fingerprint": runtime_fingerprint(),
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


def query_records_path(out_path: Path, identifier: str) -> Path:
    """Stable sidecar path for a cell's per-question outcomes."""
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return out_path.parent / f"{out_path.stem}_query_records" / f"{digest}.jsonl"


def run_cell(cell: Cell, query_path: Path | None = None) -> list[dict[str, Any]]:
    start = time.perf_counter()
    results = ExperimentRunner(cell.config).run()
    wall = time.perf_counter() - start
    if query_path is not None:
        dataset_spec = cell.config.dataset
        dataset_name = dataset_spec.get("name", "toy")
        dataset_source = dataset_spec.get("path") or (
            "hotpotqa/hotpot_qa" if dataset_name in {"hotpot", "hotpotqa"} else dataset_name
        )
        query_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = query_path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            for result in results:
                metadata = {
                    "record_type": "query_outcome",
                    "study": cell.study,
                    **cell.key,
                    **cell.extra,
                    "dataset_source": dataset_source,
                    "dataset_config": dataset_spec.get("config"),
                    "dataset_revision": dataset_spec.get("revision"),
                    "dataset_split": dataset_spec.get("split"),
                    "config_fingerprint": config_fingerprint(cell.config),
                    "code_fingerprint": code_fingerprint(),
                    "runtime_fingerprint": runtime_fingerprint(),
                    "attack": result.attack,
                    "run_name": result.name,
                    "_cell": cell_id(cell),
                    "_cell_label": cell_label(cell),
                }
                for record in result.records:
                    outcome = record.to_dict()
                    outcome.pop("detections", None)
                    handle.write(json.dumps({**metadata, **outcome}) + "\n")
        temporary_path.replace(query_path)
    return [_row(cell, result, wall / max(1, len(results))) for result in results]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_fingerprint(config: ExperimentConfig) -> str:
    """Hash the full resolved configuration without exposing credential fields in results."""
    return hashlib.sha256(_canonical_json(config.to_dict()).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Hash package sources and project metadata, including uncommitted working-tree edits."""
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    sources = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json"}
    )
    sources.append(project_root / "pyproject.toml")
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def runtime_fingerprint() -> str:
    """Fingerprint the interpreter and packages that affect data, embedding or retrieval."""
    packages = (
        "numpy",
        "datasets",
        "sentence-transformers",
        "transformers",
        "torch",
        "faiss-cpu",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        "python": sys.version,
        "packages": versions,
        "environment": {
            name: os.environ.get(name)
            for name in ("RAGSEC_EMBEDDER", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES")
        },
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def cell_label(cell: Cell) -> str:
    """Stable label for detecting stale versions of a logical experiment cell."""
    return _canonical_json(
        {"study": cell.study, "key": cell.key, "extra": cell.extra}
    )


def cell_id(cell: Cell) -> str:
    """Identity of a cell's complete configuration, implementation and runtime."""
    payload = {
        "label": cell_label(cell),
        "config": cell.config.to_dict(),
        "config_fingerprint": config_fingerprint(cell.config),
        "code_fingerprint": code_fingerprint(),
        "runtime_fingerprint": runtime_fingerprint(),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def run_study(
    cells: list[Cell],
    out_path: Path,
    resume: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run cells and append summaries; main-study query outcomes are saved per cell."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    rows: list[dict[str, Any]] = []
    if not resume:
        out_path.unlink(missing_ok=True)
        shutil.rmtree(out_path.parent / f"{out_path.stem}_query_records", ignore_errors=True)
    if resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row)
                    done.add(row["_cell"])
    for cell in cells:
        label = cell_label(cell)
        identifier = cell_id(cell)
        stale = any(
            row.get("_cell") != identifier
            and (
                row.get("_cell_label") == label
                or (
                    row.get("_cell_label") is None
                    and row.get("study") == cell.study
                    and all(row.get(key) == value for key, value in cell.key.items())
                    and all(row.get(key) == value for key, value in cell.extra.items())
                )
            )
            for row in rows
        )
        if stale:
            raise ValueError(
                f"Existing results for {cell.study} cell {label} use another config or runtime; "
                "choose a fresh output directory or rerun with resume=False"
            )
    mode = "a" if resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for index, cell in enumerate(cells, 1):
            identifier = cell_id(cell)
            query_path = (
                query_records_path(out_path, identifier) if cell.study == "main" else None
            )
            if identifier in done and (query_path is None or query_path.exists()):
                continue
            started = time.perf_counter()
            cell_rows = run_cell(cell, query_path=query_path)
            if identifier not in done:
                for row in cell_rows:
                    row["_cell"] = identifier
                    row["_cell_label"] = cell_label(cell)
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
                config.sign_corpus = bool(options.get("signed"))
                yield Cell(
                    study="provenance",
                    key={"dataset": dataset, "variant": variant, "seed": seed},
                    config=config,
                )


def study_confirmation(
    datasets: list[str],
    seeds: list[int],
    configs: list[str] | None = None,
) -> Iterator[Cell]:
    """Repeat the headline comparison on real language models.

    Every other study uses the deterministic mock generator. That is a deliberate instrument:
    it fixes generator behaviour exactly, so a change in measured attack success is
    attributable to the retrieval pipeline rather than to sampling. It is also the obvious
    objection to the work, because the mock obeys injected instructions by construction and
    believes whatever it is given. This study answers the objection on a reduced grid:
    the configurations that matter, the attacks that separate them, one seed, and a corpus
    small enough to run against a paid endpoint.

    The adaptive attack succeeds because the generator believes a corroborated document. A
    real model may resist that, so this is the experiment most able to overturn the paper's
    central claim.
    """
    configs = tuple(CONFIRMATION_CONFIGS if configs is None else configs)
    attacks = [
        {"name": "poisoned_rag", "n_per_target": 1},
        {"name": "prompt_injection", "n_per_target": 1},
        {"name": "adaptive_poison", "n_per_target": 3},
    ]
    for dataset in datasets:
        spec = dict(DATASETS[dataset])
        if "limit" in spec or spec.get("name") == "hf":
            spec["limit"] = CONFIRMATION_QUERIES
        for model in CONFIRMATION_MODELS:
            for config in configs:
                for seed in seeds:
                    cfg = base_config(dataset, config, seed, attacks=attacks)
                    cfg.dataset = spec
                    cfg.llm = {
                        "name": "openai",
                        "model": model,
                        "base_url": CONFIRMATION_BASE_URL,
                        "api_key_env": CONFIRMATION_KEY_ENV,
                        "temperature": 0.0,
                        "max_tokens": 128,
                        "cache_path": CONFIRMATION_CACHE,
                    }
                    yield Cell(
                        study="confirmation",
                        key={
                            "dataset": dataset,
                            "model": model,
                            "config": config,
                            "seed": seed,
                        },
                        config=cfg,
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
    "confirmation": study_confirmation,
}

#: Studies that call a paid endpoint. Excluded from a bare `ragsec suite` so that the
#: default run stays free and offline-reproducible.
PAID_STUDIES = {"confirmation"}


def run_suite(
    studies: list[str] | None = None,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    output_dir: str | Path = "results/paper",
    resume: bool = True,
    models: list[str] | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    queries: int | None = None,
    confirmation_configs: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run the requested studies and return ``{study: rows}``.

    Studies in :data:`PAID_STUDIES` call a metered endpoint and are therefore opt-in: a bare
    call runs everything else, and the paid ones only when named explicitly.
    """
    global CONFIRMATION_MODELS, CONFIRMATION_BASE_URL, CONFIRMATION_KEY_ENV
    global CONFIRMATION_QUERIES
    if models:
        CONFIRMATION_MODELS = list(models)
    if queries:
        CONFIRMATION_QUERIES = queries
    if confirmation_configs is not None:
        valid_configs = set(CONFIRMATION_CONFIGS)
        unknown_configs = set(confirmation_configs) - valid_configs
        if unknown_configs:
            raise ValueError(
                f"Unknown confirmation configurations {sorted(unknown_configs)}; "
                f"choose from {sorted(valid_configs)}"
            )
        if not confirmation_configs:
            raise ValueError("confirmation_configs must contain at least one configuration")
        confirmation_configs = list(dict.fromkeys(confirmation_configs))
    if base_url:
        CONFIRMATION_BASE_URL = base_url
    if api_key_env:
        CONFIRMATION_KEY_ENV = api_key_env
    studies = studies or [s for s in STUDIES if s not in PAID_STUDIES]
    datasets = datasets or ["squad", "hotpotqa", "toy"]
    seeds = seeds or [0, 1, 2]
    directory = Path(output_dir)
    out: dict[str, list[dict[str, Any]]] = {}
    for study in studies:
        if study not in STUDIES:
            raise ValueError(f"Unknown study {study!r}; choose from {sorted(STUDIES)}")
        if study == "confirmation":
            cells = list(study_confirmation(datasets, seeds, confirmation_configs))
        else:
            cells = list(STUDIES[study](datasets, seeds))
        print(f"\n=== {study}: {len(cells)} cells ===", flush=True)
        out[study] = run_study(cells, directory / f"{study}.jsonl", resume=resume)
    return out
