#!/usr/bin/env python3
"""Export compact, query-text-free results used to rebuild the paper report."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from ragsec.eval.metrics import token_f1

QUERY_FIELDS = (
    "record_type",
    "study",
    "dataset",
    "config",
    "model",
    "seed",
    "attack",
    "attacked",
    "attack_success",
    "baseline_correct",
    "target_token_f1",
)


def export_query_outcomes(
    source: Path,
    destination: Path,
    additional_sources: tuple[Path, ...] = (),
) -> int:
    roots = dict.fromkeys(path.resolve() for path in (source, *additional_sources))
    sidecars = sorted({
        path
        for root in roots
        for directory in root.glob("*_query_records")
        for path in directory.glob("*.jsonl")
    })
    output = destination / "query_outcomes.jsonl.gz"
    questions_by_dataset: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in sidecars:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row: dict[str, Any] = json.loads(line)
                question = row.get("question")
                if row.get("record_type") == "query_outcome" and question:
                    group = (str(row.get("dataset", "")), str(row.get("model", "")))
                    questions_by_dataset[group].add(str(question))
    cluster_order = {
        group: {question: index for index, question in enumerate(sorted(questions))}
        for group, questions in questions_by_dataset.items()
    }
    count = 0
    with output.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as compressed:
        for path in sidecars:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row: dict[str, Any] = json.loads(line)
                    if row.get("record_type") != "query_outcome":
                        continue
                    question = row.get("question")
                    if not question:
                        continue
                    public_row = {field: row.get(field) for field in QUERY_FIELDS}
                    if row.get("target_answer") is not None and row.get("answer") is not None:
                        public_row["target_token_f1"] = token_f1(
                            str(row["target_answer"]), str(row["answer"])
                        )
                    public_row["question_cluster_id"] = hashlib.sha256(
                        str(question).encode("utf-8")
                    ).hexdigest()
                    group = (str(row.get("dataset", "")), str(row.get("model", "")))
                    public_row["question_cluster_order"] = cluster_order[
                        group
                    ][str(question)]
                    compressed.write(
                        (json.dumps(public_row, sort_keys=True, separators=(",", ":")) + "\n")
                        .encode("utf-8")
                    )
                    count += 1
    return count


def source_manifest(source: Path) -> dict[str, Any]:
    result_identities: dict[str, set[str]] = defaultdict(set)
    for result_path in source.glob("*.jsonl"):
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("code_fingerprint", "runtime_fingerprint"):
                    if row.get(key):
                        result_identities[key].add(str(row[key]))
    result_identity = {
        key: next(iter(values))
        for key, values in result_identities.items()
        if len(values) == 1
    }

    # Prefer a manifest inside the exact results directory. A parent-level manifest may
    # describe a previous run when several named sweeps share the same results/ directory.
    candidates = [source / "run_manifest.json", source.parent / "paper-rerun-manifest.json"]
    raw = None
    for manifest_path in candidates:
        if not manifest_path.exists():
            continue
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = candidate.get("experiment_identity", {})
        if any(
            result_identity.get(key) != identity.get(key)
            for key in ("code_fingerprint", "runtime_fingerprint")
            if key in result_identity and key in identity
        ):
            continue
        raw = candidate
        break
    if raw is None:
        # Result rows remain the authoritative source for these identities even when no
        # matching run manifest was saved alongside them.
        return {"experiment_identity": result_identity} if result_identity else {}
    environment = raw.get("environment", {})
    identity = raw.get("experiment_identity", {})
    local = raw.get("local_real_model_confirmation", {})
    return {
        "run_metadata": {
            key: raw[key]
            for key in ("created_utc", "command")
            if key in raw
        },
        "experiment_identity": {
            key: result_identity.get(key, identity.get(key))
            for key in ("code_fingerprint", "runtime_fingerprint")
            if result_identity.get(key, identity.get(key)) is not None
        },
        "environment": {
            key: environment[key]
            for key in (
                "python", "packages", "embedding_model", "embedding_model_hub_revision",
                "cuda_available",
            )
            if key in environment
        },
        "local_model_metadata": {
            "dataset": local.get("dataset"),
            "dataset_revision": local.get("dataset_revision"),
            "split": local.get("split"),
            "queries_per_config": local.get("queries_per_config"),
            "seed": local.get("seed"),
            "decoding": local.get("decoding"),
            "server": {
                key: local.get("server", {})[key]
                for key in ("implementation", "version")
                if key in local.get("server", {})
            },
            "models": [
                {
                    key: model[key]
                    for key in ("name", "upstream_model", "parameters", "quantization", "sha256")
                    if key in model
                }
                for model in local.get("models", [])
            ],
            "interpretation_limits": local.get("interpretation_limits", []),
        } if local else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        help="also copy aggregate local-model confirmation JSON/JSONL outputs",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    study_rows: dict[str, int] = {}
    dataset_pins: set[tuple[str, str, str, str, str]] = set()
    for path in sorted(args.results_dir.glob("*.jsonl")):
        shutil.copy2(path, args.output_dir / path.name)
        copied.append(path.name)
        with path.open("r", encoding="utf-8") as handle:
            count = 0
            for line in handle:
                if not line.strip():
                    continue
                row: dict[str, Any] = json.loads(line)
                count += 1
                dataset_pins.add((
                    str(row.get("dataset", "")),
                    str(row.get("dataset_source", "")),
                    str(row.get("dataset_config") or ""),
                    str(row.get("dataset_revision") or ""),
                    str(row.get("dataset_split") or ""),
                ))
            study_rows[path.stem] = count
    local_query_sources = (args.local_model_dir,) if args.local_model_dir else ()
    count = export_query_outcomes(args.results_dir, args.output_dir, local_query_sources)

    local_model_files: list[str] = []
    if args.local_model_dir:
        local_destination = args.output_dir / "local_model"
        local_destination.mkdir(exist_ok=True)
        for name in ("confirmation.jsonl", "summary.json", "confirmation-printed-summary.txt"):
            source = args.local_model_dir / name
            if source.exists():
                shutil.copy2(source, local_destination / name)
                local_model_files.append(f"local_model/{name}")

    manifest = {
        "format": "RAGSec compact paper artifact v1",
        "aggregate_result_files": copied,
        "study_rows": study_rows,
        "dataset_pins": [
            {
                "dataset": dataset,
                "source": source_name,
                "config": config or None,
                "revision": revision or None,
                "split": split or None,
            }
            for dataset, source_name, config, revision, split in sorted(dataset_pins)
        ],
        "query_outcomes": {
            "file": "query_outcomes.jsonl.gz",
            "records": count,
            "cluster_id": "SHA-256 of exact UTF-8 question text",
            "cluster_order": (
                "Zero-based order after sorting distinct exact question strings within dataset; "
                "retains the original bootstrap draw order."
            ),
            "included_fields": [*QUERY_FIELDS, "question_cluster_id", "question_cluster_order"],
            "excluded": [
                "question text", "gold answer", "attack target text", "generated answer", "latency"
            ],
        },
        "local_model_outputs": local_model_files,
        **source_manifest(args.results_dir),
    }
    local_summary_path = (
        args.local_model_dir / "summary.json" if args.local_model_dir else None
    )
    if local_summary_path is not None and local_summary_path.exists():
        local_summary = json.loads(local_summary_path.read_text(encoding="utf-8"))
        local_meta = local_summary.get("metadata", {})
        manifest["local_model_metadata"] = {
            "dataset": local_meta.get("dataset"),
            "dataset_revision": local_meta.get("dataset_revision"),
            "split": local_meta.get("split"),
            "queries_per_config": local_meta.get("queries_per_configuration"),
            "seed": local_meta.get("seed"),
            "code_fingerprint": local_meta.get("code_fingerprint"),
            "runtime_fingerprint": local_meta.get("runtime_fingerprint"),
            "embedding_revision": local_meta.get("embedding_revision"),
            "models": [{"name": name} for name in local_meta.get("models", [])],
            "decoding": local_meta.get("decoding", {}),
            "provenance": local_meta.get("source"),
            "interpretation_limits": [
                "The output is aggregate-only; no query-level paired intervals can be computed.",
                "The confirmation uses 100 fixed SQuAD questions and one seed per "
                "model/configuration.",
                "Model identity is recorded by the Ollama tag; model digest and server "
                "version were not included in the supplied summary.",
                "Cached decoding makes latency incomparable.",
            ],
        }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Copied {len(copied)} aggregate study files and {count} query outcomes.")
    print(f"Wrote {args.output_dir / 'artifact_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
