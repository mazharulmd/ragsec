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
    "dataset",
    "config",
    "seed",
    "attack",
    "attacked",
    "attack_success",
    "baseline_correct",
    "target_token_f1",
)


def export_query_outcomes(source: Path, destination: Path) -> int:
    sidecars = source / "main_query_records"
    output = destination / "query_outcomes.jsonl.gz"
    questions_by_dataset: dict[str, set[str]] = defaultdict(set)
    for path in sorted(sidecars.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row: dict[str, Any] = json.loads(line)
                question = row.get("question")
                if row.get("record_type") == "query_outcome" and question:
                    questions_by_dataset[str(row.get("dataset", ""))].add(str(question))
    cluster_order = {
        dataset: {question: index for index, question in enumerate(sorted(questions))}
        for dataset, questions in questions_by_dataset.items()
    }
    count = 0
    with output.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as compressed:
        for path in sorted(sidecars.glob("*.jsonl")):
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
                    public_row["question_cluster_order"] = cluster_order[
                        str(row.get("dataset", ""))
                    ][str(question)]
                    compressed.write(
                        (json.dumps(public_row, sort_keys=True, separators=(",", ":")) + "\n")
                        .encode("utf-8")
                    )
                    count += 1
    return count


def source_manifest(source: Path) -> dict[str, Any]:
    candidates = [source.parent / "paper-rerun-manifest.json", source / "run_manifest.json"]
    manifest_path = next((path for path in candidates if path.exists()), None)
    if manifest_path is None:
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment = raw.get("environment", {})
    local = raw.get("local_real_model_confirmation", {})
    return {
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
    count = export_query_outcomes(args.results_dir, args.output_dir)

    local_model_files: list[str] = []
    if args.local_model_dir:
        local_destination = args.output_dir / "local_model"
        local_destination.mkdir(exist_ok=True)
        for name in ("confirmation.jsonl", "summary.json"):
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
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Copied {len(copied)} aggregate study files and {count} query outcomes.")
    print(f"Wrote {args.output_dir / 'artifact_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
