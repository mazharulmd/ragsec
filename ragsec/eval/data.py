"""Dataset loaders.

The bundled toy corpus runs fully offline. SQuAD-style HuggingFace datasets (``context`` /
``question`` / ``answers``) can be loaded when the optional ``datasets`` extra is installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from ragsec.core.models import Document, Provenance

TOY_SOURCE = "curated-encyclopedia"


@dataclass(frozen=True)
class QAExample:
    question: str
    answer: str
    doc_id: str | None = None


@dataclass
class Dataset:
    name: str
    documents: list[Document]
    qa: list[QAExample]


def _from_records(name: str, records: list[dict[str, Any]], source_id: str) -> Dataset:
    documents: list[Document] = []
    qa: list[QAExample] = []
    for index, record in enumerate(records):
        doc_id = str(record.get("id", f"{name}-{index:04d}"))
        documents.append(
            Document(
                text=record["text"],
                doc_id=doc_id,
                provenance=Provenance(
                    source_id=record.get("source_id", source_id),
                    trust_score=float(record.get("trust", 0.5)),
                ),
                metadata={"poison": False, "title": record.get("title", "")},
            )
        )
        if record.get("question") and record.get("answer"):
            qa.append(QAExample(record["question"], record["answer"], doc_id))
    return Dataset(name=name, documents=documents, qa=qa)


def load_toy_dataset() -> Dataset:
    """Thirty short factual passages, each paired with one question and gold answer."""
    path = resources.files("ragsec.eval.resources").joinpath("toy_corpus.json")
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return _from_records("toy", records, TOY_SOURCE)


def load_json_dataset(path: str | Path, source_id: str = "local-file") -> Dataset:
    """Load a JSON list of ``{id, title, text, question, answer}`` records."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return _from_records(path.stem, records, source_id)


def load_hf_dataset(
    path: str = "squad",
    split: str = "validation",
    limit: int = 200,
    source_id: str = "huggingface",
) -> Dataset:
    """Load a SQuAD-style dataset (``context``, ``question``, ``answers.text``)."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the optional extra: pip install 'ragsec[datasets]'") from exc

    rows = hf_load_dataset(path, split=split)
    records: list[dict[str, Any]] = []
    seen_contexts: dict[str, str] = {}
    for row in rows:
        if len(records) >= limit:
            break
        context = row["context"]
        answers = row["answers"]["text"] if isinstance(row["answers"], dict) else row["answers"]
        if not answers:
            continue
        if context in seen_contexts:
            continue
        doc_id = f"{path}-{len(records):05d}"
        seen_contexts[context] = doc_id
        records.append(
            {
                "id": doc_id,
                "title": row.get("title", ""),
                "text": context,
                "question": row["question"],
                "answer": answers[0],
            }
        )
    return _from_records(path, records, source_id)


def load_dataset(spec: dict[str, Any] | None = None) -> Dataset:
    """Dispatch on ``spec["name"]``: ``toy`` (default), ``json`` or ``hf``."""
    spec = dict(spec or {"name": "toy"})
    name = spec.pop("name", "toy")
    if name == "toy":
        return load_toy_dataset()
    if name == "json":
        return load_json_dataset(**spec)
    if name == "hf":
        return load_hf_dataset(**spec)
    raise ValueError(f"Unknown dataset {name!r}; expected 'toy', 'json' or 'hf'")
