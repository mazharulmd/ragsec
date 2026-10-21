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
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
HOTPOTQA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"


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
    path: str = "rajpurkar/squad",
    split: str = "validation",
    limit: int = 200,
    source_id: str = "huggingface",
    revision: str | None = None,
) -> Dataset:
    """Load a SQuAD-style dataset, optionally pinned to a Hub commit revision."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the optional extra: pip install 'ragsec[datasets]'") from exc

    if revision is None:
        if path != "rajpurkar/squad":
            raise ValueError("revision is required for non-default Hugging Face datasets")
        revision = SQUAD_REVISION
    rows = hf_load_dataset(path, split=split, revision=revision)
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


def load_hotpotqa_dataset(
    config: str = "distractor",
    split: str = "validation",
    limit: int = 300,
    source_id: str = "hotpotqa",
    include_distractors: bool = True,
    skip_yes_no: bool = True,
    revision: str | None = None,
) -> Dataset:
    """Load HotpotQA as a document collection.

    Each question contributes its supporting paragraphs, and (by default) the distractor
    paragraphs shipped with the ``distractor`` configuration. The distractors matter here:
    they give the corpus realistic near-miss content, so a retrieval filter cannot obtain a
    low false-positive rate simply because every non-poison passage is topically unrelated.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the optional extra: pip install 'ragsec[datasets]'") from exc

    rows = hf_load_dataset(
        "hotpotqa/hotpot_qa",
        config,
        split=split,
        revision=revision or HOTPOTQA_REVISION,
    )
    records: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    n_questions = 0
    for row in rows:
        if n_questions >= limit:
            break
        answer = str(row["answer"]).strip()
        # Yes/no answers are unusable under the lenient substring criterion: "yes" occurs
        # inside unrelated text, and no decoy answer can be constructed for them.
        if skip_yes_no and answer.lower() in {"yes", "no"}:
            continue
        if len(answer) < 3:
            continue
        supporting = {str(t) for t in row["supporting_facts"]["title"]}
        titles = list(row["context"]["title"])
        sentences = list(row["context"]["sentences"])
        gold_doc_id: str | None = None
        pending: list[dict[str, Any]] = []
        for title, sents in zip(titles, sentences, strict=False):
            is_support = title in supporting
            if not include_distractors and not is_support:
                continue
            if title in seen_titles:
                if is_support and gold_doc_id is None:
                    gold_doc_id = f"hotpot-{title}"
                continue
            text = " ".join(str(x).strip() for x in sents).strip()
            if not text:
                continue
            seen_titles.add(title)
            doc_id = f"hotpot-{title}"
            if is_support and gold_doc_id is None:
                gold_doc_id = doc_id
            pending.append({"id": doc_id, "title": title, "text": text})
        if gold_doc_id is None or not answer:
            continue
        # Attach the question to whichever supporting paragraph is being emitted (or, when it
        # was already emitted for an earlier question, to the first new paragraph) so that
        # every QA pair has an answerable document in the index.
        for record in pending:
            if record["id"] == gold_doc_id:
                record["question"] = row["question"]
                record["answer"] = answer
                break
        else:
            records.append({"id": f"{gold_doc_id}#q{n_questions}", "title": "", "text": "",
                            "question": row["question"], "answer": answer})
        records.extend(pending)
        n_questions += 1
    records = [r for r in records if r["text"] or r.get("question")]
    return _from_records("hotpotqa", records, source_id)


def load_dataset(spec: dict[str, Any] | None = None) -> Dataset:
    """Dispatch on ``spec["name"]``: ``toy`` (default), ``json``, ``hf`` or ``hotpotqa``."""
    spec = dict(spec or {"name": "toy"})
    name = spec.pop("name", "toy")
    if name == "toy":
        return load_toy_dataset()
    if name == "json":
        return load_json_dataset(**spec)
    if name == "hf":
        return load_hf_dataset(**spec)
    if name in {"hotpot", "hotpotqa"}:
        return load_hotpotqa_dataset(**spec)
    raise ValueError(
        f"Unknown dataset {name!r}; expected 'toy', 'json', 'hf' or 'hotpotqa'"
    )
