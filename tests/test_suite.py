"""Tests for revision-aware study identities and resumable outputs."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import ragsec.eval.suite as suite
from ragsec.core.embedder import (
    _EMBED_CACHE,
    _MODEL_CACHE,
    DEFAULT_ST_MODEL_REVISION,
    SentenceTransformerEmbedder,
)
from ragsec.eval.runner import ExperimentConfig
from ragsec.eval.suite import Cell, cell_id, run_study


def _cell() -> Cell:
    return Cell(
        study="test",
        key={"dataset": "toy", "seed": 0},
        config=ExperimentConfig(
            dataset={"name": "toy", "revision": "toy-v1"},
            embedder={"name": "hashing", "dim": 32},
            pipeline={"top_k": 3},
        ),
    )


def test_cell_id_includes_complete_experiment_configuration() -> None:
    first = _cell()
    second = _cell()
    second.config.pipeline["top_k"] = 5
    assert cell_id(first) != cell_id(second)


def test_resume_rejects_stale_cell_and_no_resume_replaces_it(tmp_path, monkeypatch) -> None:
    cell = _cell()
    output = tmp_path / "study.jsonl"
    calls = {"count": 0}

    def fake_run_cell(current, query_path=None):
        calls["count"] += 1
        return [{"study": current.study, **current.key, "run": calls["count"]}]

    monkeypatch.setattr(suite, "run_cell", fake_run_cell)
    run_study([cell], output, resume=True, verbose=False)
    cell.config.pipeline["top_k"] = 5

    with pytest.raises(ValueError, match="another config or runtime"):
        run_study([cell], output, resume=True, verbose=False)

    run_study([cell], output, resume=False, verbose=False)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["run"] == 2
    assert rows[0]["_cell"] == cell_id(cell)


def test_sentence_transformer_cache_is_separated_by_revision(monkeypatch) -> None:
    calls: list[tuple[str, str | None, str | None]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, revision=None, device=None):
            calls.append((model_name, revision, device))

        @staticmethod
        def get_sentence_embedding_dimension() -> int:
            return 4

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    _MODEL_CACHE.clear()
    _EMBED_CACHE.clear()
    first = SentenceTransformerEmbedder("test/model", revision="rev-one")
    second = SentenceTransformerEmbedder("test/model", revision="rev-two")

    assert first.revision == "rev-one"
    assert second.revision == "rev-two"
    assert calls == [
        ("test/model", "rev-one", None),
        ("test/model", "rev-two", None),
    ]


def test_default_embedding_model_uses_recorded_hub_revision(monkeypatch) -> None:
    calls: list[str | None] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, revision=None, device=None):
            calls.append(revision)

        @staticmethod
        def get_sentence_embedding_dimension() -> int:
            return 4

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    _MODEL_CACHE.clear()
    model = SentenceTransformerEmbedder()
    assert model.revision == DEFAULT_ST_MODEL_REVISION
    assert calls == [DEFAULT_ST_MODEL_REVISION]


def test_custom_hub_model_requires_explicit_revision() -> None:
    with pytest.raises(ValueError, match="revision is required"):
        SentenceTransformerEmbedder("organization/custom-model")
