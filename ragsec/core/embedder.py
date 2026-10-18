"""Embedding backends.

:class:`SentenceTransformerEmbedder` is the default for experiments. :class:`HashingEmbedder`
is a dependency-free, deterministic feature-hashing embedder used for tests, CI and offline runs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ST_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

#: ``{(model_name, revision): {text: vector}}``. Shared across experiment cells without mixing
#: embeddings produced by different immutable model revisions.
_EMBED_CACHE: dict[tuple[str, str | None], dict[str, np.ndarray]] = {}
_MODEL_CACHE: dict[tuple[str, str | None, str | None], object] = {}


def _load_st_model(
    model_name: str, revision: str | None, device: str | None
):  # noqa: ANN202 - third-party type
    """Load (and memoise) a sentence-transformers model; loading dominates short runs."""
    key = (model_name, revision, device)
    if key not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[key] = SentenceTransformer(model_name, revision=revision, device=device)
    return _MODEL_CACHE[key]


def clear_embedding_cache() -> None:
    _EMBED_CACHE.clear()


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row; zero rows are left as zeros."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class BaseEmbedder(ABC):
    """Maps text to unit-norm float32 vectors so inner product equals cosine similarity."""

    dim: int
    name: str = "base"

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape ``(len(texts), dim)``."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(BaseEmbedder):
    """Deterministic feature-hashing embedder over word unigrams and bigrams."""

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    @staticmethod
    def _features(text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        return tokens + bigrams

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                sign = 1.0 if value & (1 << 63) else -1.0
                out[row, value % self.dim] += sign
        return normalize_rows(out)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wrapper around ``sentence-transformers`` models (optional dependency)."""

    name = "sentence-transformers"

    def __init__(
        self,
        model_name: str = DEFAULT_ST_MODEL,
        revision: str | None = None,
        device: str | None = None,
        cache: bool = True,
        batch_size: int = 64,
    ) -> None:

        if revision is None and model_name == DEFAULT_ST_MODEL:
            revision = DEFAULT_ST_MODEL_REVISION
        elif revision is None and not Path(model_name).exists():
            raise ValueError(
                "revision is required for a SentenceTransformers Hub model; "
                "pin an immutable commit or pass a local model path"
            )
        self.model_name = model_name
        self.revision = revision
        self.cache = cache
        self.batch_size = batch_size
        self._model = _load_st_model(model_name, revision, device)
        # sentence-transformers renamed this accessor; support both so the framework runs
        # against the installed version without emitting a deprecation warning per process.
        dimension = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim = int(dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self.cache:
            return self._encode(texts)
        # Sweeps rebuild the pipeline once per (configuration, attack, seed) cell over the same
        # corpus, so a process-level cache removes almost all repeated encoder work.
        store = _EMBED_CACHE.setdefault((self.model_name, self.revision), {})
        missing = [t for t in dict.fromkeys(texts) if t not in store]
        if missing:
            for text, vector in zip(missing, self._encode(missing), strict=True):
                store[text] = vector
        return np.ascontiguousarray(np.stack([store[t] for t in texts]))

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(name: str | None = None, **kwargs: object) -> BaseEmbedder:
    """Factory honouring the ``RAGSEC_EMBEDDER`` environment variable.

    ``"auto"`` (the default, and what the shipped configs request) tries sentence-transformers
    and falls back to hashing when the dependency or the model download is unavailable, so the
    framework always runs offline. ``RAGSEC_EMBEDDER`` overrides ``auto`` but not an explicitly
    named backend: setting it to ``hashing`` is how tests and CI pin the deterministic offline
    embedder without having to rewrite every config, while a config that names
    ``sentence-transformers`` outright still means it.
    """
    resolved = (name or "auto").lower()
    override = os.environ.get("RAGSEC_EMBEDDER")
    if resolved == "auto" and override:
        resolved = override.lower()
    if resolved == "hashing":
        return HashingEmbedder(**kwargs)  # type: ignore[arg-type]
    if resolved in {"auto", "sentence-transformers", "st"}:
        try:
            return SentenceTransformerEmbedder(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - any import/download failure triggers fallback
            if resolved != "auto":
                raise
            logger.warning(
                "sentence-transformers unavailable (%s); falling back to HashingEmbedder", exc
            )
            return HashingEmbedder()
    raise ValueError(f"Unknown embedder: {resolved!r}")
