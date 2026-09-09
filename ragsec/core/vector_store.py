"""In-memory vector store backed by FAISS when available, NumPy otherwise."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ragsec.core.models import Chunk

try:
    import faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover - exercised only without faiss installed
    faiss = None
    _HAS_FAISS = False


class VectorStore:
    """Flat inner-product index over unit-norm embeddings with aligned chunk metadata."""

    def __init__(self, dim: int, use_faiss: bool | None = None) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.use_faiss = _HAS_FAISS if use_faiss is None else bool(use_faiss and _HAS_FAISS)
        self._chunks: list[Chunk] = []
        self._embeddings = np.zeros((0, dim), dtype=np.float32)
        self._index = faiss.IndexFlatIP(dim) if self.use_faiss else None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    def get(self, index: int) -> Chunk:
        return self._chunks[index]

    def embedding_of(self, index: int) -> np.ndarray:
        return self._embeddings[index]

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape != (len(chunks), self.dim):
            raise ValueError(
                f"embeddings must have shape ({len(chunks)}, {self.dim}), got {embeddings.shape}"
            )
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._embeddings = np.ascontiguousarray(np.vstack([self._embeddings, embeddings]))
        if self._index is not None:
            self._index.add(embeddings)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return ``[(index, score), ...]`` for the ``k`` most similar chunks, best first."""
        if len(self) == 0 or k <= 0:
            return []
        k = min(k, len(self))
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if self._index is not None:
            scores, indices = self._index.search(q, k)
            return [
                (int(i), float(s))
                for i, s in zip(indices[0], scores[0], strict=False)
                if i >= 0
            ]
        scores = self._embeddings @ q[0]
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top]

    def centroid(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return self._embeddings.mean(axis=0).astype(np.float32)

    def remove_by_doc_ids(self, doc_ids: Iterable[str]) -> int:
        """Remove all chunks belonging to ``doc_ids``; returns the number removed."""
        targets = set(doc_ids)
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.doc_id not in targets]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0
        self._chunks = [self._chunks[i] for i in keep]
        self._embeddings = (
            np.ascontiguousarray(self._embeddings[keep])
            if keep
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        if self._index is not None:
            self._index.reset()
            if len(self._chunks):
                self._index.add(self._embeddings)
        return removed

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = np.zeros((0, self.dim), dtype=np.float32)
        if self._index is not None:
            self._index.reset()
