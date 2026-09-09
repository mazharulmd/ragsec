"""Dense retriever combining an embedder and a vector store."""

from __future__ import annotations

import numpy as np

from ragsec.core.embedder import BaseEmbedder
from ragsec.core.models import RetrievedChunk
from ragsec.core.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: BaseEmbedder, store: VectorStore) -> None:
        if embedder.dim != store.dim:
            raise ValueError(
                f"embedder dim {embedder.dim} does not match store dim {store.dim}"
            )
        self.embedder = embedder
        self.store = store

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed_one(query)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        hits = self.store.search(self.embed_query(query), k)
        return [
            RetrievedChunk(chunk=self.store.get(index), score=score, rank=rank, index=index)
            for rank, (index, score) in enumerate(hits)
        ]
