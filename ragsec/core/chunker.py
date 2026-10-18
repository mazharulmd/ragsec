"""Sliding-window word chunker."""

from __future__ import annotations

from copy import deepcopy

from ragsec.core.models import Chunk, Document


class Chunker:
    """Split documents into overlapping windows of ``chunk_size`` words."""

    def __init__(self, chunk_size: int = 128, overlap: int = 16) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        words = document.text.split()
        if not words:
            return []
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        start = 0
        position = 0
        while True:
            window = words[start : start + self.chunk_size]
            chunks.append(
                Chunk(
                    text=" ".join(window),
                    doc_id=document.doc_id,
                    chunk_id=f"{document.doc_id}:{position}",
                    position=position,
                    provenance=deepcopy(document.provenance),
                    metadata=dict(document.metadata),
                )
            )
            if start + self.chunk_size >= len(words):
                break
            start += step
            position += 1
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for doc in documents for chunk in self.chunk(doc)]
