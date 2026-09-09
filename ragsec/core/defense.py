"""Base class for defense layers.

A defense layer can intervene at four points in the pipeline. All hooks are optional; the
defaults are no-ops so a layer only overrides what it needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragsec.core.llm import BaseLLM
from ragsec.core.models import Document, RAGResponse, RetrievedChunk

if TYPE_CHECKING:
    from ragsec.core.pipeline import RAGPipeline


class DefenseLayer:
    name: str = "base"

    def bind(self, pipeline: RAGPipeline) -> None:
        """Called when the layer is attached to a pipeline; grants access to its components."""
        return None

    def on_ingest(self, documents: list[Document]) -> list[Document]:
        """Inspect, annotate or drop documents before they enter the knowledge base."""
        return documents

    def on_retrieve(self, query: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Inspect, re-rank or drop retrieved chunks before generation."""
        return retrieved

    def generate(
        self, query: str, retrieved: list[RetrievedChunk], llm: BaseLLM
    ) -> str | None:
        """Optionally take over answer generation. Return ``None`` to defer."""
        return None

    def on_response(self, response: RAGResponse) -> None:
        """Observe the final response, e.g. for monitoring and alerting."""
        return None
