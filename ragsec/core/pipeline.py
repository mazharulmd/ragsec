"""The RAG pipeline that composes core components with optional defense layers."""

from __future__ import annotations

import time
from collections.abc import Sequence

from ragsec.core.chunker import Chunker
from ragsec.core.defense import DefenseLayer
from ragsec.core.embedder import BaseEmbedder, get_embedder
from ragsec.core.llm import SYSTEM_PROMPT, BaseLLM, MockLLM, build_prompt
from ragsec.core.models import Document, IngestResult, RAGResponse, RetrievedChunk
from ragsec.core.retriever import Retriever
from ragsec.core.vector_store import VectorStore


class RAGPipeline:
    """Ingest documents and answer queries, routing through defense layers in order."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: VectorStore,
        llm: BaseLLM,
        chunker: Chunker | None = None,
        defenses: Sequence[DefenseLayer] | None = None,
        top_k: int = 5,
        candidate_multiplier: int = 3,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.chunker = chunker or Chunker()
        self.defenses: list[DefenseLayer] = []
        self.top_k = top_k
        self.candidate_multiplier = candidate_multiplier
        self.retriever = Retriever(embedder, store)
        for layer in defenses or []:
            self.add_defense(layer)

    @classmethod
    def from_defaults(
        cls,
        embedder: BaseEmbedder | None = None,
        llm: BaseLLM | None = None,
        defenses: Sequence[DefenseLayer] | None = None,
        top_k: int = 5,
        chunk_size: int = 128,
        chunk_overlap: int = 16,
        use_faiss: bool | None = None,
        candidate_multiplier: int = 3,
    ) -> RAGPipeline:
        embedder = embedder or get_embedder()
        return cls(
            embedder=embedder,
            store=VectorStore(embedder.dim, use_faiss=use_faiss),
            llm=llm or MockLLM(),
            chunker=Chunker(chunk_size=chunk_size, overlap=chunk_overlap),
            defenses=defenses,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
        )

    def add_defense(self, layer: DefenseLayer) -> None:
        self.defenses.append(layer)
        layer.bind(self)

    def ingest(self, documents: Sequence[Document]) -> IngestResult:
        submitted = len(documents)
        accepted = list(documents)
        for layer in self.defenses:
            accepted = layer.on_ingest(accepted)
        chunks = self.chunker.chunk_many(accepted)
        if chunks:
            self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        return IngestResult(
            accepted_documents=len(accepted),
            rejected_documents=submitted - len(accepted),
            chunks_added=len(chunks),
        )

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Raw retrieval without defense filtering."""
        return self.retriever.retrieve(query, k or self.top_k)

    def query(self, query: str) -> RAGResponse:
        start = time.perf_counter()
        # With defenses attached, over-retrieve so that dropped candidates can be backfilled.
        n_candidates = self.top_k * self.candidate_multiplier if self.defenses else self.top_k
        retrieved = self.retrieve(query, n_candidates)
        retrieve_ms = (time.perf_counter() - start) * 1000

        used = retrieved
        for layer in self.defenses:
            used = layer.on_retrieve(query, used)
        used = used[: self.top_k]

        answer: str | None = None
        generator = "llm"
        for layer in self.defenses:
            answer = layer.generate(query, used, self.llm)
            if answer is not None:
                generator = layer.name
                break
        if answer is None:
            prompt = build_prompt(query, [rc.chunk.text for rc in used])
            answer = self.llm.generate(prompt, system=SYSTEM_PROMPT)

        response = RAGResponse(
            query=query,
            answer=answer,
            retrieved=retrieved,
            used=used,
            metadata={
                "retrieve_ms": retrieve_ms,
                "total_ms": (time.perf_counter() - start) * 1000,
                "num_candidates": n_candidates,
                "num_retrieved": len(retrieved),
                "num_used": len(used),
                "num_filtered": len(retrieved) - len(used),
                "generator": generator,
                "defenses": [layer.name for layer in self.defenses],
            },
        )
        for layer in self.defenses:
            layer.on_response(response)
        return response
