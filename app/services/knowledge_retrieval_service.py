from __future__ import annotations

from app.core.config import Settings
from app.db.repositories import LearningTraceRepository
from app.knowledge.embedding_provider import EmbeddingProvider
from app.knowledge.faiss_index_store import FaissIndexStore


class KnowledgeRetrievalService:
    def __init__(
        self,
        *,
        repository: LearningTraceRepository,
        embedding_provider: EmbeddingProvider,
        vector_index_store: FaissIndexStore,
        settings: Settings | None = None,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.vector_index_store = vector_index_store
        self.settings = settings or Settings()

    def retrieve_context(self, *, resource_id: str, query: str, top_k: int | None = None) -> list[dict]:
        document = self.repository.get_knowledge_document_for_resource(resource_id)
        if document is None:
            return []

        top_k_value = top_k or self.settings.knowledge_retrieval_top_k
        query_vector = self.embedding_provider.embed_text(query)
        hits = self.vector_index_store.search(
            resource_id=resource_id,
            query_vector=query_vector,
            top_k=top_k_value,
        )

        results: list[dict] = []
        for chunk_id, score in hits:
            chunk = self.repository.get_knowledge_chunk(chunk_id)
            if chunk is None:
                continue
            results.append(
                {
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "score": float(score),
                    "resource_id": resource_id,
                }
            )
        return results
