from __future__ import annotations

# Settings provides the default number of chunks requested during retrieval.
from app.core.config import Settings
# The repository reads prepared documents and their stored chunk text.
from app.db.repositories import LearningTraceRepository
# The embedding interface keeps query encoding compatible with preparation encoding.
from app.knowledge.embedding_provider import EmbeddingProvider
# The FAISS store searches vectors created for the resource.
from app.knowledge.faiss_index_store import FaissIndexStore


class KnowledgeRetrievalService:
    # The service coordinates vector search with relational text lookup.
    def __init__(
        self,
        *,
        repository: LearningTraceRepository,
        embedding_provider: EmbeddingProvider,
        vector_index_store: FaissIndexStore,
        settings: Settings | None = None,
    ) -> None:
        # Repository access retrieves the complete chunk records after vector search.
        self.repository = repository
        # This must be the same model family used to embed transcript chunks.
        self.embedding_provider = embedding_provider
        # The store returns matching chunk IDs and similarity scores.
        self.vector_index_store = vector_index_store
        # Explicit settings support tests and deployments with different retrieval limits.
        self.settings = settings or Settings()

    '''
    This function:

    Finds the prepared knowledge document.
    Embeds the question.
    Searches the FAISS index.
    Gets matching chunk IDs.
    Loads complete chunk text from SQLite.
    Returns ranked chunks.

    '''

    def retrieve_context(self, *, resource_id: str, query: str, top_k: int | None = None) -> list[dict]:
        # Retrieval is available only after preparation has created a document and index.
        # The document check prevents searches against resources that were never prepared.
        document = self.repository.get_knowledge_document_for_resource(resource_id)
        if document is None:
            # An empty list lets the assistant use its no-context fallback path.
            return []

        # Use the caller's limit when provided, otherwise use the configured default.
        top_k_value = top_k or self.settings.knowledge_retrieval_top_k
        # The query must use the same embedding model as the stored document chunks.
        query_vector = self.embedding_provider.embed_text(query)
        # Search only the selected resource's index so unrelated videos cannot contribute context.
        hits = self.vector_index_store.search(
            resource_id=resource_id,
            query_vector=query_vector,
            top_k=top_k_value,
        )

        # FAISS returns IDs and scores; the database remains the source of chunk text.
        # This result list is the typed-by-convention boundary consumed by RAG and CRAG that is not implemented completely yet 
        results: list[dict] = []
        for chunk_id, score in hits:
            # A stale or invalid sidecar ID is ignored instead of breaking the whole response.
            #creating chunk database record
            chunk = self.repository.get_knowledge_chunk(chunk_id)
            if chunk is None:
                continue
            # Return both evidence text and its score so CRAG can judge retrieval quality, but not implemented completely yet
            results.append(
                {
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "score": float(score),
                    "resource_id": resource_id,
                }
            )
        return results
