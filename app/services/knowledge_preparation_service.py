from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import Settings
from app.db.repositories import LearningTraceRepository
from app.knowledge.embedding_provider import EmbeddingProvider
from app.knowledge.faiss_index_store import FaissIndexStore
from app.knowledge.transcript_provider import TranscriptProvider
from app.services.errors import KnowledgePreparationError, ResourceNotFoundError


@dataclass(frozen=True)
class PreparedKnowledgeDocument:
    resource_id: str
    document_id: str
    chunk_count: int
    status: str = "ready"


class KnowledgePreparationService:
    def __init__(
        self,
        *,
        repository: LearningTraceRepository,
        transcript_provider: TranscriptProvider,
        embedding_provider: EmbeddingProvider,
        vector_index_store: FaissIndexStore,
        settings: Settings | None = None,
    ) -> None:
        self.repository = repository
        self.transcript_provider = transcript_provider
        self.embedding_provider = embedding_provider
        self.vector_index_store = vector_index_store
        self.settings = settings or Settings()

    def prepare_resource(self, resource_id: str) -> PreparedKnowledgeDocument:
        resource = self.repository.get_educational_resource(resource_id)
        if resource is None:
            raise ResourceNotFoundError("Educational resource was not found.")

        transcript = self.transcript_provider.get_transcript_for_resource(
            resource_id=resource_id,
            resource=resource,
        )
        cleaned_transcript = self._preprocess_transcript(transcript)
        chunks = self._chunk_text(cleaned_transcript)
        if not chunks:
            raise KnowledgePreparationError("No usable transcript content was produced for the resource.")

        document = self.repository.create_knowledge_document(
            resource_id=resource_id,
            title=resource.title,
            source=resource.source,
            transcript_text=cleaned_transcript,
            metadata={"resource_url": resource.url or ""},
        )

        embeddings = self.embedding_provider.embed_documents(chunks)
        chunk_ids = []
        for index, chunk in enumerate(chunks):
            chunk_record = self.repository.create_knowledge_chunk(
                document_id=document.id,
                chunk_index=index,
                content=chunk,
                metadata={"source_resource_id": resource_id},
            )
            chunk_ids.append(chunk_record.id)

        index_state = self.vector_index_store.build_index(
            resource_id=resource_id,
            embeddings=embeddings,
            chunk_ids=chunk_ids,
        )
        self.repository.create_knowledge_index(
            document_id=document.id,
            index_name=f"{resource_id}-index",
            index_path=index_state["path"],
            embedding_model=self.embedding_provider.model_name,
            chunk_count=len(chunk_ids),
            metadata={"resource_id": resource_id},
        )

        self.repository.update_knowledge_document_status(document.id, "ready")
        return PreparedKnowledgeDocument(
            resource_id=resource_id,
            document_id=document.id,
            chunk_count=len(chunks),
            status="ready",
        )

    @staticmethod
    def _preprocess_transcript(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\[Music\]|\[Applause\]|\[Laughter\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _chunk_text(text: str, *, chunk_size: int = 500, overlap: int = 80) -> list[str]:
        if not text:
            return []

        normalized = text.strip()
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]
        if not sentences:
            return []

        chunk_size = max(1, chunk_size)
        overlap = min(overlap, max(0, chunk_size - 1))
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current and current_length + sentence_length > chunk_size:
                chunk_text = " ".join(current)
                chunks.append(chunk_text)
                if overlap > 0:
                    overlap_text = current[-1]
                    current = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current = []
                    current_length = 0

            current.append(sentence)
            current_length += sentence_length

        if current:
            chunks.append(" ".join(current))

        return [chunk for chunk in chunks if chunk.strip()]
