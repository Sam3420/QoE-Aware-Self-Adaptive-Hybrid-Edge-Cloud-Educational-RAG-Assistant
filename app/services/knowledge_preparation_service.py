#turn that selected resource into searchable knowledge for RAG
from __future__ import annotations

# Regular expressions clean transcript text and identify sentence boundaries.
import re
# Dataclasses provide a small immutable result object for the preparation workflow.
from dataclasses import dataclass

# Settings supplies configurable knowledge and retrieval defaults.
from app.core.config import Settings
# The repository persists documents, chunks, and index metadata in SQLite.
from app.db.repositories import LearningTraceRepository
# This interface allows different embedding models to be substituted.
from app.knowledge.embedding_provider import EmbeddingProvider
# This store writes and searches the FAISS vector index on disk.
from app.knowledge.faiss_index_store import FaissIndexStore
# This interface allows transcript sources to be changed without changing preparation.
from app.knowledge.transcript_provider import TranscriptProvider
# These errors become meaningful API responses in the route layer.
from app.services.errors import KnowledgePreparationError, ResourceNotFoundError


@dataclass(frozen=True)
class PreparedKnowledgeDocument:
    # The resource links the prepared knowledge back to the selected video.
    resource_id: str
    # The document ID identifies the stored transcript record.
    document_id: str
    # This tells the caller how many transcript chunks were created.
    chunk_count: int
    # A successful preparation is reported as ready.
    status: str = "ready"


class KnowledgePreparationService:
    # The service coordinates transcript, embedding, vector, and relational components.
    def __init__(
        self,
        *,
        repository: LearningTraceRepository,
        transcript_provider: TranscriptProvider,
        embedding_provider: EmbeddingProvider,
        vector_index_store: FaissIndexStore,
        settings: Settings | None = None,
    ) -> None:
        # Repository access is used for resources, documents, chunks, and index metadata.
        self.repository = repository
        # The provider supplies the raw transcript for the selected resource.
        self.transcript_provider = transcript_provider
        # The embedding provider converts text into numeric vectors.
        self.embedding_provider = embedding_provider
        # The vector store persists vectors separately from the relational database.
        self.vector_index_store = vector_index_store
        # Use explicit settings when supplied, otherwise load application defaults.
        self.settings = settings or Settings()

    '''
    This function:

    Loads the resource from SQLite.
    Gets the YouTube video ID.
    Fetches its transcript.
    Cleans the transcript.
    Splits it into chunks.
    Generates embeddings.
    Stores transcript and chunks in SQLite.
    Builds a FAISS index.
    Stores index metadata.
    Returns the document ID and chunk count.
    '''

    def prepare_resource(self, resource_id: str) -> PreparedKnowledgeDocument:
        # Preparation creates the relational document/chunks and the separate vector index.
        # Look up the resource so its title, source, URL, and external video ID are available.
        resource = self.repository.get_educational_resource(resource_id)
        if resource is None and resource_id.startswith("youtube:"):
            resource = self.repository.get_educational_resource_by_source_and_external_id(
                source="youtube",
                external_resource_id=resource_id.split(":", 1)[1],
            )
        # Preparation cannot continue when the caller supplied an unknown resource ID.
        if resource is None:
            raise ResourceNotFoundError("Educational resource was not found.")
        resource_id = resource.id

        # Transcript acquisition is isolated behind an interface so providers can be replaced.
        try:
            # Pass both IDs and the ORM resource to support provider implementations with context.
            transcript = self.transcript_provider.get_transcript_for_resource(
                resource_id=resource_id,
                resource=resource,
            )
        except TypeError:
            # Preserve compatibility with older providers that accept only the resource object.
            transcript = self.transcript_provider.get_transcript_for_resource(resource)
        except Exception as exc:
            raise KnowledgePreparationError(
                f"Transcript could not be fetched: {exc}"
            ) from exc
        # Remove formatting noise before splitting the transcript into chunks.
        cleaned_transcript = self._preprocess_transcript(transcript)
        # Split cleaned text into semantically useful pieces for embedding and retrieval.
        chunks = self._chunk_text(cleaned_transcript)
        # An empty transcript cannot produce searchable knowledge.
        if not chunks:
            raise KnowledgePreparationError("No usable transcript content was produced for the resource.")

        # Store the complete cleaned transcript as the parent knowledge document.
        document = self.repository.create_knowledge_document(
            resource_id=resource_id,
            title=resource.title,
            source=resource.source,
            transcript_text=cleaned_transcript,
            metadata={"resource_url": resource.url or ""},
        )

        # Store chunk IDs in the database; FAISS stores only vectors and their ID mapping.
        # Generate one vector for every chunk in the same order used below for chunk IDs.
        try:
            embeddings = self.embedding_provider.embed_documents(chunks)
        except Exception as exc:
            raise KnowledgePreparationError(
                f"Embeddings could not be generated: {exc}"
            ) from exc
        # Keep the database IDs that correspond to FAISS vector positions.
        chunk_ids = []
        # Enumerate preserves each chunk's order for later vector-to-record mapping.
        for index, chunk in enumerate(chunks):
            # Persist the text because FAISS should not be the source of complete content.
            chunk_record = self.repository.create_knowledge_chunk(
                document_id=document.id,
                chunk_index=index,
                content=chunk,
                metadata={"source_resource_id": resource_id},
            )
            # The index sidecar will use this ID to reconnect a vector hit to its text.
            chunk_ids.append(chunk_record.id)

        # Build the index after all chunk IDs are known so search can map hits back to text.
        # The vector store receives matching embeddings and database identifiers.
        try:
            index_state = self.vector_index_store.build_index(
                resource_id=resource.id,
                embeddings=embeddings,
                chunk_ids=chunk_ids,
            )
        except Exception as exc:
            raise KnowledgePreparationError(
                f"FAISS index could not be built: {exc}"
            ) from exc
        # Record where the index lives and which embedding model created it.
        self.repository.create_knowledge_index(
            document_id=document.id,
            index_name=f"{resource_id}-index",
            index_path=index_state["path"],
            embedding_model=self.embedding_provider.model_name,
            chunk_count=len(chunk_ids),
            metadata={"resource_id": resource_id},
        )

        # Mark preparation complete only after document, chunks, and index metadata exist.
        self.repository.update_knowledge_document_status(document.id, "ready")
        # Return a compact typed result to the API layer.
        return PreparedKnowledgeDocument(
            resource_id=resource_id,
            document_id=document.id,
            chunk_count=len(chunks),
            status="ready",
        )

    '''
    Cleans transcript text 
    '''
    @staticmethod
    def _preprocess_transcript(text: str) -> str:
        # Remove transcript artifacts before chunking to improve retrieval quality.
        # Normalize Windows and old-Mac line endings to one newline representation.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove common non-educational caption markers case-insensitively.
        text = re.sub(r"\[Music\]|\[Applause\]|\[Laughter\]", " ", text, flags=re.IGNORECASE)
        # Collapse tabs, newlines, and repeated spaces into single spaces.
        text = re.sub(r"\s+", " ", text)
        # Remove whitespace at both ends before storage and chunking.
        return text.strip()

    '''
    Splits the transcript into sentence-based chunks.
    '''
    @staticmethod
    def _chunk_text(text: str, *, chunk_size: int = 120, overlap: int = 25) -> list[str]:
        # Sentence boundaries preserve meaning better than cutting at arbitrary characters.
        # No input means there is nothing to index.
        if not text:
            return []

        # Remove leading and trailing whitespace before sentence detection.
        normalized = text.strip()
        # Split after sentence punctuation while retaining complete sentence meaning.
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]
        # Text without detectable sentences cannot produce useful chunks here.
        if not sentences:
            return []

        # Prevent invalid configuration values from breaking the chunking loop.
        chunk_size = max(1, chunk_size)
        # Overlap must leave room for at least one new character in a chunk.
        overlap = min(overlap, max(0, chunk_size - 1))
        # This list receives the final chunks in document order.
        chunks: list[str] = []
        # This list accumulates sentences for the current chunk.
        current: list[str] = []
        # Track approximate chunk length without repeatedly joining the list.
        current_length = 0

        for sentence in sentences:
            # Start a new chunk when adding the sentence would exceed the target size.
            sentence_length = len(sentence)
            if current and current_length + sentence_length > chunk_size:
                # Save the completed chunk before beginning the next one.
                chunk_text = " ".join(current)
                chunks.append(chunk_text)
                if overlap > 0:
                    # Retain the last sentence so adjacent chunks share context.
                    overlap_text = current[-1]
                    current = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    # With zero overlap, start the next chunk empty.
                    current = []
                    current_length = 0

            # Add the current sentence to the active chunk.
            current.append(sentence)
            # Include the sentence length in the running size estimate.
            current_length += sentence_length

        if current:
            # Flush the final partial chunk after the loop ends.
            chunks.append(" ".join(current))

        # Remove any defensive blank entries before returning the result.
        return [chunk for chunk in chunks if chunk.strip()]
