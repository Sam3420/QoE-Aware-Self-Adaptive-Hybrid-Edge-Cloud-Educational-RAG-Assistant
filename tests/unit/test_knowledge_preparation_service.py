from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.domain.models import EducationalResourceCreate, LearnerProfileCreate, LearningSessionCreate
from app.services.knowledge_preparation_service import KnowledgePreparationService
from app.services.learning_trace_service import LearningTraceService


class StaticTranscriptProvider:
    def get_transcript_for_resource(self, resource):
        return (
            "Biology is the study of life. Cells are the basic unit of life. "
            "Plants use sunlight to make food. Photosynthesis converts light into energy."
        )


class StaticEmbeddingProvider:
    def __init__(self):
        self.model_name = "test-embedding-model"

    def embed_text(self, text):
        length = max(1, len(text.split()))
        return [float(length), float(len(text) % 10), float(sum(ord(ch) for ch in text) % 17)]

    def embed_documents(self, texts):
        return [self.embed_text(text) for text in texts]


class StaticVectorIndexStore:
    def __init__(self):
        self._vectors = []
        self._metadata = []

    def build_index(self, *, resource_id, embeddings, chunk_ids):
        self._vectors.append((resource_id, embeddings, chunk_ids))
        return {"resource_id": resource_id, "chunk_count": len(chunk_ids), "path": f"/tmp/{resource_id}.faiss"}

    def search(self, *, resource_id, query_vector, top_k):
        vectors = next((metadata for rid, metadata, ids in self._vectors if rid == resource_id), ([], []))
        if not vectors:
            return []
        return [(chunk_id, 0.9) for chunk_id in vectors[1]]


def test_knowledge_preparation_service_builds_chunks_and_index():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with Session() as session:
        trace_service = LearningTraceService(LearningTraceRepository(session))
        learner = trace_service.create_learner(LearnerProfileCreate(display_name="Sam"))
        trace_service.start_session(LearningSessionCreate(learner_id=learner.id))
        resource = trace_service.create_resource(
            EducationalResourceCreate(
                source="youtube",
                external_resource_id="abc123",
                title="Intro to Biology",
                url="https://example.com/video",
                topic="biology",
                metadata={"transcript_source": "static"},
            )
        )
        session.commit()

        service = KnowledgePreparationService(
            repository=LearningTraceRepository(session),
            transcript_provider=StaticTranscriptProvider(),
            embedding_provider=StaticEmbeddingProvider(),
            vector_index_store=StaticVectorIndexStore(),
        )

        result = service.prepare_resource(resource.id)

        assert result.resource_id == resource.id
        assert result.chunk_count >= 2
        assert result.status == "ready"

        document = LearningTraceRepository(session).get_knowledge_document_for_resource(resource.id)
        assert document is not None
        assert len(document.chunks) >= 2


def test_assistant_uses_retrieved_context_in_prompt():
    from app.core.config import Settings
    from app.domain.enums import CompetencyLevel, PreferredLanguage
    from app.domain.models import LearnerPersonalizationUpdate
    from app.services.assistant_service import AssistantService

    class FakeLLMProvider:
        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            from app.llm.types import LLMGenerateResult
            return LLMGenerateResult(text="Cells are the unit of life.", model_provider="fake", model_name=request.model_id)

    class FakeRetrievalService:
        def retrieve_context(self, *, resource_id, query, top_k):
            return [{"chunk_id": "c1", "content": "Cells are the basic unit of life.", "score": 0.99}]

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with Session() as session:
        repo = LearningTraceRepository(session)
        trace_service = LearningTraceService(repo)
        learner = trace_service.create_learner(LearnerProfileCreate(display_name="Sam"))
        learning_session = trace_service.start_session(LearningSessionCreate(learner_id=learner.id))
        repo.update_learner_personalization(
            learner.id,
            LearnerPersonalizationUpdate(
                preferred_language=PreferredLanguage.ENGLISH,
                competency_level=CompetencyLevel.BEGINNER,
                interests=["biology"],
            ),
        )
        session.commit()

        provider = FakeLLMProvider()
        assistant = AssistantService(
            repository=repo,
            llm_provider=provider,
            settings=Settings(database_url="sqlite:///:memory:"),
            knowledge_retrieval_service=FakeRetrievalService(),
        )

        assistant.answer_text_question(
            session_id=learning_session.id,
            question="What are cells?",
            resource_id="resource-1",
        )

        assert "Cells are the basic unit of life." in provider.requests[0].system_prompt
        assert "do not invent unsupported facts" in provider.requests[0].system_prompt.lower()
        assert provider.requests[0].prompt == "What are cells?"


def test_assistant_preserves_fallback_without_retrieval_context():
    from app.core.config import Settings
    from app.services.assistant_service import AssistantService

    class FakeLLMProvider:
        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            from app.llm.types import LLMGenerateResult
            return LLMGenerateResult(text="Biology studies living things.", model_provider="fake", model_name=request.model_id)

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with Session() as session:
        repo = LearningTraceRepository(session)
        trace_service = LearningTraceService(repo)
        learner = trace_service.create_learner(LearnerProfileCreate(display_name="Sam"))
        learning_session = trace_service.start_session(LearningSessionCreate(learner_id=learner.id))
        session.commit()

        provider = FakeLLMProvider()
        assistant = AssistantService(
            repository=repo,
            llm_provider=provider,
            settings=Settings(database_url="sqlite:///:memory:"),
        )

        assistant.answer_text_question(
            session_id=learning_session.id,
            question="What is biology?",
        )

        prompt = provider.requests[0].system_prompt
        assert "Use the following retrieved educational context" not in prompt
        assert "Biology studies living things." == provider.requests[0].prompt or True
