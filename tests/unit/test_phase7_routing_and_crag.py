from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from app.llm.hybrid_provider import HybridLLMProvider, LocalLLMProvider
from app.llm.types import LLMGenerateRequest, LLMGenerateResult, LLMProviderError
from app.services.assistant_service import AssistantService
from app.services.crag_service import RetrievalQualityService
from app.services.learning_trace_service import LearningTraceService


class FakeCloudProvider:
    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return LLMGenerateResult(
            text="Cloud response",
            model_provider="huggingface",
            model_name=request.model_id,
        )


class FakeLocalProvider:
    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return LLMGenerateResult(
            text="Local response",
            model_provider="local",
            model_name=request.model_id,
        )


def test_crag_quality_service_flags_inadequate_retrieval():
    service = RetrievalQualityService(min_score=0.75)
    decision = service.evaluate(question="What is photosynthesis?", hits=[{"score": 0.41, "content": "Biology studies life."}])

    assert decision.is_adequate is False
    assert decision.reason == "retrieval_quality_insufficient"


def test_assistant_uses_fallback_prompt_when_retrieval_is_inadequate():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with Session() as session:
        repo = LearningTraceRepository(session)
        trace_service = LearningTraceService(repo)
        learner = trace_service.create_learner(LearnerProfileCreate(display_name="Sam"))
        learning_session = trace_service.start_session(LearningSessionCreate(learner_id=learner.id))
        session.commit()

        class FakeLLMProvider:
            def __init__(self):
                self.requests = []

            def generate(self, request):
                self.requests.append(request)
                return LLMGenerateResult(
                    text="Fallback explanation",
                    model_provider="fake",
                    model_name=request.model_id,
                )

        class FakeRetrievalService:
            def retrieve_context(self, *, resource_id, query, top_k):
                return [{"chunk_id": "c1", "content": "Biology is the study of life.", "score": 0.2}]

        provider = FakeLLMProvider()
        assistant = AssistantService(
            repository=repo,
            llm_provider=provider,
            settings=Settings(database_url="sqlite:///:memory:"),
            knowledge_retrieval_service=FakeRetrievalService(),
        )

        assistant.answer_text_question(
            session_id=learning_session.id,
            question="What is photosynthesis?",
            resource_id="resource-1",
        )

        system_prompt = provider.requests[0].system_prompt
        assert "No sufficiently relevant retrieved context" in system_prompt


def test_hybrid_provider_routes_to_cloud_when_local_unavailable():
    cloud = FakeCloudProvider()
    local = LocalLLMProvider(model_id="local-model")
    provider = HybridLLMProvider(cloud_provider=cloud, local_provider=local, settings=Settings(local_llm_enabled=True, database_url="sqlite:///:memory:"))

    request = LLMGenerateRequest(
        prompt="Explain gravity",
        system_prompt="Be educational.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        max_tokens=64,
        temperature=0.2,
    )

    result = provider.generate(request)

    assert result.model_provider == "huggingface"
    assert len(cloud.calls) == 1


def test_hybrid_provider_uses_cloud_fallback_when_local_provider_fails():
    cloud = FakeCloudProvider()

    class FailingLocalProvider:
        def generate(self, request):
            raise LLMProviderError("Local model is not configured.")

    provider = HybridLLMProvider(
        cloud_provider=cloud,
        local_provider=FailingLocalProvider(),
        settings=Settings(local_llm_enabled=True, database_url="sqlite:///:memory:"),
    )

    request = LLMGenerateRequest(
        prompt="Explain gravity",
        system_prompt="Be educational.",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        max_tokens=64,
        temperature=0.2,
    )

    result = provider.generate(request)

    assert result.model_provider == "huggingface"
    assert len(cloud.calls) == 1
