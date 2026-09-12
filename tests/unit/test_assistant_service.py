import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.base import Base
from app.db.models import Experience, Interaction
from app.db.repositories import LearningTraceRepository
from app.domain.enums import CompetencyLevel, InteractionStatus, PreferredLanguage
from app.domain.models import (
    LearnerPersonalizationUpdate,
    LearnerProfileCreate,
    LearningSessionCreate,
)
from app.llm.types import LLMGenerateRequest, LLMGenerateResult, LLMProviderError
from app.services.assistant_service import AssistantService
from app.services.errors import AssistantProviderUnavailableError, LearningSessionNotFoundError
from app.services.learning_trace_service import LearningTraceService
from app.services.personalization_service import LearnerPersonalizationService


class FakeLLMProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[LLMGenerateRequest] = []

    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        self.requests.append(request)
        if self.fail:
            raise LLMProviderError("provider exploded")
        return LLMGenerateResult(
            text="A variable is a named value that can change.",
            model_provider="fake",
            model_name=request.model_id,
        )


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with Session() as session:
        yield session


def create_learning_session(db_session):
    trace_service = LearningTraceService(LearningTraceRepository(db_session))
    learner = trace_service.create_learner(LearnerProfileCreate(display_name="Sam"))
    session = trace_service.start_session(LearningSessionCreate(learner_id=learner.id))
    return session


def test_assistant_records_successful_text_question(db_session):
    learning_session = create_learning_session(db_session)
    provider = FakeLLMProvider()
    service = AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=provider,
        settings=Settings(database_url="sqlite:///:memory:"),
    )

    result = service.answer_text_question(
        session_id=learning_session.id,
        question="What is a variable?",
    )
    db_session.commit()

    stored = db_session.get(Interaction, result.interaction_id)
    assert stored is not None
    assert stored.session.learner_id == learning_session.learner_id
    assert stored.assistant_output == "A variable is a named value that can change."
    assert stored.model_provider == "fake"
    assert stored.model_name == service.settings.llm_model_id
    assert stored.response_latency_ms is not None
    assert stored.status == InteractionStatus.COMPLETED.value
    assert provider.requests[0].prompt == "What is a variable?"


def test_assistant_uses_personalization_context_in_prompt(db_session):
    learning_session = create_learning_session(db_session)
    learner_id = learning_session.learner_id
    personalization_service = LearnerPersonalizationService(LearningTraceRepository(db_session))
    personalization_service.update_learner_personalization(
        learner_id,
        LearnerPersonalizationUpdate(
            preferred_language=PreferredLanguage.HINDI,
            competency_level=CompetencyLevel.BEGINNER,
            interests=["biology"],
        ),
    )
    provider = FakeLLMProvider()
    service = AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=provider,
        settings=Settings(database_url="sqlite:///:memory:"),
        personalization_service=personalization_service,
    )

    service.answer_text_question(session_id=learning_session.id, question="What is a variable?")

    prompt = provider.requests[0]
    assert "Hindi" in prompt.system_prompt
    assert "simple" in prompt.system_prompt.lower()


def test_learner_personalization_service_builds_context(db_session):
    trace_service = LearningTraceService(LearningTraceRepository(db_session))
    learner = trace_service.create_learner(LearnerProfileCreate(display_name="Sam"))
    personalization_service = LearnerPersonalizationService(LearningTraceRepository(db_session))

    personalization_service.update_learner_personalization(
        learner.id,
        LearnerPersonalizationUpdate(
            preferred_language=PreferredLanguage.ENGLISH,
            competency_level=CompetencyLevel.INTERMEDIATE,
            interests=["mathematics", "science"],
        ),
    )

    context = personalization_service.get_context_for_learner(learner.id)
    assert context.learner_id == learner.id
    assert context.preferred_language == PreferredLanguage.ENGLISH
    assert context.competency_level == CompetencyLevel.INTERMEDIATE
    assert context.interests == ["mathematics", "science"]


def test_assistant_rejects_unknown_session(db_session):
    service = AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=FakeLLMProvider(),
        settings=Settings(database_url="sqlite:///:memory:"),
    )

    with pytest.raises(LearningSessionNotFoundError):
        service.answer_text_question(session_id="missing", question="Hello?")


def test_assistant_records_failed_interaction_for_provider_error(db_session):
    learning_session = create_learning_session(db_session)
    service = AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=FakeLLMProvider(fail=True),
        settings=Settings(database_url="sqlite:///:memory:"),
    )

    with pytest.raises(AssistantProviderUnavailableError) as exc_info:
        service.answer_text_question(
            session_id=learning_session.id,
            question="What is gravity?",
        )
    db_session.commit()

    stored = db_session.get(Interaction, exc_info.value.interaction_id)
    assert stored is not None
    assert stored.status == InteractionStatus.FAILED.value
    assert stored.assistant_output is None
    assert stored.error_message == "LLM provider failed to generate a response."
    assert stored.model_provider == "huggingface"
    assert stored.model_name == service.settings.llm_model_id


def test_assistant_creates_experience_record_automatically(db_session):
    learning_session = create_learning_session(db_session)
    provider = FakeLLMProvider()
    service = AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=provider,
        settings=Settings(database_url="sqlite:///:memory:"),
    )

    result = service.answer_text_question(
        session_id=learning_session.id,
        question="What is a variable?",
    )
    db_session.commit()

    stored_experience = (
        db_session.query(Experience)
        .filter(Experience.interaction_id == result.interaction_id)
        .one_or_none()
    )

    assert stored_experience is not None
    assert stored_experience.session_id == learning_session.id
    assert stored_experience.learner_id == learning_session.learner_id
    assert stored_experience.runtime_configuration_id is not None


def test_assistant_marks_retrieval_as_not_applicable_for_non_rag_question(db_session):
    learning_session = create_learning_session(db_session)

    class RecordingAdaptiveService:
        def __init__(self):
            self.retrieval_quality_is_adequate = "not-called"

        def select_action(self, **kwargs):
            self.retrieval_quality_is_adequate = kwargs["retrieval_quality_is_adequate"]
            return None

    adaptive_service = RecordingAdaptiveService()
    service = AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=FakeLLMProvider(),
        settings=Settings(database_url="sqlite:///:memory:"),
        adaptive_intelligence_service=adaptive_service,
    )

    service.answer_text_question(
        session_id=learning_session.id,
        question="What is a variable?",
    )

    assert adaptive_service.retrieval_quality_is_adequate is None


def test_assistant_resolves_latest_runtime_configuration_snapshot(db_session):
    trace_service = LearningTraceService(LearningTraceRepository(db_session))
    trace_service.create_runtime_configuration_snapshot(
        name="huggingface-qwen-text",
        description="adaptive runtime configuration",
        configuration_data={"provider": "local", "model_id": "local-model", "max_tokens": 256},
    )

    service = AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=FakeLLMProvider(),
        settings=Settings(database_url="sqlite:///:memory:"),
    )

    runtime_configuration = service._resolve_runtime_configuration(None)

    assert runtime_configuration.configuration_data["provider"] == "local"
    assert runtime_configuration.configuration_data["model_id"] == "local-model"


def test_assistant_uses_runtime_configuration_retrieval_top_k(db_session):
    learning_session = create_learning_session(db_session)
    trace_service = LearningTraceService(LearningTraceRepository(db_session))
    runtime_configuration = trace_service.create_runtime_configuration_snapshot(
        name="retrieval-depth",
        description="configured retrieval depth",
        configuration_data={"provider": "huggingface", "knowledge_retrieval_top_k": 7},
    )

    class RecordingRetrievalService:
        def __init__(self):
            self.top_k = None

        def retrieve_context(self, *, resource_id, query, top_k):
            self.top_k = top_k
            return [{"chunk_id": "chunk-1", "content": "Retrieved context.", "score": 0.9}]

    retrieval_service = RecordingRetrievalService()
    service = AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=FakeLLMProvider(),
        settings=Settings(database_url="sqlite:///:memory:"),
        knowledge_retrieval_service=retrieval_service,
    )

    service.answer_text_question(
        session_id=learning_session.id,
        question="What is a variable?",
        resource_id="resource-1",
        runtime_configuration_id=runtime_configuration.id,
    )

    assert retrieval_service.top_k == 7


def test_assistant_uses_settings_retrieval_top_k_without_runtime_value(db_session):
    learning_session = create_learning_session(db_session)

    class RecordingRetrievalService:
        def __init__(self):
            self.top_k = None

        def retrieve_context(self, *, resource_id, query, top_k):
            self.top_k = top_k
            return [{"chunk_id": "chunk-1", "content": "Retrieved context.", "score": 0.9}]

    retrieval_service = RecordingRetrievalService()
    settings = Settings(database_url="sqlite:///:memory:", knowledge_retrieval_top_k=4)
    service = AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=FakeLLMProvider(),
        settings=settings,
        knowledge_retrieval_service=retrieval_service,
    )

    service.answer_text_question(
        session_id=learning_session.id,
        question="What is a variable?",
        resource_id="resource-1",
    )

    assert retrieval_service.top_k == settings.knowledge_retrieval_top_k
