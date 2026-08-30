import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.base import Base
from app.db.models import Interaction
from app.db.repositories import LearningTraceRepository
from app.domain.enums import InteractionStatus
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from app.llm.types import LLMGenerateRequest, LLMGenerateResult, LLMProviderError
from app.services.assistant_service import AssistantService
from app.services.errors import AssistantProviderUnavailableError, LearningSessionNotFoundError
from app.services.learning_trace_service import LearningTraceService


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
    assert stored.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert stored.response_latency_ms is not None
    assert stored.status == InteractionStatus.COMPLETED.value
    assert provider.requests[0].prompt == "What is a variable?"


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
    assert stored.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
