from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_db_session, get_llm_provider
from app.db.base import Base
from app.db.models import Interaction
from app.db.repositories import LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from app.llm.types import LLMGenerateRequest, LLMGenerateResult
from app.main import create_app
from app.services.learning_trace_service import LearningTraceService


class FakeLLMProvider:
    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        return LLMGenerateResult(
            text="Photosynthesis is how plants make food using light.",
            model_provider="fake",
            model_name=request.model_id,
        )


def test_text_question_endpoint_returns_answer_and_persists_interaction():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with TestingSession() as seed_session:
        trace_service = LearningTraceService(LearningTraceRepository(seed_session))
        learner = trace_service.create_learner(LearnerProfileCreate(display_name="Sam"))
        learning_session = trace_service.start_session(
            LearningSessionCreate(learner_id=learner.id)
        )
        seed_session.commit()
        session_id = learning_session.id

    def override_db_session() -> Generator[Session, None, None]:
        with TestingSession() as db_session:
            yield db_session

    app = create_app()
    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLMProvider()

    with TestClient(app) as client:
        response = client.post(
            f"/sessions/{session_id}/interactions/text-question",
            json={"question": "What is photosynthesis?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Photosynthesis is how plants make food using light."
    assert payload["model_provider"] == "fake"
    assert payload["model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"

    with TestingSession() as assert_session:
        interaction = assert_session.get(Interaction, payload["interaction_id"])
        assert interaction is not None
        assert interaction.session_id == session_id
        assert interaction.assistant_output == payload["answer"]
        assert interaction.runtime_configuration_id == payload["runtime_configuration_id"]
