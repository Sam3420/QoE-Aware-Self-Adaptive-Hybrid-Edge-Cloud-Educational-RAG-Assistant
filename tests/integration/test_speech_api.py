from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_db_session
from app.main import create_app
from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from app.services.learning_trace_service import LearningTraceService


class FakeSTTProvider:
    def transcribe(self, audio_bytes: bytes, *, mime_type: str | None = None) -> str:
        return "What is photosynthesis?"


class FakeTTSProvider:
    def synthesize(self, text: str) -> bytes:
        return b"speech-audio"


class FakeLLMProvider:
    def generate(self, request):
        return type(
            "Result",
            (),
            {"text": "Photosynthesis is how plants make food using light.", "model_provider": "fake", "model_name": request.model_id},
        )()


def test_speech_endpoint_returns_answer_and_audio():
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

    from app.api.dependencies import get_speech_service
    app.dependency_overrides[get_speech_service] = lambda: type(
        "SpeechServiceStub",
        (),
        {
            "answer_audio_question": lambda self, **kwargs: type(
                "Result",
                (),
                {
                    "interaction_id": "int-123",
                    "session_id": session_id,
                    "transcript": "What is photosynthesis?",
                    "answer": "Photosynthesis is how plants make food using light.",
                    "audio_base64": "c3BlZWNoLWF1ZGlv",
                    "response_latency_ms": 100,
                    "model_provider": "fake",
                    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                    "runtime_configuration_id": "rt-123",
                },
            )(),
        },
    )()

    with TestClient(app) as client:
        response = client.post(
            f"/sessions/{session_id}/interactions/speech",
            files={"audio_file": ("voice.wav", b"voice-data", "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript"] == "What is photosynthesis?"
    assert payload["answer"] == "Photosynthesis is how plants make food using light."
    assert payload["audio_base64"] == "c3BlZWNoLWF1ZGlv"
