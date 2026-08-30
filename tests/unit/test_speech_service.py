from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.llm.types import LLMGenerateRequest, LLMGenerateResult
from app.services.assistant_service import AssistantService
from app.services.errors import STTProviderError, TTSProviderError
from app.services.learning_trace_service import LearningTraceService
from app.services.speech_service import SpeechResponse, SpeechService


class FakeSTTProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def transcribe(self, audio_bytes: bytes, *, mime_type: str | None = None) -> str:
        if self.fail:
            raise STTProviderError("speech recognition failed")
        if not audio_bytes:
            raise STTProviderError("no audio data")
        return "What is photosynthesis?"


class FakeTTSProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def synthesize(self, text: str) -> bytes:
        if self.fail:
            raise TTSProviderError("speech synthesis failed")
        if not text:
            raise TTSProviderError("empty text")
        return b"audio-bytes"


class FakeLLMProvider:
    def generate(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        return LLMGenerateResult(
            text="Photosynthesis is how plants make food using light.",
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
    learner = trace_service.create_learner({"display_name": "Sam"})
    session = trace_service.start_session({"learner_id": learner.id})
    return session


def test_stt_provider_success_and_failure():
    assert FakeSTTProvider().transcribe(b"voice-bytes") == "What is photosynthesis?"
    with pytest.raises(STTProviderError):
        FakeSTTProvider(fail=True).transcribe(b"voice-bytes")


def test_tts_provider_success_and_failure():
    assert FakeTTSProvider().synthesize("hello") == b"audio-bytes"
    with pytest.raises(TTSProviderError):
        FakeTTSProvider(fail=True).synthesize("hello")


def test_speech_service_orchestrates_audio_to_assistant_to_tts(db_session):
    session = create_learning_session(db_session)
    service = SpeechService(
        repository=LearningTraceRepository(db_session),
        assistant_service=AssistantService(
            repository=LearningTraceRepository(db_session),
            llm_provider=FakeLLMProvider(),
            settings=Settings(database_url="sqlite:///:memory:"),
        ),
        stt_provider=FakeSTTProvider(),
        tts_provider=FakeTTSProvider(),
    )

    result = service.answer_audio_question(
        session_id=session.id,
        audio_bytes=b"voice-bytes",
        mime_type="audio/wav",
    )

    assert isinstance(result, SpeechResponse)
    assert result.transcript == "What is photosynthesis?"
    assert result.answer == "Photosynthesis is how plants make food using light."
    assert result.audio_bytes == b"audio-bytes"
    assert result.response_latency_ms >= 0


def test_speech_service_propagates_stt_and_tts_errors(db_session):
    session = create_learning_session(db_session)

    service = SpeechService(
        repository=LearningTraceRepository(db_session),
        assistant_service=AssistantService(
            repository=LearningTraceRepository(db_session),
            llm_provider=FakeLLMProvider(),
            settings=Settings(database_url="sqlite:///:memory:"),
        ),
        stt_provider=FakeSTTProvider(fail=True),
        tts_provider=FakeTTSProvider(),
    )
    with pytest.raises(STTProviderError):
        service.answer_audio_question(session_id=session.id, audio_bytes=b"voice-bytes")

    service = SpeechService(
        repository=LearningTraceRepository(db_session),
        assistant_service=AssistantService(
            repository=LearningTraceRepository(db_session),
            llm_provider=FakeLLMProvider(),
            settings=Settings(database_url="sqlite:///:memory:"),
        ),
        stt_provider=FakeSTTProvider(),
        tts_provider=FakeTTSProvider(fail=True),
    )
    with pytest.raises(TTSProviderError):
        service.answer_audio_question(session_id=session.id, audio_bytes=b"voice-bytes")
