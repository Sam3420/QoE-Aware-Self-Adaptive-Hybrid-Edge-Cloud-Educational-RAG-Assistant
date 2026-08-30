from __future__ import annotations

import base64
from dataclasses import dataclass
from time import perf_counter

from app.llm.stt_provider import STTProvider
from app.llm.tts_provider import TTSProvider
from app.services.assistant_service import AssistantService
from app.services.errors import STTProviderError, TTSProviderError


@dataclass(frozen=True)
class SpeechResponse:
    interaction_id: str
    session_id: str
    transcript: str
    answer: str
    audio_base64: str
    response_latency_ms: int
    model_provider: str
    model_name: str
    runtime_configuration_id: str

    @property
    def audio_bytes(self) -> bytes:
        return base64.b64decode(self.audio_base64)


class SpeechService:
    def __init__(
        self,
        *,
        assistant_service: AssistantService | None = None,
        stt_provider: STTProvider | None = None,
        tts_provider: TTSProvider | None = None,
        repository: object | None = None,
    ) -> None:
        self.repository = repository
        self.assistant_service = assistant_service
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider

    def answer_audio_question(
        self,
        *,
        session_id: str,
        audio_bytes: bytes,
        mime_type: str | None = None,
        runtime_configuration_id: str | None = None,
        resource_id: str | None = None,
    ) -> SpeechResponse:
        started_at = perf_counter()

        try:
            transcript = self.stt_provider.transcribe(audio_bytes, mime_type=mime_type)
        except STTProviderError:
            raise
        if not transcript or not transcript.strip():
            raise STTProviderError("The configured STT provider returned an empty transcript.")

        assistant_response = self.assistant_service.answer_text_question(
            session_id=session_id,
            question=transcript,
            runtime_configuration_id=runtime_configuration_id,
            resource_id=resource_id,
        )

        try:
            synthesized_audio = self.tts_provider.synthesize(assistant_response.answer)
        except TTSProviderError:
            raise

        response_latency_ms = max(0, int((perf_counter() - started_at) * 1000))
        return SpeechResponse(
            interaction_id=assistant_response.interaction_id,
            session_id=assistant_response.session_id,
            transcript=transcript,
            answer=assistant_response.answer,
            audio_base64=base64.b64encode(synthesized_audio).decode("utf-8"),
            response_latency_ms=response_latency_ms,
            model_provider=assistant_response.model_provider,
            model_name=assistant_response.model_name,
            runtime_configuration_id=assistant_response.runtime_configuration_id,
        )
