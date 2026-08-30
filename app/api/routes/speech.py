from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db_session, get_speech_service
from app.domain.models import SpeechQuestionRequest, SpeechQuestionResponse
from app.services.errors import AssistantProviderUnavailableError, LearningSessionNotFoundError, RuntimeConfigurationNotFoundError, STTProviderError, TTSProviderError
from app.services.speech_service import SpeechService

router = APIRouter()


@router.post(
    "/sessions/{session_id}/interactions/speech",
    response_model=SpeechQuestionResponse,
)
async def submit_speech_question(
    session_id: str,
    audio_file: UploadFile = File(...),
    runtime_configuration_id: str | None = Form(default=None),
    resource_id: str | None = Form(default=None),
    db_session: Session = Depends(get_db_session),
    speech_service: SpeechService = Depends(get_speech_service),
) -> SpeechQuestionResponse:
    try:
        audio_bytes = await audio_file.read()
        result = speech_service.answer_audio_question(
            session_id=session_id,
            audio_bytes=audio_bytes,
            mime_type=audio_file.content_type,
            runtime_configuration_id=runtime_configuration_id,
            resource_id=resource_id,
        )
        db_session.commit()

        if hasattr(result, "model_dump"):
            payload = result.model_dump()
        else:
            payload = {}
            if hasattr(result, "__dict__") and result.__dict__:
                payload.update(dict(result.__dict__))
            for field_name in (
                "interaction_id",
                "session_id",
                "transcript",
                "answer",
                "audio_base64",
                "response_latency_ms",
                "model_provider",
                "model_name",
                "runtime_configuration_id",
            ):
                if field_name not in payload:
                    value = getattr(result, field_name, None)
                    if value is not None:
                        payload[field_name] = value
            if not payload:
                for name, value in vars(type(result)).items():
                    if not name.startswith("__") and not callable(value):
                        payload[name] = getattr(result, name)
        return SpeechQuestionResponse(**payload)
    except LearningSessionNotFoundError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning session was not found.",
        ) from exc
    except RuntimeConfigurationNotFoundError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Runtime configuration was not found.",
        ) from exc
    except (STTProviderError, TTSProviderError, AssistantProviderUnavailableError) as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except SQLAlchemyError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Speech interaction could not be stored.",
        ) from exc
