from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_assistant_service, get_db_session
from app.domain.models import TextQuestionRequest, TextQuestionResponse
from app.services.assistant_service import AssistantService
from app.services.errors import (
    AssistantProviderUnavailableError,
    LearningSessionNotFoundError,
    RuntimeConfigurationNotFoundError,
)

router = APIRouter()


@router.post(
    "/sessions/{session_id}/interactions/text-question",
    response_model=TextQuestionResponse,
)
def submit_text_question(
    session_id: str,
    request: TextQuestionRequest,
    db_session: Session = Depends(get_db_session),
    assistant_service: AssistantService = Depends(get_assistant_service),
) -> TextQuestionResponse:
    try:
        result = assistant_service.answer_text_question(
            session_id=session_id,
            question=request.question,
            runtime_configuration_id=request.runtime_configuration_id,
            resource_id=request.resource_id,
        )
        db_session.commit()
        return TextQuestionResponse(**result.__dict__)
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
    except AssistantProviderUnavailableError as exc:
        db_session.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Assistant provider is currently unavailable.",
                "interaction_id": exc.interaction_id,
            },
        ) from exc
    except SQLAlchemyError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Interaction could not be stored.",
        ) from exc
