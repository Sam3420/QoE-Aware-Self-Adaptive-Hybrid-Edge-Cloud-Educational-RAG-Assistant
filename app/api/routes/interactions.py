# FastAPI supplies routing, dependency injection, and HTTP error responses.
from fastapi import APIRouter, Depends, HTTPException, status
# Database failures are converted into safe HTTP responses.
from sqlalchemy.exc import SQLAlchemyError
# Session is the request-scoped database connection type.
from sqlalchemy.orm import Session

# These dependencies construct the assistant and provide the database session.
from app.api.dependencies import get_assistant_service, get_db_session
# These schemas validate incoming JSON and serialize the successful response.
from app.domain.models import TextQuestionRequest, TextQuestionResponse
# The service contains the Phase 6 RAG orchestration logic.
from app.services.assistant_service import AssistantService
from app.services.errors import (
    AssistantProviderUnavailableError,
    LearningSessionNotFoundError,
    RuntimeConfigurationNotFoundError,
)

# This router is registered by app.main to expose interaction endpoints.
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
    # The optional resource ID determines whether this request uses RAG and CRAG.
    try:
        # Pass the URL session ID and validated request fields into the service.
        result = assistant_service.answer_text_question(
            session_id=session_id,
            question=request.question,
            runtime_configuration_id=request.runtime_configuration_id,
            resource_id=request.resource_id,
        )
        # Commit the interaction and runtime configuration created by the service.
        db_session.commit()
        # Convert the service response dataclass into the public response schema.
        return TextQuestionResponse(**result.__dict__)
    except LearningSessionNotFoundError as exc:
        # Roll back pending work before reporting an unknown session.
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning session was not found.",
        ) from exc
    except RuntimeConfigurationNotFoundError as exc:
        # Roll back when an explicitly requested configuration does not exist.
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Runtime configuration was not found.",
        ) from exc
    except AssistantProviderUnavailableError as exc:
        # Keep the failed interaction record, then return a safe provider error.
        db_session.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Assistant provider is currently unavailable.",
                "interaction_id": exc.interaction_id,
            },
        ) from exc
    except SQLAlchemyError as exc:
        # Roll back database errors so the session is clean for later requests.
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Interaction could not be stored.",
        ) from exc
