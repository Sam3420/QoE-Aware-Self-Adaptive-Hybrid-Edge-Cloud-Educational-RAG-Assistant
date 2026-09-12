from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db_session
from app.db.repositories import LearningTraceRepository
from app.domain.models import (
    LearnerProfileCreate,
    LearnerProfileResponse,
    LearningSessionCreate,
    LearningSessionResponse,
)
from app.services.learning_trace_service import LearningTraceService

router = APIRouter()


@router.post("/learners", response_model=LearnerProfileResponse, status_code=status.HTTP_201_CREATED)
def create_learner(
    request: LearnerProfileCreate,
    db_session: Session = Depends(get_db_session),
) -> LearnerProfileResponse:
    try:
        learner = LearningTraceService(LearningTraceRepository(db_session)).create_learner(request)
        db_session.commit()
        return LearnerProfileResponse.model_validate(learner)
    except IntegrityError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid learner data or database schema.",
        ) from exc
    except SQLAlchemyError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Learner could not be created. Check the database schema.",
        ) from exc


@router.post("/sessions", response_model=LearningSessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    request: LearningSessionCreate,
    db_session: Session = Depends(get_db_session),
) -> LearningSessionResponse:
    try:
        repository = LearningTraceRepository(db_session)
        if repository.get_learner(request.learner_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Learner not found.",
            )
        session = LearningTraceService(repository).start_session(request)
        db_session.commit()
        return LearningSessionResponse.model_validate(session)
    except HTTPException:
        db_session.rollback()
        raise
    except IntegrityError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid learner_id or session data.",
        ) from exc
    except SQLAlchemyError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Learning session could not be created. Check the database schema.",
        ) from exc