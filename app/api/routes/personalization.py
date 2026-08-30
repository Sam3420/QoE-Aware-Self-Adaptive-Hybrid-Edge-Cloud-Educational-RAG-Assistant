from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db_session
from app.db.repositories import LearningTraceRepository
from app.domain.models import (
    AssessmentResultCreate,
    AssessmentResultResponse,
    LearnerPersonalizationProfile,
    LearnerPersonalizationUpdate,
)
from app.services.personalization_service import LearnerPersonalizationService

router = APIRouter()


@router.get(
    "/learners/{learner_id}/personalization",
    response_model=LearnerPersonalizationProfile,
)
def get_learner_personalization(
    learner_id: str,
    db_session: Session = Depends(get_db_session),
) -> LearnerPersonalizationProfile:
    service = LearnerPersonalizationService(LearningTraceRepository(db_session))
    try:
        profile = service.get_learner_personalization(learner_id)
        if profile is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Learner not found.",
            )
        return profile
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Learner personalization could not be retrieved.",
        ) from exc


@router.put(
    "/learners/{learner_id}/personalization",
    response_model=LearnerPersonalizationProfile,
)
def update_learner_personalization(
    learner_id: str,
    request: LearnerPersonalizationUpdate,
    db_session: Session = Depends(get_db_session),
) -> LearnerPersonalizationProfile:
    service = LearnerPersonalizationService(LearningTraceRepository(db_session))
    try:
        profile = service.update_learner_personalization(learner_id, request)
        db_session.commit()
        return profile
    except ValueError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except SQLAlchemyError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Learner personalization could not be updated.",
        ) from exc


@router.post(
    "/learners/{learner_id}/assessments",
    response_model=AssessmentResultResponse,
)
def record_assessment_result(
    learner_id: str,
    request: AssessmentResultCreate,
    db_session: Session = Depends(get_db_session),
) -> AssessmentResultResponse:
    service = LearnerPersonalizationService(LearningTraceRepository(db_session))
    try:
        result = service.record_assessment_result(learner_id, request)
        db_session.commit()
        return result
    except ValueError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except SQLAlchemyError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assessment result could not be saved.",
        ) from exc
