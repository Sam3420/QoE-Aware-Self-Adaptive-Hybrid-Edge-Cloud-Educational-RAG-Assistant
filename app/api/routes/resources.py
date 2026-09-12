from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db_session, get_resource_recommendation_service
from app.domain.models import (
    ResourceRecommendationRequest,
    ResourceRecommendationResponse,
    ResourceSelectionResponse,
)
from app.services.errors import (
    LearnerNotFoundError,
    LearningSessionNotFoundError,
    LearningSessionOwnershipError,
    NoRecommendationsFoundError,
    ResourceNotFoundError,
    YouTubeProviderConfigurationError,
    YouTubeProviderError,
)
from app.services.resource_recommendation_service import ResourceRecommendationService

router = APIRouter()#grp for all api end points


@router.post(
    "/sessions/{session_id}/resources/recommendations",
    response_model=ResourceRecommendationResponse,#The successful response from this endpoint must follow the structure defined by ResourceRecommendationResponse
)
def get_resource_recommendations(
    session_id: str,
    request: ResourceRecommendationRequest,#JSON body sent by the frontend.
    db_session: Session = Depends(get_db_session),
    recommendation_service: ResourceRecommendationService = Depends(
        get_resource_recommendation_service,
    ),
) -> ResourceRecommendationResponse:
    # Search and rank provider candidates using the session learner's personalization context.
    try:
        result = recommendation_service.get_recommendations_for_session(
            session_id=session_id,
            request=request,
        )
        db_session.commit()
        return result
    except LearningSessionNotFoundError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning session was not found.",
        ) from exc
    except LearnerNotFoundError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learner was not found.",
        ) from exc
    except NoRecommendationsFoundError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recommendations were found for the requested topic.",
        ) from exc
    except YouTubeProviderConfigurationError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YouTube API configuration is missing or invalid.",
        ) from exc
    except YouTubeProviderError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YouTube recommendation provider is unavailable.",
        ) from exc
    except SQLAlchemyError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation request could not be persisted.",
        ) from exc


@router.post(
    "/sessions/{session_id}/resources/{resource_id}/select",
    response_model=ResourceSelectionResponse,
)
def select_resource(
    session_id: str,
    resource_id: str,
    db_session: Session = Depends(get_db_session),
    recommendation_service: ResourceRecommendationService = Depends(
        get_resource_recommendation_service,
    ),
) -> ResourceSelectionResponse:
    # Confirm the chosen resource before knowledge preparation begins.
    try:
        result = recommendation_service.select_resource_for_session(
            session_id=session_id,
            resource_id=resource_id,
        )
        db_session.commit()
        return result
    except LearningSessionNotFoundError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning session was not found.",
        ) from exc
    except ResourceNotFoundError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Educational resource was not found.",
        ) from exc
    except LearningSessionOwnershipError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session does not belong to the specified learner.",
        ) from exc
    except SQLAlchemyError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resource selection could not be stored.",
        ) from exc
