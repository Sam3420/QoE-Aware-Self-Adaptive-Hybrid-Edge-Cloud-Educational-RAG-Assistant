"""Video recommendation endpoints (YouTube Data API v3 + optional transcripts), backed by agent.recommendations.

A run whose YouTube search failed is still stored and returned with status="failed" and its errors,
so clients can show the reason and offer a retry.
"""
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.db.errors import ReferencedRecordNotFoundError

router = APIRouter(tags=["recommendations"])


class RecommendationCreate(BaseModel):
    topic_id: str = Field(min_length=1, max_length=200)
    objective: str | None = Field(default=None, max_length=500)  # default: topic + knowledge-gap concepts
    concepts: list[str] | None = Field(default=None, max_length=20)  # default: concepts missed in the latest test
    session_id: str | None = None
    revision: bool = False  # True: completed/dismissed videos may be recommended again
    count: int | None = Field(default=None, ge=1, le=20)


class RecommendationEventCreate(BaseModel):
    event_type: str  # displayed | opened | accepted | dismissed | rated | completed
    item_id: str | None = None
    value: dict[str, Any] | None = None  # {"rating": 1-5} for "rated"


@lru_cache
def _default_recommender():
    from agent.recommendations import make_recommender  # imported lazily: wires the agent's services

    recommender = make_recommender()
    recommender.services.app.ensure_ready()
    return recommender


def get_recommender():
    """FastAPI dependency (override in tests)."""
    return _default_recommender()


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/learners/{student_id}/recommendations")
def create_recommendations(student_id: str, body: RecommendationCreate,
                           recommender=Depends(get_recommender)) -> dict[str, Any]:
    from agent.interfaces import LearnerNotFound

    try:
        return recommender.recommend(student_id, body.topic_id, objective=body.objective, concepts=body.concepts,
                                     session_id=body.session_id, revision=body.revision, count=body.count)
    except (LearnerNotFound, ReferencedRecordNotFoundError) as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/learners/{student_id}/recommendations")
def list_recommendations(student_id: str, limit: int = 20, recommender=Depends(get_recommender)) -> list[dict[str, Any]]:
    return recommender.list_runs(student_id=student_id, limit=max(1, min(limit, 100)))


@router.get("/recommendations/{run_id}")
def get_recommendation_run(run_id: str, recommender=Depends(get_recommender)) -> dict[str, Any]:
    run = recommender.get_run(run_id)
    if run is None:
        raise _not_found(LookupError(f"Recommendation run {run_id!r} does not exist."))
    return run


@router.post("/recommendations/{run_id}/events", status_code=status.HTTP_201_CREATED)
def create_recommendation_event(run_id: str, body: RecommendationEventCreate,
                                recommender=Depends(get_recommender)) -> dict[str, Any]:
    try:
        return recommender.record_event(run_id, body.event_type, item_id=body.item_id, value=body.value)
    except ReferencedRecordNotFoundError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/recommendations/{run_id}/metrics")
def get_recommendation_metrics(run_id: str, recommender=Depends(get_recommender)) -> dict[str, Any]:
    try:
        return recommender.evaluate(run_id)
    except LookupError as exc:
        raise _not_found(exc) from exc
