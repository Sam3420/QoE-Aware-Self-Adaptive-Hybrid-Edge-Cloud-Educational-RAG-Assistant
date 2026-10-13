from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.errors import ReferencedRecordNotFoundError
from app.db.models import (
    Assessment,
    LearnerProfile,
    LearningSession,
    RecommendationEvent,
    RecommendationItem,
    RecommendationRun,
)
from app.domain.enums import AssessmentType

EVENT_TYPES = frozenset({"generated", "displayed", "opened", "accepted", "dismissed", "rated", "completed"})


class RecommendationRepository:
    """Persistence for recommendation runs, items and learner events. Returns plain dicts."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # Runs and items

    def create_run(self, *, student_id: str, session_id: str | None, topic_id: str, objective: str,
                   query: str | None, strategy_id: str, decision_reason: str | None, context: dict[str, Any],
                   strategy: dict[str, Any], request: dict[str, Any], status: str, errors: list[str],
                   metrics: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
        """Store a run with its ranked items (each item dict is a Recommendation payload)."""
        if self.session.get(LearnerProfile, student_id) is None:
            raise ReferencedRecordNotFoundError(f"Learner {student_id!r} does not exist.")
        if session_id is not None and self.session.get(LearningSession, session_id) is None:
            raise ReferencedRecordNotFoundError(f"Learning session {session_id!r} does not exist.")
        run = RecommendationRun(student_id=student_id, session_id=session_id, topic_id=topic_id, objective=objective,
                                query=query, strategy_id=strategy_id, decision_reason=decision_reason,
                                context=context, strategy=strategy, request=request, status=status,
                                errors=errors, metrics=metrics)
        self.session.add(run)
        self.session.flush()
        for payload in items:
            self.session.add(RecommendationItem(
                run_id=run.id, rank=payload["rank"], video_id=payload["video_id"], title=payload["title"],
                score=payload["score"], evidence_level=payload["evidence_level"],
                transcript_status=payload["transcript_status"], payload=payload))
        self.session.flush()
        return self.get_run(run.id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a run with its items (ordered by rank), or None."""
        run = self.session.get(RecommendationRun, run_id)
        if run is None:
            return None
        items = self.session.scalars(select(RecommendationItem).where(RecommendationItem.run_id == run_id)
                                     .order_by(RecommendationItem.rank))
        return {**_run_to_dict(run), "items": [_item_to_dict(i) for i in items]}

    def list_runs(self, *, student_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Runs newest first (without items)."""
        statement = select(RecommendationRun).order_by(RecommendationRun.created_at.desc()).limit(limit)
        if student_id is not None:
            statement = statement.where(RecommendationRun.student_id == student_id)
        return [_run_to_dict(r) for r in self.session.scalars(statement)]

    def update_outcome(self, run_id: str, *, outcome: dict[str, Any], qoe_signals: dict[str, Any],
                       reward: float | None) -> None:
        run = self.session.get(RecommendationRun, run_id)
        if run is None:
            raise ReferencedRecordNotFoundError(f"Recommendation run {run_id!r} does not exist.")
        run.outcome, run.qoe_signals, run.reward = outcome, qoe_signals, reward
        self.session.flush()

    def get_experience_runs(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Runs shaped like get_experiences() for kNN: {session_id, state, policy_id, reward}."""
        statement = select(RecommendationRun).order_by(RecommendationRun.created_at.desc())
        if limit is not None:
            statement = statement.limit(limit)
        return [{"session_id": r.id, "state": r.context, "policy_id": r.strategy_id, "reward": r.reward}
                for r in self.session.scalars(statement)]

    # Events

    def add_event(self, *, run_id: str, event_type: str, item_id: str | None = None,
                  value: dict[str, Any] | None = None) -> dict[str, Any]:
        """Store one event; the item must belong to the run, and the learner is taken from the run."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type {event_type!r}; allowed: {sorted(EVENT_TYPES)}.")
        run = self.session.get(RecommendationRun, run_id)
        if run is None:
            raise ReferencedRecordNotFoundError(f"Recommendation run {run_id!r} does not exist.")
        video_id = None
        if item_id is not None:
            item = self.session.get(RecommendationItem, item_id)
            if item is None or item.run_id != run_id:
                raise ReferencedRecordNotFoundError(f"Item {item_id!r} is not part of run {run_id!r}.")
            video_id = item.video_id
        event = RecommendationEvent(run_id=run_id, item_id=item_id, student_id=run.student_id, video_id=video_id,
                                    event_type=event_type, value=value)
        self.session.add(event)
        self.session.flush()
        return _event_to_dict(event)

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        statement = (select(RecommendationEvent).where(RecommendationEvent.run_id == run_id)
                     .order_by(RecommendationEvent.created_at))
        return [_event_to_dict(e) for e in self.session.scalars(statement)]

    def learner_video_states(self, student_id: str) -> dict[str, str]:
        """video_id → 'completed' or 'dismissed' (completed wins), for excluding repeats."""
        statement = select(RecommendationEvent.video_id, RecommendationEvent.event_type).where(
            RecommendationEvent.student_id == student_id,
            RecommendationEvent.event_type.in_(("completed", "dismissed")),
            RecommendationEvent.video_id.is_not(None))
        states: dict[str, str] = {}
        for video_id, event_type in self.session.execute(statement):
            if states.get(video_id) != "completed":
                states[video_id] = event_type
        return states

    def video_history(self, video_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Feedback across all learners for these videos: accepted/dismissed counts and ratings."""
        if not video_ids:
            return {}
        statement = select(RecommendationEvent).where(
            RecommendationEvent.video_id.in_(video_ids),
            RecommendationEvent.event_type.in_(("accepted", "dismissed", "rated")))
        history: dict[str, dict[str, Any]] = {}
        for e in self.session.scalars(statement):
            h = history.setdefault(e.video_id, {"accepted": 0, "dismissed": 0, "ratings": []})
            if e.event_type == "rated":
                rating = (e.value or {}).get("rating")
                if isinstance(rating, (int, float)):
                    h["ratings"].append(rating)
            else:
                h[e.event_type] += 1
        return history

    # Assessments around a run (read-only; written by the pre/post-test services)

    def assessments_around(self, student_id: str, topic_id: str, at: datetime) -> tuple[float | None, float | None]:
        """(latest pre-test score at or before `at`, earliest post-test score after `at`) for the topic."""
        base = select(Assessment.score).where(Assessment.student_id == student_id, Assessment.topic_id == topic_id)
        pre = self.session.scalar(base.where(Assessment.type == AssessmentType.PRE.value, Assessment.created_at <= at)
                                  .order_by(Assessment.created_at.desc()).limit(1))
        post = self.session.scalar(base.where(Assessment.type == AssessmentType.POST.value, Assessment.created_at > at)
                                   .order_by(Assessment.created_at).limit(1))
        return pre, post


def _run_to_dict(run: RecommendationRun) -> dict[str, Any]:
    return {
        "id": run.id, "student_id": run.student_id, "session_id": run.session_id, "topic_id": run.topic_id,
        "objective": run.objective, "query": run.query, "strategy_id": run.strategy_id,
        "decision_reason": run.decision_reason, "context": run.context, "strategy": run.strategy,
        "request": run.request, "status": run.status, "errors": run.errors, "metrics": run.metrics,
        "outcome": run.outcome, "qoe_signals": run.qoe_signals, "reward": run.reward, "created_at": run.created_at,
    }


def _item_to_dict(item: RecommendationItem) -> dict[str, Any]:
    return {"id": item.id, "run_id": item.run_id, "rank": item.rank, "video_id": item.video_id, "title": item.title,
            "score": item.score, "evidence_level": item.evidence_level, "transcript_status": item.transcript_status,
            "payload": item.payload, "created_at": item.created_at}


def _event_to_dict(event: RecommendationEvent) -> dict[str, Any]:
    return {"id": event.id, "run_id": event.run_id, "item_id": event.item_id, "student_id": event.student_id,
            "video_id": event.video_id, "event_type": event.event_type, "value": event.value,
            "created_at": event.created_at}
