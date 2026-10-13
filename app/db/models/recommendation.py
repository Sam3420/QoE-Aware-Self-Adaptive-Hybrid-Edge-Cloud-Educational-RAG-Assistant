from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin, utc_now


class RecommendationRun(TimestampMixin, Base):
    """One video-recommendation request: context, chosen strategy, results, instrumentation and outcomes.

    Shaped like an experience (state + strategy + reward) so the strategy selector can run kNN over
    runs. Strategy ids (R1-R3) are kept apart from the Q&A `experiences` table on purpose: that table's
    policy_id references the Q&A policies P1-P4, and mixing the two would let either selector pick the
    other's actions.
    """

    __tablename__ = "recommendation_runs"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("learner_profiles.id", ondelete="CASCADE"), nullable=False,
                                            index=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("learning_sessions.id", ondelete="SET NULL"),
                                                   nullable=True, index=True)
    topic_id: Mapped[str] = mapped_column(nullable=False, index=True)
    objective: Mapped[str] = mapped_column(nullable=False)
    query: Mapped[str | None] = mapped_column(nullable=True)
    strategy_id: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    decision_reason: Mapped[str | None] = mapped_column(nullable=True)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)  # kNN state
    strategy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)  # config executed
    request: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    errors: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)  # pipeline instrumentation
    outcome: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # evaluation metrics
    qoe_signals: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    reward: Mapped[float | None] = mapped_column(nullable=True)


class RecommendationItem(Base):
    """One recommended video as shown to the learner (full payload kept for replay)."""

    __tablename__ = "recommendation_items"
    __table_args__ = (Index("ix_recommendation_items_run_rank", "run_id", "rank"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("recommendation_runs.id", ondelete="CASCADE"), nullable=False)
    rank: Mapped[int] = mapped_column(nullable=False)
    video_id: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    title: Mapped[str] = mapped_column(nullable=False)
    score: Mapped[float] = mapped_column(nullable=False)
    evidence_level: Mapped[str] = mapped_column(String(16), nullable=False)
    transcript_status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)


class RecommendationEvent(Base):
    """A learner interaction with a recommendation (displayed, opened, accepted, dismissed, rated, completed)."""

    __tablename__ = "recommendation_events"
    __table_args__ = (Index("ix_recommendation_events_student_video", "student_id", "video_id"),)

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("recommendation_runs.id", ondelete="CASCADE"), nullable=False,
                                        index=True)
    item_id: Mapped[str | None] = mapped_column(ForeignKey("recommendation_items.id", ondelete="CASCADE"),
                                                nullable=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("learner_profiles.id", ondelete="CASCADE"), nullable=False)
    video_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    value: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # e.g. {"rating": 4}
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
