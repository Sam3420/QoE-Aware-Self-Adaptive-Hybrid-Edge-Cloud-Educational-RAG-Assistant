from typing import Any

from sqlalchemy import JSON, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin


class Assessment(TimestampMixin, Base):
    """A pre- or post-test attempt within one learning session.

    Kept separate from ``assessment_results`` on purpose: that table stores a
    session-less, 0-100 competency snapshot for personalization, while this one
    stores a session-bound 0-1 MCQ attempt (items + answers) used by the agent.
    """

    __tablename__ = "assessments"
    __table_args__ = (
        Index("ix_assessments_student_topic_type", "student_id", "topic_id", "type"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    student_id: Mapped[str] = mapped_column(
        ForeignKey("learner_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    topic_id: Mapped[str] = mapped_column(nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(8), nullable=False)
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    answers: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    score: Mapped[float] = mapped_column(nullable=False)
    competency: Mapped[str] = mapped_column(String(32), nullable=False)
