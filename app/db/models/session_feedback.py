from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.core.ids import new_id
from app.db.base import Base, utc_now


class SessionFeedback(Base):
    """Learner feedback given with the post-test: an optional 1-5 rating and comment."""

    __tablename__ = "session_feedback"
    __table_args__ = (
        CheckConstraint("rating IS NULL OR rating BETWEEN 1 AND 5", name="ck_session_feedback_rating"),
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
    rating: Mapped[int | None] = mapped_column(nullable=True)
    comment: Mapped[str | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
