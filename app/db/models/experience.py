from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.core.ids import new_id
from app.db.base import Base, utc_now


class Experience(Base):
    """One stored session outcome: context state, chosen policy, and its evaluation."""

    __tablename__ = "experiences"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(
        ForeignKey("learner_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    topic_id: Mapped[str] = mapped_column(nullable=False, index=True)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    policy_id: Mapped[str] = mapped_column(
        ForeignKey("policies.policy_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    qoe_before: Mapped[float | None] = mapped_column(nullable=True)
    qoe_after: Mapped[float | None] = mapped_column(nullable=True)
    sre_before: Mapped[float | None] = mapped_column(nullable=True)
    sre_after: Mapped[float | None] = mapped_column(nullable=True)
    reward: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
