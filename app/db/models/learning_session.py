from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin, utc_now
from app.domain.enums import SessionStatus


class LearningSession(TimestampMixin, Base):
    __tablename__ = "learning_sessions"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    learner_id: Mapped[str] = mapped_column(
        ForeignKey("learner_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        default=SessionStatus.ACTIVE.value,
        nullable=False,
    )

    learner: Mapped["LearnerProfile"] = relationship(back_populates="sessions")
    interactions: Mapped[list["Interaction"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    runtime_metrics: Mapped[list["RuntimeMetric"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    qoe_scores: Mapped[list["QoEScoreRecord"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    experiences: Mapped[list["Experience"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
