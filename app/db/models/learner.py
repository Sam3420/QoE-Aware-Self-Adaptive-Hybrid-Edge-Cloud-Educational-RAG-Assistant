from typing import Any

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin


class LearnerProfile(TimestampMixin, Base):
    __tablename__ = "learner_profiles"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    display_name: Mapped[str | None] = mapped_column(nullable=True)
    profile_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    preferred_language: Mapped[str | None] = mapped_column(nullable=True)
    competency_level: Mapped[str | None] = mapped_column(nullable=True)
    interests: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    learning_preferences: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    sessions: Mapped[list["LearningSession"]] = relationship(
        back_populates="learner",
        cascade="all, delete-orphan",
    )
    assessment_results: Mapped[list["AssessmentResult"]] = relationship(
        back_populates="learner",
        cascade="all, delete-orphan",
    )
    experiences: Mapped[list["Experience"]] = relationship(
        back_populates="learner",
        cascade="all, delete-orphan",
    )
