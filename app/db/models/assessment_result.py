from typing import Any

from sqlalchemy import JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin


class AssessmentResult(TimestampMixin, Base):
    __tablename__ = "assessment_results"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    learner_id: Mapped[str] = mapped_column(
        ForeignKey("learner_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    topic: Mapped[str] = mapped_column(nullable=False, index=True)
    competency_level: Mapped[str] = mapped_column(nullable=False)
    score: Mapped[float | None] = mapped_column(nullable=True)
    assessment_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    learner: Mapped["LearnerProfile"] = relationship(back_populates="assessment_results")
