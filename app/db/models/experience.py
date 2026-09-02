from typing import Any

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin


class Experience(TimestampMixin, Base):
    __tablename__ = "experiences"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    learner_id: Mapped[str] = mapped_column(
        ForeignKey("learner_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    interaction_id: Mapped[str] = mapped_column(
        ForeignKey("interactions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,
    )
    runtime_configuration_id: Mapped[str] = mapped_column(
        ForeignKey("runtime_configurations.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    qoe_score_id: Mapped[str | None] = mapped_column(
        ForeignKey("qoe_scores.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    resource_id: Mapped[str | None] = mapped_column(
        ForeignKey("educational_resources.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    topic: Mapped[str | None] = mapped_column(nullable=True)
    state_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    action_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    configuration_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    qoe_outcome: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    reward_score: Mapped[float] = mapped_column(nullable=False, default=0.0)
    reward_details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    outcome_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    performance_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    learner: Mapped["LearnerProfile"] = relationship(back_populates="experiences")
    session: Mapped["LearningSession"] = relationship(back_populates="experiences")
    interaction: Mapped["Interaction"] = relationship(back_populates="experiences")
    runtime_configuration: Mapped["RuntimeConfiguration"] = relationship(back_populates="experiences")
    qoe_score: Mapped["QoEScoreRecord | None"] = relationship(back_populates="experience")
    resource: Mapped["EducationalResource | None"] = relationship(back_populates="experiences")
