from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin, utc_now
from app.domain.enums import InteractionStatus


class Interaction(TimestampMixin, Base):
    __tablename__ = "interactions"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    runtime_configuration_id: Mapped[str] = mapped_column(
        ForeignKey("runtime_configurations.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    educational_resource_id: Mapped[str | None] = mapped_column(
        ForeignKey("educational_resources.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    user_input: Mapped[str] = mapped_column(nullable=False)
    assistant_output: Mapped[str | None] = mapped_column(nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    response_latency_ms: Mapped[int | None] = mapped_column(nullable=True)
    model_provider: Mapped[str | None] = mapped_column(nullable=True)
    model_name: Mapped[str | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        default=InteractionStatus.COMPLETED.value,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(nullable=True)

    session: Mapped["LearningSession"] = relationship(back_populates="interactions")
    runtime_configuration: Mapped["RuntimeConfiguration"] = relationship(
        back_populates="interactions",
    )
    educational_resource: Mapped["EducationalResource | None"] = relationship(
        back_populates="interactions",
    )
    runtime_metrics: Mapped[list["RuntimeMetric"]] = relationship(
        back_populates="interaction",
        cascade="all, delete-orphan",
    )
    qoe_scores: Mapped[list["QoEScoreRecord"]] = relationship(
        back_populates="interaction",
        cascade="all, delete-orphan",
    )
