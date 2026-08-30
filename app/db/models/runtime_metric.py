from datetime import datetime
from typing import Any
from sqlalchemy import JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin, utc_now


class RuntimeMetric(TimestampMixin, Base):
    __tablename__ = "runtime_metrics"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    interaction_id: Mapped[str] = mapped_column(
        ForeignKey("interactions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    metric_name: Mapped[str] = mapped_column(nullable=False, index=True)
    metric_category: Mapped[str] = mapped_column(nullable=False, default="performance")
    metric_value: Mapped[float] = mapped_column(nullable=False)
    unit: Mapped[str | None] = mapped_column(nullable=True)
    observed_at: Mapped[datetime] = mapped_column(
    default=utc_now,
    nullable=False,
)
    observation_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    interaction: Mapped["Interaction"] = relationship(back_populates="runtime_metrics")
    session: Mapped["LearningSession"] = relationship(back_populates="runtime_metrics")
