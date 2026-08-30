from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin, utc_now


class QoEScoreRecord(TimestampMixin, Base):
    __tablename__ = "qoe_scores"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    interaction_id: Mapped[str] = mapped_column(
        ForeignKey("interactions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    score: Mapped[float] = mapped_column(nullable=False)
    quality_label: Mapped[str] = mapped_column(String(32), nullable=False, default="good")
    latency_ms: Mapped[int | None] = mapped_column(nullable=True)
    model_provider: Mapped[str | None] = mapped_column(nullable=True)
    model_name: Mapped[str | None] = mapped_column(nullable=True)
    observed_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)

    interaction: Mapped["Interaction"] = relationship(back_populates="qoe_scores")
    session: Mapped["LearningSession"] = relationship(back_populates="qoe_scores")
