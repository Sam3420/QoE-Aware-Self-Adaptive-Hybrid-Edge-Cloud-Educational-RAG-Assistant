from typing import Any

from sqlalchemy import JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin


class EducationalResource(TimestampMixin, Base):
    __tablename__ = "educational_resources"
    __table_args__ = (
        UniqueConstraint("source", "external_resource_id", name="uq_resource_source_external_id"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    source: Mapped[str] = mapped_column(nullable=False)
    external_resource_id: Mapped[str] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(nullable=False)
    url: Mapped[str | None] = mapped_column(nullable=True)
    topic: Mapped[str | None] = mapped_column(nullable=True)
    resource_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )

    interactions: Mapped[list["Interaction"]] = relationship(
        back_populates="educational_resource",
    )
