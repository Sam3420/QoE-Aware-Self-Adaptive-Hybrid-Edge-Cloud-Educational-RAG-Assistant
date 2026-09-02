from typing import Any

from sqlalchemy import JSON, UniqueConstraint, event, func, select
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin
from app.db.errors import ImmutableRuntimeConfigurationError


class RuntimeConfiguration(TimestampMixin, Base):
    __tablename__ = "runtime_configurations"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_runtime_config_name_version"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(nullable=False, index=True)
    version: Mapped[int] = mapped_column(nullable=False)
    description: Mapped[str | None] = mapped_column(nullable=True)
    configuration_data: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )

    interactions: Mapped[list["Interaction"]] = relationship(
        back_populates="runtime_configuration",
    )
    experiences: Mapped[list["Experience"]] = relationship(
        back_populates="runtime_configuration",
        cascade="all, delete-orphan",
    )


@event.listens_for(RuntimeConfiguration, "before_update")
def prevent_referenced_runtime_configuration_updates(mapper, connection, target) -> None:
    from app.db.models.interaction import Interaction

    interaction_count = connection.scalar(
        select(func.count(Interaction.id)).where(
            Interaction.runtime_configuration_id == target.id,
        )
    )
    if interaction_count:
        raise ImmutableRuntimeConfigurationError(
            "RuntimeConfiguration records referenced by interactions are immutable; "
            "create a new version instead."
        )
