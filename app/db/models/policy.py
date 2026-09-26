from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Policy(Base):
    """A fixed, predefined runtime policy (P1-P4) the agent can select.

    Kept separate from ``runtime_configurations``: those are versioned, immutable
    snapshots of what an interaction actually ran with, keyed by UUID, whereas
    policies are a small static catalogue keyed by ``"P1"``..``"P4"``.
    """

    __tablename__ = "policies"

    policy_id: Mapped[str] = mapped_column(String(8), primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    top_k: Mapped[int] = mapped_column(nullable=False)
    threshold: Mapped[float] = mapped_column(nullable=False)
    max_retries: Mapped[int] = mapped_column(nullable=False)
    route: Mapped[str] = mapped_column(String(8), nullable=False)
    is_default: Mapped[bool] = mapped_column(default=False, nullable=False)
    description: Mapped[str | None] = mapped_column(nullable=True)
