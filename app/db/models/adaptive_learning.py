from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utc_now


class ExperienceTransition(Base):
    """The full (s, a, R, s_next) record for one `experiences` row (same id), written by the adaptive loop.

    The `experiences` row keeps the selection context, policy and QoE/SRE before/after; this row adds
    what reward computation and policy selection need: both states' metrics, the executed action, the
    component scores, predicted vs. observed utility and the weights version. One per session (idempotent).
    """

    __tablename__ = "experience_transitions"

    experience_id: Mapped[str] = mapped_column(ForeignKey("experiences.id", ondelete="CASCADE"), primary_key=True)
    session_id: Mapped[str] = mapped_column(nullable=False, unique=True)
    action: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # {"action_id", "config"}
    prev_state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # RuntimeState s
    next_state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # RuntimeState s_next
    scores: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # RewardResult (components, deltas)
    utility_before: Mapped[float] = mapped_column(nullable=False)
    utility_after: Mapped[float] = mapped_column(nullable=False)
    predicted_utility: Mapped[float | None] = mapped_column(nullable=True)  # estimate at selection time
    baseline_used: Mapped[bool] = mapped_column(nullable=False, default=False)
    weights_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)


class ExperienceVector(Base):
    """FAISS id ↔ experience mapping plus the embedded state vector (so the index can be rebuilt)."""

    __tablename__ = "experience_vectors"

    faiss_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experience_id: Mapped[str] = mapped_column(ForeignKey("experiences.id", ondelete="CASCADE"), nullable=False,
                                               unique=True)
    vector: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)


class WeightSetRecord(Base):
    """A derived weight group (EWM) with the data window it came from; `active` ones override config."""

    __tablename__ = "weight_sets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    group: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(16), nullable=False)  # "ewm" | "fallback"
    weights: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    baseline: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # entropy, divergence, fallback reason
    n_observations: Mapped[int] = mapped_column(nullable=False)
    n_used: Mapped[int] = mapped_column(nullable=False)
    window_start: Mapped[datetime | None] = mapped_column(nullable=True)
    window_end: Mapped[datetime | None] = mapped_column(nullable=True)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. experience count + last id
    active: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
