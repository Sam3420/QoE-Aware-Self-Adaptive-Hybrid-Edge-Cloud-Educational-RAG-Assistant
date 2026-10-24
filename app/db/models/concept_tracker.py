from datetime import datetime

from sqlalchemy import JSON, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin, utc_now


class LearnerTopicStrength(TimestampMixin, Base):
    """A learner's standing on one concept of a topic (LearnerTopicStrength in the design ER diagram).

    One row per (learner, topic, concept). ``attempts``/``correct`` count graded answers on the concept (pre-test,
    post-test, formative checks); ``mastery`` = (correct + 1) / (attempts + 2) is stored for querying and recomputed on
    every attempt. ``covered_count`` counts exposures (the concept came up in a question or a video) and says nothing
    about mastery. Rows are an aggregate of ``concept_events``, which holds the raw observations.
    """

    __tablename__ = "learner_topic_strength"
    __table_args__ = (
        UniqueConstraint("student_id", "topic_id", "concept_id", name="uq_strength_student_topic_concept"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("learner_profiles.id", ondelete="CASCADE"), nullable=False,
                                            index=True)
    topic_id: Mapped[str] = mapped_column(nullable=False, index=True)
    concept_id: Mapped[str] = mapped_column(String(64), nullable=False)
    attempts: Mapped[int] = mapped_column(default=0, nullable=False)
    correct: Mapped[int] = mapped_column(default=0, nullable=False)
    mastery: Mapped[float] = mapped_column(default=0.5, nullable=False)  # (0 + 1) / (0 + 2) before any evidence
    covered_count: Mapped[int] = mapped_column(default=0, nullable=False)
    first_covered_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_covered_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_attempt_at: Mapped[datetime | None] = mapped_column(nullable=True)
    # Added by migration 0002 (tracker.py): a video covered the concept; questions the learner asked about it; source of
    # the latest graded attempt ("pre-test", "post-test", "formative").
    covered_video: Mapped[bool] = mapped_column(default=False, server_default="0", nullable=False)
    asked_count: Mapped[int] = mapped_column(default=0, server_default="0", nullable=False)
    last_source: Mapped[str | None] = mapped_column(String(32), nullable=True)


class ConceptEvent(Base):
    """One raw observation behind a LearnerTopicStrength row: a graded attempt or an exposure (coverage).

    ``(kind, source, source_ref, concept_id)`` is unique, so recording the same observation twice (a re-run backfill,
    a retried node) changes nothing. ``source_ref`` points at the origin, e.g. ``"<assessment id>:<item index>"``.
    """

    __tablename__ = "concept_events"
    __table_args__ = (
        UniqueConstraint("kind", "source", "source_ref", "concept_id", name="uq_concept_event_origin"),
        Index("ix_concept_events_student_topic", "student_id", "topic_id"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    student_id: Mapped[str] = mapped_column(ForeignKey("learner_profiles.id", ondelete="CASCADE"), nullable=False)
    topic_id: Mapped[str] = mapped_column(nullable=False)
    concept_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(8), nullable=False)  # "attempt" | "covered"
    correct: Mapped[bool | None] = mapped_column(nullable=True)  # attempts only
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # "pre-test", "post-test", "formative", "question"
    source_ref: Mapped[str] = mapped_column(nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)


class FormativeItem(Base):
    """One formative-check question on a concept (formative_pool.py).

    Kept apart from topic_question_banks on purpose: any bank item can be a pre- or post-test question, so formative
    practice drawn from the bank would rehearse the post-test and inflate learning gain. ``question_key`` (normalised
    text) is unique per topic, so a question is never stored twice.
    """

    __tablename__ = "formative_items"
    __table_args__ = (
        UniqueConstraint("topic_id", "question_key", name="uq_formative_topic_question"),
        Index("ix_formative_topic_concept", "topic_id", "concept_id"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # "fp_<hex>", stable: the Item.item_id
    topic_id: Mapped[str] = mapped_column(nullable=False)
    concept_id: Mapped[str] = mapped_column(String(64), nullable=False)
    position: Mapped[int] = mapped_column(nullable=False)  # order within the concept's pool (served oldest first)
    question: Mapped[str] = mapped_column(nullable=False)
    question_key: Mapped[str] = mapped_column(nullable=False)
    options: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    answer_index: Mapped[int] = mapped_column(nullable=False)
    explanation: Mapped[str] = mapped_column(default="", nullable=False)
    concept_check: Mapped[str] = mapped_column(String(16), nullable=False)  # label | matched | unverified
    model_provider: Mapped[str | None] = mapped_column(nullable=True)
    model_name: Mapped[str | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
