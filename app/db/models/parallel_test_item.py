from datetime import datetime
from typing import Any

from sqlalchemy import JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utc_now


class ParallelTestItem(Base):
    """The post-test question paired with one pre-test bank question (same concept and difficulty).

    Kept out of ``topic_question_banks.items`` on purpose: bank items are served by the pre-test and copied into
    pre-test attempts, so a pairing stored there would put post-test questions and their answer keys into pre-test
    payloads. These rows are read only when a post-test begins (app/services/posttest.py), by the pre-test item's
    content id (``pre_item_id`` = the bank item's ``id``). ``item`` holds the full question including
    ``answer_index``; never send it to a learner as-is.
    """

    __tablename__ = "parallel_test_items"
    __table_args__ = (UniqueConstraint("topic_id", "pre_item_id", name="uq_parallel_topic_pre_item"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # the post-test item's content id
    topic_id: Mapped[str] = mapped_column(nullable=False, index=True)
    pre_item_id: Mapped[str] = mapped_column(String(40), nullable=False)
    item: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source: Mapped[str] = mapped_column(nullable=False)  # e.g. "package:life_processes@0.1.0-draft"
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
