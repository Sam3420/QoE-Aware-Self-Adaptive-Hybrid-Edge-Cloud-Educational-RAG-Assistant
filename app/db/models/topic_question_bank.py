from typing import Any

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.core.ids import new_id
from app.db.base import Base, TimestampMixin


class TopicQuestionBank(TimestampMixin, Base):
    """Cached MCQ items for a topic, shared by the pre-test and the post-test.

    ``items`` includes each ``answer_index``; it must never be sent to the learner as-is.
    """

    __tablename__ = "topic_question_banks"

    id: Mapped[str] = mapped_column(primary_key=True, default=new_id)
    topic_id: Mapped[str] = mapped_column(nullable=False, unique=True, index=True)
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    n_items: Mapped[int] = mapped_column(nullable=False)
    model_provider: Mapped[str | None] = mapped_column(nullable=True)
    model_name: Mapped[str | None] = mapped_column(nullable=True)
