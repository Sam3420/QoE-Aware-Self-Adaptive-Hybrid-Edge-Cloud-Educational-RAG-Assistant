from typing import Any

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class KnowledgeIndex(TimestampMixin, Base):
    __tablename__ = "knowledge_indices"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: __import__('uuid').uuid4().hex)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index_name: Mapped[str] = mapped_column(String(128), nullable=False)
    index_path: Mapped[str] = mapped_column(nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_count: Mapped[int] = mapped_column(nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), default="ready", nullable=False)
    metadata: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    document: Mapped["KnowledgeDocument"] = relationship(back_populates="indices")
