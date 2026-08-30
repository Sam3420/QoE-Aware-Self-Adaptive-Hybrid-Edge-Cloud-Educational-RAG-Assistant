from typing import Any

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class KnowledgeDocument(TimestampMixin, Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: __import__('uuid').uuid4().hex)
    resource_id: Mapped[str] = mapped_column(
        ForeignKey("educational_resources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="youtube", nullable=False)
    transcript_text: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    document_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict, nullable=False)

    resource: Mapped["EducationalResource"] = relationship(back_populates="knowledge_documents")
    chunks: Mapped[list["KnowledgeChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    indices: Mapped[list["KnowledgeIndex"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
