from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.db.repositories import LearningTraceRepository
from app.llm.groq_provider import make_groq_provider
from app.llm.provider import LLMProvider
from app.llm.types import LLMProviderError
from app.services.assistant_service import AssistantService


def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    try:
        return make_groq_provider(settings)
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant provider is not configured.",
        ) from exc


def get_assistant_service(
    db_session: Session = Depends(get_db_session),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    settings: Settings = Depends(get_settings),
) -> AssistantService:
    return AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=llm_provider,
        settings=settings,
    )
