from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.db.repositories import LearningTraceRepository
from app.llm.huggingface_provider import HuggingFaceInferenceProvider
from app.llm.provider import LLMProvider
from app.llm.types import LLMProviderError
from app.knowledge.embedding_provider import SentenceTransformerEmbeddingProvider
from app.knowledge.faiss_index_store import FaissIndexStore
from app.llm.hybrid_provider import HybridLLMProvider, LocalLLMProvider
from app.llm.stt_provider import PassthroughSTTProvider
from app.llm.tts_provider import PassthroughTTSProvider
from app.services.assistant_service import AssistantService
from app.services.crag_service import RetrievalQualityService
from app.services.knowledge_preparation_service import KnowledgePreparationService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService
from app.services.resource_recommendation_service import ResourceRecommendationService
from app.services.speech_service import SpeechService
from app.youtube.data_api_provider import YouTubeDataAPIProvider


class StaticTranscriptProvider:
    def get_transcript_for_resource(
        self,
        *,
        resource_id: str | None = None,
        resource: object | None = None,
        **_: object,
    ) -> str:
        return ""


def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    token = settings.hf_token.get_secret_value() if settings.hf_token else ""
    try:
        cloud_provider = HuggingFaceInferenceProvider(
            token=token,
            inference_url=settings.hf_inference_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        local_provider = LocalLLMProvider(model_id=settings.local_llm_model_id) if settings.local_llm_enabled else None
        return HybridLLMProvider(
            cloud_provider=cloud_provider,
            local_provider=local_provider,
            settings=settings,
        )
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant provider is not configured.",
        ) from exc


def get_resource_recommendation_service(
    db_session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> ResourceRecommendationService:
    return ResourceRecommendationService(
        repository=LearningTraceRepository(db_session),
        youtube_provider=YouTubeDataAPIProvider(settings=settings),
    )


def get_knowledge_preparation_service(
    db_session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> KnowledgePreparationService:
    return KnowledgePreparationService(
        repository=LearningTraceRepository(db_session),
        transcript_provider=StaticTranscriptProvider(),
        embedding_provider=SentenceTransformerEmbeddingProvider(
            model_name=settings.knowledge_embedding_model,
        ),
        vector_index_store=FaissIndexStore(base_dir=settings.faiss_storage_path),
        settings=settings,
    )


def get_knowledge_retrieval_service(
    db_session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> KnowledgeRetrievalService:
    return KnowledgeRetrievalService(
        repository=LearningTraceRepository(db_session),
        embedding_provider=SentenceTransformerEmbeddingProvider(
            model_name=settings.knowledge_embedding_model,
        ),
        vector_index_store=FaissIndexStore(base_dir=settings.faiss_storage_path),
        settings=settings,
    )


def get_assistant_service(
    db_session: Session = Depends(get_db_session),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    settings: Settings = Depends(get_settings),
    knowledge_retrieval_service: KnowledgeRetrievalService = Depends(get_knowledge_retrieval_service),
) -> AssistantService:
    return AssistantService(
        repository=LearningTraceRepository(db_session),
        llm_provider=llm_provider,
        settings=settings,
        knowledge_retrieval_service=knowledge_retrieval_service,
        retrieval_quality_service=RetrievalQualityService(min_score=settings.crag_quality_min_score),
    )


def get_speech_service(
    db_session: Session = Depends(get_db_session),
    assistant_service: AssistantService = Depends(get_assistant_service),
    settings: Settings = Depends(get_settings),
) -> SpeechService:
    stt_provider = PassthroughSTTProvider(
        provider_name=settings.stt_provider_name,
        enabled=settings.stt_provider_enabled,
    )
    tts_provider = PassthroughTTSProvider(
        provider_name=settings.tts_provider_name,
        enabled=settings.tts_provider_enabled,
    )
    return SpeechService(
        assistant_service=assistant_service,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
    )
