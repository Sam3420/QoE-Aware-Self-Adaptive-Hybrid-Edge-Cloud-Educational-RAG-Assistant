from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db_session, get_knowledge_preparation_service, get_knowledge_retrieval_service
from app.domain.models import KnowledgePreparationRequest, KnowledgePreparationResponse, KnowledgeRetrievalRequest, KnowledgeRetrievalResponse
from app.services.errors import KnowledgePreparationError, ResourceNotFoundError
from app.services.knowledge_preparation_service import KnowledgePreparationService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService

router = APIRouter()


@router.post(
    "/resources/{resource_id}/knowledge/prepare",
    response_model=KnowledgePreparationResponse,
)
def prepare_knowledge(
    resource_id: str,
    db_session: Session = Depends(get_db_session),
    preparation_service: KnowledgePreparationService = Depends(get_knowledge_preparation_service),
) -> KnowledgePreparationResponse:
    # Build transcript chunks, embeddings, and the resource-specific FAISS index.
    try:
        result = preparation_service.prepare_resource(resource_id)
        db_session.commit()
        return KnowledgePreparationResponse(
            resource_id=result.resource_id,
            document_id=result.document_id,
            chunk_count=result.chunk_count,
            status=result.status,
        )
    except ResourceNotFoundError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Educational resource was not found.",
        ) from exc
    except KnowledgePreparationError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except SQLAlchemyError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Knowledge preparation could not be completed.",
        ) from exc


@router.post(
    "/resources/{resource_id}/knowledge/retrieve",
    response_model=KnowledgeRetrievalResponse,
)
def retrieve_knowledge(
    resource_id: str,
    request: KnowledgeRetrievalRequest,
    db_session: Session = Depends(get_db_session),
    retrieval_service: KnowledgeRetrievalService = Depends(get_knowledge_retrieval_service),
) -> KnowledgeRetrievalResponse:
    # Search prepared chunks and return database-backed text with similarity scores.
    try:
        hits = retrieval_service.retrieve_context(
            resource_id=resource_id,
            query=request.question,
            top_k=request.top_k,
        )
        db_session.commit()
        return KnowledgeRetrievalResponse(
            resource_id=resource_id,
            query=request.question,
            results=[
                {
                    "chunk_id": hit["chunk_id"],
                    "content": hit["content"],
                    "score": hit["score"],
                    "resource_id": hit["resource_id"],
                }
                for hit in hits
            ],
        )
    except SQLAlchemyError as exc:
        db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Knowledge retrieval failed.",
        ) from exc
