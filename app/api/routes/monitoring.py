from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db_session, get_monitoring_service
from app.domain.models import QoEScoreResponse
from app.services.monitoring_service import MonitoringService

router = APIRouter()


@router.get(
    "/interactions/{interaction_id}/monitoring/qoe",
    response_model=QoEScoreResponse,
)
def get_interaction_qoe(
    interaction_id: str,
    db_session: Session = Depends(get_db_session),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
) -> QoEScoreResponse:
    from app.db.models import Interaction

    interaction = db_session.get(Interaction, interaction_id)
    if interaction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction was not found.",
        )

    qoe = monitoring_service.evaluate_qoe(interaction)
    db_session.commit()
    return QoEScoreResponse(
        interaction_id=qoe.interaction_id,
        session_id=qoe.session_id,
        score=qoe.score,
        quality_label=qoe.quality_label,
        latency_ms=qoe.latency_ms,
        model_provider=qoe.model_provider,
        model_name=qoe.model_name,
    )
