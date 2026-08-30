import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.domain.models import InteractionCreate, LearnerProfileCreate, LearningSessionCreate
from app.services.learning_trace_service import LearningTraceService
from app.services.monitoring_service import MonitoringService


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with Session() as session:
        yield session


def test_monitoring_service_records_and_scores_qoe(db_session):
    trace_service = LearningTraceService(LearningTraceRepository(db_session))
    learner = trace_service.create_learner(LearnerProfileCreate(display_name="Ava"))
    session = trace_service.start_session(LearningSessionCreate(learner_id=learner.id))
    runtime_config = trace_service.create_runtime_configuration_snapshot(
        name="baseline",
        description="monitoring baseline",
        configuration_data={"model_provider": "local"},
    )

    interaction = trace_service.record_interaction(
        InteractionCreate(
            session_id=session.id,
            runtime_configuration_id=runtime_config.id,
            user_input="What is photosynthesis?",
            assistant_output="Photosynthesis is how plants make food.",
            response_latency_ms=1200,
        )
    )
    db_session.commit()

    service = MonitoringService(repository=LearningTraceRepository(db_session))
    metrics = service.record_interaction_observations(interaction)
    score = service.evaluate_qoe(interaction)

    assert len(metrics) >= 2
    assert 0.0 <= score.score <= 100.0
    assert score.quality_label in {"excellent", "good", "fair", "poor"}
    assert score.session_id == session.id
