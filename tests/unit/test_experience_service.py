import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.domain.models import InteractionCreate, LearnerProfileCreate, LearningSessionCreate
from app.services.experience_service import ExperienceService
from app.services.learning_trace_service import LearningTraceService
from app.services.monitoring_service import MonitoringService


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with Session() as session:
        yield session


def test_experience_service_creates_persistent_experience_record(db_session):
    trace_service = LearningTraceService(LearningTraceRepository(db_session))
    learner = trace_service.create_learner(LearnerProfileCreate(display_name="Ava"))
    session = trace_service.start_session(LearningSessionCreate(learner_id=learner.id))
    runtime_config = trace_service.create_runtime_configuration_snapshot(
        name="baseline",
        description="phase10 baseline",
        configuration_data={"model_provider": "local", "temperature": 0.2},
    )
    interaction = trace_service.record_interaction(
        InteractionCreate(
            session_id=session.id,
            runtime_configuration_id=runtime_config.id,
            user_input="Explain photosynthesis.",
            assistant_output="Photosynthesis is how plants make food.",
            response_latency_ms=1200,
        )
    )
    db_session.commit()

    repository = LearningTraceRepository(db_session)
    monitoring_service = MonitoringService(repository=repository)
    monitoring_service.record_interaction_observations(interaction)
    qoe = monitoring_service.evaluate_qoe(interaction)
    db_session.commit()

    service = ExperienceService(repository=repository)
    experience = service.create_experience_for_interaction(interaction.id)

    assert experience is not None
    assert experience.session_id == session.id
    assert experience.learner_id == learner.id
    assert experience.runtime_configuration_id == runtime_config.id
    assert experience.qoe_score_id == repository.get_qoe_score_for_interaction(interaction.id).id
    assert experience.reward_score == pytest.approx(qoe.score)
    assert experience.state_snapshot["session_id"] == session.id
    assert experience.action_snapshot["action_type"] == "text_question"
    assert service.get_experiences_for_session(session.id)[0].id == experience.id
