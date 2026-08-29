import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.errors import ImmutableRuntimeConfigurationError
from app.db.models import Interaction, RuntimeConfiguration
from app.db.repositories import LearningTraceRepository
from app.domain.models import (
    EducationalResourceCreate,
    InteractionCreate,
    LearnerProfileCreate,
    LearningSessionCreate,
)
from app.services.learning_trace_service import LearningTraceService


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with Session() as session:
        yield session


def test_persists_full_learning_trace_without_interaction_learner_duplication(db_session):
    service = LearningTraceService(LearningTraceRepository(db_session))

    learner = service.create_learner(LearnerProfileCreate(display_name="Sam"))
    runtime_config = service.create_runtime_configuration_snapshot(
        name="baseline",
        description="Local text baseline",
        configuration_data={"model_provider": "local", "response_length": "short"},
    )
    resource = service.create_resource(
        EducationalResourceCreate(
            source="manual",
            external_resource_id="algebra-001",
            title="Algebra basics",
            topic="algebra",
        )
    )
    learning_session = service.start_session(LearningSessionCreate(learner_id=learner.id))
    interaction = service.record_interaction(
        InteractionCreate(
            session_id=learning_session.id,
            runtime_configuration_id=runtime_config.id,
            educational_resource_id=resource.id,
            user_input="What is a variable?",
            assistant_output="A variable represents an unknown or changing value.",
            response_latency_ms=42,
        )
    )
    db_session.commit()

    assert "learner_id" not in Interaction.__table__.columns

    stored = LearningTraceRepository(db_session).get_interaction_with_trace(interaction.id)
    assert stored is not None
    assert stored.session.learner.id == learner.id
    assert stored.runtime_configuration.id == runtime_config.id
    assert stored.runtime_configuration.configuration_data["model_provider"] == "local"
    assert stored.educational_resource.id == resource.id


def test_invalid_session_learner_foreign_key_is_rejected(db_session):
    repository = LearningTraceRepository(db_session)

    with pytest.raises(IntegrityError):
        repository.create_learning_session(LearningSessionCreate(learner_id="missing-learner"))
        db_session.commit()


def test_runtime_configuration_updates_create_new_versions_for_history(db_session):
    service = LearningTraceService(LearningTraceRepository(db_session))

    learner = service.create_learner(LearnerProfileCreate(display_name="Learner"))
    session = service.start_session(LearningSessionCreate(learner_id=learner.id))
    first_config = service.create_runtime_configuration_snapshot(
        name="baseline",
        description="First snapshot",
        configuration_data={"model_provider": "local", "temperature": 0.2},
    )
    interaction = service.record_interaction(
        InteractionCreate(
            session_id=session.id,
            runtime_configuration_id=first_config.id,
            user_input="Explain fractions.",
        )
    )
    second_config = service.create_runtime_configuration_snapshot(
        name="baseline",
        description="Second snapshot",
        configuration_data={"model_provider": "local", "temperature": 0.4},
    )
    db_session.commit()

    stored_interaction = LearningTraceRepository(db_session).get_interaction_with_trace(interaction.id)
    assert stored_interaction.runtime_configuration.id == first_config.id
    assert stored_interaction.runtime_configuration.version == 1
    assert second_config.id != first_config.id
    assert second_config.version == 2


def test_referenced_runtime_configuration_snapshot_cannot_be_modified(db_session):
    service = LearningTraceService(LearningTraceRepository(db_session))

    learner = service.create_learner(LearnerProfileCreate(display_name="Learner"))
    session = service.start_session(LearningSessionCreate(learner_id=learner.id))
    runtime_config = service.create_runtime_configuration_snapshot(
        name="baseline",
        description="Immutable once referenced",
        configuration_data={"model_provider": "local"},
    )
    service.record_interaction(
        InteractionCreate(
            session_id=session.id,
            runtime_configuration_id=runtime_config.id,
            user_input="What is gravity?",
        )
    )
    db_session.commit()

    stored_config = db_session.get(RuntimeConfiguration, runtime_config.id)
    stored_config.description = "Mutated description"

    with pytest.raises(ImmutableRuntimeConfigurationError):
        db_session.commit()
