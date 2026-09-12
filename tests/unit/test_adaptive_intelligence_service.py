from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.repositories import LearningTraceRepository
from app.domain.models import InteractionCreate, LearnerProfileCreate, LearningSessionCreate
from app.services.adaptive_intelligence_service import AdaptiveIntelligenceService
from app.services.learning_trace_service import LearningTraceService


def _create_learner_and_session(db_session):
    trace_service = LearningTraceService(LearningTraceRepository(db_session))
    learner = trace_service.create_learner(LearnerProfileCreate(display_name="Ada"))
    session = trace_service.start_session(LearningSessionCreate(learner_id=learner.id))
    return learner, session


def test_adaptive_service_exposes_allowed_actions():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with Session() as session:
        repository = LearningTraceRepository(session)
        service = AdaptiveIntelligenceService(repository=repository)

        actions = service.get_allowed_actions()

        assert actions == [
            "CHANGE_MODEL",
            "CHANGE_RETRIEVAL_K",
            "CHANGE_RESPONSE_DETAIL",
            "CHANGE_RESPONSE_STYLE",
            "CHANGE_LANGUAGE",
            "CHANGE_PROVIDER",
        ]


def test_adaptive_service_recommends_provider_change_for_low_local_qoe():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with Session() as session:
        repository = LearningTraceRepository(session)
        trace_service = LearningTraceService(repository)
        learner, session_record = _create_learner_and_session(session)

        runtime_configuration = trace_service.create_runtime_configuration_snapshot(
            name="local-baseline",
            description="phase11 baseline",
            configuration_data={"provider": "local", "model_id": "local-model"},
        )

        interaction = trace_service.record_interaction(
            InteractionCreate(
                session_id=session_record.id,
                runtime_configuration_id=runtime_configuration.id,
                user_input="Explain photosynthesis.",
                assistant_output="Short answer.",
                response_latency_ms=1200,
                model_provider="local",
                model_name="local-model",
                status="completed",
            )
        )

        repository.create_qoe_score(
            interaction_id=interaction.id,
            session_id=session_record.id,
            score=35.0,
            quality_label="poor",
            latency_ms=1200,
            model_provider="local",
            model_name="local-model",
            details={"completion_score": 25.0, "latency_score": 0.0, "content_score": 100.0},
        )
        repository.create_experience(
            learner_id=learner.id,
            session_id=session_record.id,
            interaction_id=interaction.id,
            runtime_configuration_id=runtime_configuration.id,
            qoe_score_id=repository.get_qoe_score_for_interaction(interaction.id).id,
            resource_id=None,
            topic=None,
            state_snapshot={"session_id": session_record.id},
            action_snapshot={"action_type": "text_question", "model_provider": "local"},
            configuration_snapshot={"provider": "local"},
            qoe_outcome={"score": 35.0, "quality_label": "poor"},
            reward_score=35.0,
            reward_details={"reward_basis": "qoe_score"},
            outcome_label="poor",
            performance_summary={"status": "completed"},
        )

        service = AdaptiveIntelligenceService(repository=repository)
        decision = service.select_action(
            learner_id=learner.id,
            current_qoe_score=35.0,
            current_runtime_configuration={"provider": "local", "model_id": "local-model"},
        )

        assert decision.action == "CHANGE_PROVIDER"
        assert "local" in decision.reason.lower()


def test_adaptive_service_transforms_runtime_configuration_for_action():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with Session() as session:
        repository = LearningTraceRepository(session)
        service = AdaptiveIntelligenceService(repository=repository)

        transformed = service.transform_configuration(
            action="CHANGE_PROVIDER",
            current_runtime_configuration={"provider": "local", "model_id": "local-model"},
        )

        assert transformed["provider"] == "huggingface"
        assert transformed["model_id"] == "local-model"
