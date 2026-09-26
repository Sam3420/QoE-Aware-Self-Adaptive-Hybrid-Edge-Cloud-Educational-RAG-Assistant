import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.errors import ReferencedRecordNotFoundError
from app.db.models import Experience
from app.db.repositories import AdaptationRepository, LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from app.services.errors import LearningSessionNotFoundError
from app.services.experience import store_experience

CONTEXT = {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.6, "competency_num": 1, "last_qoe": 0.72}
EVALUATION = {"qoe_before": 0.72, "qoe_after": 0.81, "sre_before": 0.55, "sre_after": 0.64, "reward": 0.12}


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with Session() as session:
        AdaptationRepository(session).seed_policies()
        yield session


@pytest.fixture()
def learner_session(db_session):
    trace = LearningTraceRepository(db_session)
    learner = trace.create_learner(LearnerProfileCreate(display_name="Learner"))
    learning_session = trace.create_learning_session(LearningSessionCreate(learner_id=learner.id))
    return learner.id, learning_session.id


def count_experiences(db_session):
    return db_session.scalar(select(func.count(Experience.id)))


def test_stores_experience_with_given_policy_context_and_evaluation(db_session, learner_session):
    student_id, session_id = learner_session

    result = store_experience(student_id, session_id, "photosynthesis", "P3", CONTEXT, EVALUATION, db=db_session)

    assert result["policy_id"] == "P3"
    assert result["state"] == {**CONTEXT, "competency_num": 1.0}
    assert result["state_vector"] == [180.0, 0.35, 0.6, 1.0, 0.72]
    assert {key: result[key] for key in EVALUATION} == EVALUATION
    assert result["student_id"] == student_id and result["session_id"] == session_id
    assert [e["id"] for e in AdaptationRepository(db_session).get_experiences()] == [result["id"]]


def test_policy_id_is_stored_as_given_not_selected(db_session, learner_session):
    student_id, session_id = learner_session
    default_policy = AdaptationRepository(db_session).get_default_policy()["policy_id"]

    result = store_experience(student_id, session_id, "photosynthesis", "P4", CONTEXT, EVALUATION, db=db_session)

    assert default_policy == "P1"
    assert result["policy_id"] == "P4"


def test_extra_context_keys_are_ignored_and_missing_last_qoe_stays_none(db_session, learner_session):
    student_id, session_id = learner_session
    context = {key: value for key, value in CONTEXT.items() if key != "last_qoe"} | {"network": "3g"}

    result = store_experience(student_id, session_id, "photosynthesis", "P1", context, EVALUATION, db=db_session)

    assert result["state"]["last_qoe"] is None
    assert "network" not in result["state"]


def test_second_call_for_same_session_returns_existing_without_duplicate(db_session, learner_session):
    student_id, session_id = learner_session

    first = store_experience(student_id, session_id, "photosynthesis", "P1", CONTEXT, EVALUATION, db=db_session)
    second = store_experience(
        student_id, session_id, "photosynthesis", "P2", CONTEXT, {**EVALUATION, "reward": 0.5}, db=db_session
    )

    assert second["id"] == first["id"]
    assert second["policy_id"] == "P1"
    assert count_experiences(db_session) == 1


@pytest.mark.parametrize("missing", ["qoe_before", "qoe_after", "sre_before", "sre_after", "reward"])
def test_missing_evaluation_field_is_rejected(db_session, learner_session, missing):
    student_id, session_id = learner_session
    evaluation = {key: value for key, value in EVALUATION.items() if key != missing}

    with pytest.raises(ValueError, match=missing):
        store_experience(student_id, session_id, "photosynthesis", "P1", CONTEXT, evaluation, db=db_session)
    assert count_experiences(db_session) == 0


def test_none_evaluation_values_count_as_missing(db_session, learner_session):
    student_id, session_id = learner_session

    with pytest.raises(ValueError, match="sre_before, sre_after"):
        store_experience(
            student_id, session_id, "photosynthesis", "P1", CONTEXT,
            {**EVALUATION, "sre_before": None, "sre_after": None}, db=db_session,
        )


@pytest.mark.parametrize("missing", ["rtt_ms", "cpu", "mem", "competency_num"])
def test_missing_context_measurement_is_rejected(db_session, learner_session, missing):
    student_id, session_id = learner_session
    context = {key: value for key, value in CONTEXT.items() if key != missing}

    with pytest.raises(ValueError, match=missing):
        store_experience(student_id, session_id, "photosynthesis", "P1", context, EVALUATION, db=db_session)
    assert count_experiences(db_session) == 0


def test_unknown_policy_and_foreign_session_are_rejected(db_session, learner_session):
    student_id, session_id = learner_session
    other = LearningTraceRepository(db_session).create_learner(LearnerProfileCreate(display_name="Other"))

    with pytest.raises(ReferencedRecordNotFoundError):
        store_experience(student_id, session_id, "photosynthesis", "P9", CONTEXT, EVALUATION, db=db_session)
    with pytest.raises(LearningSessionNotFoundError):
        store_experience(other.id, session_id, "photosynthesis", "P1", CONTEXT, EVALUATION, db=db_session)


def test_get_experience_for_session(db_session, learner_session):
    student_id, session_id = learner_session
    repository = AdaptationRepository(db_session)
    assert repository.get_experience_for_session(session_id) is None

    stored = store_experience(student_id, session_id, "photosynthesis", "P1", CONTEXT, EVALUATION, db=db_session)

    assert repository.get_experience_for_session(session_id)["id"] == stored["id"]
