from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.errors import ReferencedRecordNotFoundError
from app.db.init_db import create_tables
from app.db.models import Experience
from app.db.repositories import AdaptationRepository, LearningTraceRepository
from app.domain.enums import AssessmentType, CompetencyLevel
from app.domain.models import (
    AssessmentCreate,
    ExperienceCreate,
    ExperienceState,
    LearnerProfileCreate,
    LearningSessionCreate,
)


@pytest.fixture()
def engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    create_tables(engine)
    return engine


@pytest.fixture()
def db_session(engine):
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with Session() as session:
        yield session


@pytest.fixture()
def repository(db_session):
    return AdaptationRepository(db_session)


@pytest.fixture()
def learner_session(db_session):
    trace = LearningTraceRepository(db_session)
    learner = trace.create_learner(LearnerProfileCreate(display_name="Learner"))
    learning_session = trace.create_learning_session(LearningSessionCreate(learner_id=learner.id))
    return learner.id, learning_session.id


def make_assessment(student_id, session_id, *, type=AssessmentType.PRE, score=0.6, topic_id="algebra"):
    return AssessmentCreate(
        session_id=session_id,
        student_id=student_id,
        topic_id=topic_id,
        type=type,
        items=[{"question": "Q", "options": ["a", "b", "c", "d"], "answer_index": 0, "concept": "c"}],
        answers=["A"],
        score=score,
        competency=CompetencyLevel.INTERMEDIATE,
    )


def make_experience(student_id, session_id, *, policy_id="P1", qoe_after=0.7, last_qoe=None):
    return ExperienceCreate(
        student_id=student_id,
        session_id=session_id,
        topic_id="algebra",
        state=ExperienceState(rtt_ms=120.0, cpu=0.4, mem=0.5, competency_num=1, last_qoe=last_qoe),
        policy_id=policy_id,
        qoe_before=0.5,
        qoe_after=qoe_after,
        sre_before=0.6,
        sre_after=0.8,
        reward=0.25,
    )


def test_creates_new_tables_alongside_existing_ones(engine):
    tables = set(inspect(engine).get_table_names())

    assert {"assessments", "policies", "experiences", "topic_question_banks"} <= tables
    assert {"assessment_results", "runtime_configurations"} <= tables


def test_seed_policies_is_idempotent_with_one_default(repository):
    assert repository.seed_policies() == 5
    assert repository.seed_policies() == 0

    policies = repository.list_policies()
    assert [policy["policy_id"] for policy in policies] == ["P1", "P2", "P3", "P4", "P5"]
    assert sum(policy["is_default"] for policy in policies) == 1
    assert {(p["top_k"], p["max_retries"], p["route"]) for p in policies} == {
        (3, 0, "local"),
        (5, 1, "local"),
        (3, 0, "cloud"),
        (5, 1, "cloud"),
        (5, 1, "escalate"),
    }


def test_get_policy_and_default_policy(repository):
    assert repository.get_default_policy() is None

    repository.seed_policies()

    assert repository.get_policy("P3")["route"] == "cloud"
    assert repository.get_policy("P9") is None
    assert repository.get_default_policy()["policy_id"] == "P1"


def test_assessment_crud_and_has_pre_score(repository, learner_session):
    student_id, session_id = learner_session
    assert repository.has_pre_score(student_id, "algebra") is False
    assert repository.get_latest_assessment(student_id, "algebra", "pre") is None

    repository.create_assessment(make_assessment(student_id, session_id, type=AssessmentType.POST))
    assert repository.has_pre_score(student_id, "algebra") is False

    created = repository.create_assessment(make_assessment(student_id, session_id, score=0.8))
    latest = repository.get_latest_assessment(student_id, "algebra", AssessmentType.PRE)

    assert latest["id"] == created["id"]
    assert latest["score"] == 0.8
    assert latest["competency"] == "intermediate"
    assert latest["answers"] == ["A"]
    assert repository.has_pre_score(student_id, "algebra") is True
    assert repository.has_pre_score(student_id, "geometry") is False


def test_assessment_score_must_be_fraction(learner_session):
    student_id, session_id = learner_session
    with pytest.raises(ValueError):
        make_assessment(student_id, session_id, score=80)


def test_assessment_rejects_unknown_learner_or_session(repository, learner_session):
    student_id, session_id = learner_session

    with pytest.raises(ReferencedRecordNotFoundError):
        repository.create_assessment(make_assessment("student-001", session_id))
    with pytest.raises(ReferencedRecordNotFoundError):
        repository.create_assessment(make_assessment(student_id, "missing-session"))


def test_experience_insert_and_fetch_shape(repository, db_session, learner_session):
    student_id, session_id = learner_session
    repository.seed_policies()

    first = repository.insert_experience(make_experience(student_id, session_id, policy_id="P1"))
    second = repository.insert_experience(make_experience(student_id, session_id, policy_id="P4", last_qoe=0.7))
    db_session.get(Experience, first["id"]).created_at = datetime.now(timezone.utc) - timedelta(hours=1)
    db_session.flush()

    experiences = repository.get_experiences()
    assert [e["id"] for e in experiences] == [second["id"], first["id"]]
    newest = experiences[0]
    assert newest["state"] == {"rtt_ms": 120.0, "cpu": 0.4, "mem": 0.5, "competency_num": 1.0, "last_qoe": 0.7}
    assert newest["state_vector"] == [120.0, 0.4, 0.5, 1.0, 0.7]
    assert newest["policy_id"] == "P4"
    assert newest["reward"] == 0.25
    assert experiences[1]["state_vector"][-1] is None

    assert len(repository.get_experiences(limit=1)) == 1


def test_experience_rejects_unknown_policy(repository, learner_session):
    student_id, session_id = learner_session
    repository.seed_policies()

    with pytest.raises(ReferencedRecordNotFoundError):
        repository.insert_experience(make_experience(student_id, session_id, policy_id="P9"))


def test_get_last_qoe(repository, db_session, learner_session):
    student_id, session_id = learner_session
    repository.seed_policies()
    assert repository.get_last_qoe(student_id) is None

    older = repository.insert_experience(make_experience(student_id, session_id, qoe_after=0.4))
    repository.insert_experience(make_experience(student_id, session_id, qoe_after=0.9))
    db_session.get(Experience, older["id"]).created_at = datetime.now(timezone.utc) - timedelta(hours=1)
    db_session.flush()

    assert repository.get_last_qoe(student_id) == 0.9
    assert repository.get_last_qoe("someone-else") is None


def test_question_bank_round_trip(repository):
    assert repository.get_question_bank("algebra") is None

    items = [{"question": "Q", "options": ["a", "b", "c", "d"], "answer_index": 2, "concept": "c"}]
    repository.save_question_bank(topic_id="algebra", items=items, model_provider="hf", model_name="llama")

    bank = repository.get_question_bank("algebra")
    assert bank["items"] == items
    assert bank["n_items"] == 1
    assert bank["model_name"] == "llama"


def test_base_metadata_contains_new_tables():
    assert {"assessments", "policies", "experiences", "topic_question_banks"} <= set(Base.metadata.tables)
