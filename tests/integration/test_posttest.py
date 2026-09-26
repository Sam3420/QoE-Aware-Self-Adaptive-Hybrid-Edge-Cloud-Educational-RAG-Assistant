import json

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.errors import ReferencedRecordNotFoundError
from app.db.repositories import AdaptationRepository, LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate, SessionFeedbackCreate
from app.services.errors import (
    LearnerNotFoundError,
    LearningSessionNotFoundError,
    QuestionBankNotFoundError,
)
from app.services.posttest import prepare_posttest, submit_posttest_and_feedback
from app.services.pretest import submit_pretest

ITEMS = [
    {
        "question": f"Question {i}?",
        "options": [f"opt {i}-a", f"opt {i}-b", f"opt {i}-c", f"opt {i}-d"],
        "answer_index": i % 4,
        "concept": f"concept {i}",
    }
    for i in range(5)
]  # correct letters: A, B, C, D, A


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with Session() as session:
        yield session


@pytest.fixture()
def learner_session(db_session):
    trace = LearningTraceRepository(db_session)
    learner = trace.create_learner(LearnerProfileCreate(display_name="Learner"))
    learning_session = trace.create_learning_session(LearningSessionCreate(learner_id=learner.id))
    return learner.id, learning_session.id


@pytest.fixture()
def bank(db_session):
    AdaptationRepository(db_session).save_question_bank(
        topic_id="photosynthesis", items=ITEMS, model_provider="fake", model_name="fake-model"
    )


# post-test

def test_prepare_posttest_reuses_bank_without_answers(db_session, bank):
    test_items = prepare_posttest("any-learner", "photosynthesis", db=db_session)

    assert [item["question"] for item in test_items] == [item["question"] for item in ITEMS]
    assert all(set(item) == {"question_no", "question", "options", "concept"} for item in test_items)
    assert "answer_index" not in json.dumps(test_items)


def test_prepare_posttest_without_bank_raises_and_generates_nothing(db_session):
    with pytest.raises(QuestionBankNotFoundError):
        prepare_posttest("any-learner", "photosynthesis", db=db_session)
    assert AdaptationRepository(db_session).get_question_bank("photosynthesis") is None


def test_submit_posttest_scores_persists_and_returns_plain_dict(db_session, learner_session, bank):
    student_id, session_id = learner_session

    result = submit_posttest_and_feedback(
        student_id, session_id, "photosynthesis", ["A", "B", "C", "D", 0], rating=4,
        comment="Clear answers, a bit slow.", db=db_session,
    )

    assert set(result) == {"assessment_id", "score", "competency", "rating", "comment"}
    assert result["score"] == 1.0
    assert result["competency"] == "advanced"
    assert result["rating"] == 4
    assert result["comment"] == "Clear answers, a bit slow."
    stored = AdaptationRepository(db_session).get_latest_assessment(student_id, "photosynthesis", "post")
    assert stored["id"] == result["assessment_id"]
    assert stored["type"] == "post"
    assert stored["items"] == ITEMS
    assert stored["answers"] == ["A", "B", "C", "D", 0]


def test_pre_and_post_assessments_stay_distinguishable(db_session, learner_session, bank):
    student_id, session_id = learner_session
    repository = AdaptationRepository(db_session)

    pre = submit_pretest(student_id, session_id, "photosynthesis", ["D", "D", "D", "D", "D"], db=db_session)
    post = submit_posttest_and_feedback(
        student_id, session_id, "photosynthesis", ["A", "B", "C", "A", "B"], db=db_session
    )

    assert pre["score"] == 0.2 and pre["competency"] == "beginner"
    assert post["score"] == 0.6 and post["competency"] == "intermediate"
    assert repository.get_latest_assessment(student_id, "photosynthesis", "pre")["id"] == pre["assessment_id"]
    assert repository.get_latest_assessment(student_id, "photosynthesis", "post")["id"] == post["assessment_id"]


def test_submit_posttest_without_bank_raises(db_session, learner_session):
    student_id, session_id = learner_session

    with pytest.raises(QuestionBankNotFoundError):
        submit_posttest_and_feedback(student_id, session_id, "photosynthesis", ["A"], db=db_session)
    assert AdaptationRepository(db_session).get_question_bank("photosynthesis") is None


def test_submit_posttest_validates_learner_and_session(db_session, learner_session, bank):
    student_id, session_id = learner_session
    other = LearningTraceRepository(db_session).create_learner(LearnerProfileCreate(display_name="Other"))

    with pytest.raises(LearnerNotFoundError):
        submit_posttest_and_feedback("student-001", session_id, "photosynthesis", ["A"], db=db_session)
    with pytest.raises(LearningSessionNotFoundError):
        submit_posttest_and_feedback(other.id, session_id, "photosynthesis", ["A"], db=db_session)
    with pytest.raises(LearningSessionNotFoundError):
        submit_posttest_and_feedback(student_id, "missing-session", "photosynthesis", ["A"], db=db_session)


# feedback

@pytest.mark.parametrize("rating", [1, 2, 3, 4, 5, None])
def test_valid_ratings_are_accepted(db_session, learner_session, bank, rating):
    student_id, session_id = learner_session

    result = submit_posttest_and_feedback(
        student_id, session_id, "photosynthesis", ["A"], rating=rating, db=db_session
    )

    assert result["rating"] == rating
    assert result["comment"] is None


@pytest.mark.parametrize("rating", [0, 6, -1, True, "4", 4.5])
def test_invalid_ratings_are_rejected_before_writing(db_session, learner_session, bank, rating):
    student_id, session_id = learner_session
    repository = AdaptationRepository(db_session)

    with pytest.raises(ValidationError):
        submit_posttest_and_feedback(
            student_id, session_id, "photosynthesis", ["A"], rating=rating, db=db_session
        )
    assert repository.get_latest_assessment(student_id, "photosynthesis", "post") is None
    assert repository.list_session_feedback(session_id) == []


def test_feedback_is_persisted_for_the_right_learner_and_session(db_session, learner_session, bank):
    student_id, session_id = learner_session

    submit_posttest_and_feedback(
        student_id, session_id, "photosynthesis", ["A"], rating=None, comment="Too fast.", db=db_session
    )

    [feedback] = AdaptationRepository(db_session).list_session_feedback(session_id)
    assert feedback["student_id"] == student_id
    assert feedback["session_id"] == session_id
    assert feedback["rating"] is None
    assert feedback["comment"] == "Too fast."


def test_feedback_repository_rejects_unknown_session(db_session, learner_session):
    student_id, _ = learner_session
    with pytest.raises(ReferencedRecordNotFoundError):
        AdaptationRepository(db_session).create_session_feedback(
            SessionFeedbackCreate(session_id="missing-session", student_id=student_id, rating=3)
        )
