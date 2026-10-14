import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.repositories import AdaptationRepository, LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from app.llm.types import LLMGenerateResult
from app.services.errors import (
    LearnerNotFoundError,
    LearningSessionNotFoundError,
    QuestionBankNotFoundError,
)
from app.services.pretest import prepare_pretest, submit_pretest

ITEMS = [
    {
        "question": f"Question {i}?",
        "options": [f"opt {i}-a", f"opt {i}-b", f"opt {i}-c", f"opt {i}-d"],
        "answer_index": i % 4,
        "concept": f"concept {i}",
    }
    for i in range(5)
]  # correct letters: A, B, C, D, A


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return LLMGenerateResult(text=json.dumps(ITEMS), model_provider="fake", model_name="fake-model")


class FakeRetriever:
    def retrieve_context(self, *, resource_id, query, top_k=None):
        return [{"chunk_id": "c1", "content": "Plants make food from light.", "score": 0.9}]


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


def test_prepare_then_submit_round_trip(db_session, learner_session):
    student_id, session_id = learner_session
    llm = FakeLLM()

    test_items = prepare_pretest(
        student_id, "photosynthesis", db=db_session, retriever=FakeRetriever(), llm=llm
    )

    assert [item["question_no"] for item in test_items] == [1, 2, 3, 4, 5]
    assert all(set(item) == {"question_no", "item_id", "question", "options", "concept", "reason"}
               for item in test_items)
    assert [item["item_id"] for item in test_items] == [0, 1, 2, 3, 4]
    assert "answer_index" not in json.dumps(test_items)

    result = submit_pretest(
        student_id, session_id, "photosynthesis", ["A", "B", "C", "A", "B"], db=db_session
    )

    assert result["score"] == 0.6
    assert result["competency"] == "intermediate"
    repository = AdaptationRepository(db_session)
    assert repository.has_pre_score(student_id, "photosynthesis") is True
    stored = repository.get_latest_assessment(student_id, "photosynthesis", "pre")
    assert stored["id"] == result["assessment_id"]
    assert stored["answers"] == ["A", "B", "C", "A", "B"]
    assert stored["items"] == ITEMS


def test_prepare_reuses_topic_bank_across_learners(db_session, learner_session):
    llm = FakeLLM()

    first = prepare_pretest("learner-1", "photosynthesis", db=db_session, retriever=FakeRetriever(), llm=llm)
    second = prepare_pretest("learner-2", "photosynthesis", db=db_session, retriever=FakeRetriever(), llm=llm)

    assert first == second
    assert llm.calls == 1


def test_submit_without_question_bank_fails(db_session, learner_session):
    student_id, session_id = learner_session

    with pytest.raises(QuestionBankNotFoundError):
        submit_pretest(student_id, session_id, "photosynthesis", ["A"], db=db_session)


def test_submit_rejects_unknown_learner_and_foreign_session(db_session, learner_session):
    student_id, session_id = learner_session
    other = LearningTraceRepository(db_session).create_learner(LearnerProfileCreate(display_name="Other"))
    prepare_pretest(student_id, "photosynthesis", db=db_session, retriever=FakeRetriever(), llm=FakeLLM())

    with pytest.raises(LearnerNotFoundError):
        submit_pretest("student-001", session_id, "photosynthesis", ["A"], db=db_session)
    with pytest.raises(LearningSessionNotFoundError):
        submit_pretest(other.id, session_id, "photosynthesis", ["A"], db=db_session)
    with pytest.raises(LearningSessionNotFoundError):
        submit_pretest(student_id, "missing-session", "photosynthesis", ["A"], db=db_session)


def test_submit_rejects_non_list_answers(db_session, learner_session):
    student_id, session_id = learner_session

    with pytest.raises(TypeError):
        submit_pretest(student_id, session_id, "photosynthesis", "ABCDA", db=db_session)
