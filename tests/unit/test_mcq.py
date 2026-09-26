import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.base import Base
from app.llm.types import LLMGenerateResult, LLMProviderError
from app.services.errors import MCQGenerationError, VectorStoreUnavailableError
from app.services.mcq import (
    competency_from_score,
    generate_mcqs,
    get_or_create_topic_items,
    normalize_answer,
    score_mcqs,
)


def make_items(n=5):
    return [
        {
            "question": f"Question {i}?",
            "options": [f"opt {i}-a", f"opt {i}-b", f"opt {i}-c", f"opt {i}-d"],
            "answer_index": i % 4,
            "concept": f"concept {i}",
        }
        for i in range(n)
    ]


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        output = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(output, Exception):
            raise output
        return LLMGenerateResult(text=output, model_provider="fake", model_name="fake-model")


class FakeRetriever:
    def __init__(self, contents=("Photosynthesis converts light into chemical energy.",)):
        self.contents = contents
        self.calls = []

    def retrieve_context(self, *, resource_id, query, top_k=None):
        self.calls.append({"resource_id": resource_id, "query": query, "top_k": top_k})
        return [{"chunk_id": str(i), "content": c, "score": 0.9} for i, c in enumerate(self.contents)]


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with Session() as session:
        yield session


# generate_mcqs

def test_generate_mcqs_valid_first_try():
    items = make_items()
    llm = FakeLLM(json.dumps(items))
    retriever = FakeRetriever()

    assert generate_mcqs("photosynthesis", 5, retriever=retriever, llm=llm) == items
    assert len(llm.requests) == 1
    assert retriever.calls[0]["resource_id"] == "photosynthesis"
    assert "Photosynthesis converts light" in llm.requests[0].prompt


def test_generate_mcqs_invalid_then_valid_retries_once():
    items = make_items()
    llm = FakeLLM("Sure! Here are your questions:", json.dumps(items))

    assert generate_mcqs("photosynthesis", 5, retriever=FakeRetriever(), llm=llm) == items
    assert len(llm.requests) == 2
    assert "rejected" in llm.requests[1].prompt


def test_generate_mcqs_always_invalid_raises_after_retry():
    llm = FakeLLM("not json")

    with pytest.raises(MCQGenerationError, match="after 2 attempts"):
        generate_mcqs("photosynthesis", 5, retriever=FakeRetriever(), llm=llm)
    assert len(llm.requests) == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda items: items[0].update(options=["a", "b", "c"]),
        lambda items: items[0].update(options=["a", "a", "b", "c"]),
        lambda items: items[0].update(answer_index=4),
        lambda items: items[0].pop("concept"),
        lambda items: items.pop(),
    ],
    ids=["three-options", "duplicate-options", "answer-out-of-range", "missing-concept", "wrong-count"],
)
def test_generate_mcqs_rejects_invalid_items(mutate):
    items = make_items()
    mutate(items)
    llm = FakeLLM(json.dumps(items))

    with pytest.raises(MCQGenerationError):
        generate_mcqs("photosynthesis", 5, retriever=FakeRetriever(), llm=llm)


def test_generate_mcqs_accepts_fenced_json():
    items = make_items()
    llm = FakeLLM(f"```json\n{json.dumps(items)}\n```")

    assert generate_mcqs("photosynthesis", 5, retriever=FakeRetriever(), llm=llm) == items


def test_generate_mcqs_without_retriever_fails_clearly():
    with pytest.raises(VectorStoreUnavailableError):
        generate_mcqs("photosynthesis", 5, llm=FakeLLM("[]"))


def test_generate_mcqs_with_no_retrieved_content_fails_clearly():
    llm = FakeLLM(json.dumps(make_items()))

    with pytest.raises(MCQGenerationError, match="No indexed content"):
        generate_mcqs("photosynthesis", 5, retriever=FakeRetriever(contents=()), llm=llm)
    assert llm.requests == []


def test_generate_mcqs_wraps_provider_errors():
    llm = FakeLLM(LLMProviderError("timed out"))

    with pytest.raises(MCQGenerationError, match="LLM provider failed"):
        generate_mcqs("photosynthesis", 5, retriever=FakeRetriever(), llm=llm)


# question bank cache

def test_get_or_create_topic_items_caches_per_topic(db_session):
    items = make_items()
    llm = FakeLLM(json.dumps(items))

    first = get_or_create_topic_items("photosynthesis", db=db_session, retriever=FakeRetriever(), llm=llm)
    second = get_or_create_topic_items("photosynthesis", db=db_session, retriever=FakeRetriever(), llm=llm)

    assert first == second == items
    assert len(llm.requests) == 1


# scoring

def test_score_mcqs_all_and_none_correct():
    items = make_items(4)  # answer indices 0, 1, 2, 3

    assert score_mcqs(items, ["A", "B", "C", "D"]) == 1.0
    assert score_mcqs(items, ["B", "C", "D", "A"]) == 0.0


def test_score_mcqs_mixes_letters_and_indices_case_insensitively():
    items = make_items(4)

    assert score_mcqs(items, ["a", 1, " c ", 3]) == 1.0


@pytest.mark.parametrize("bad", ["E", -1, 4, None, "AB", "", True, 1.0, "0"])
def test_invalid_answers_count_as_incorrect(bad):
    items = make_items(1)  # answer index 0

    assert normalize_answer(bad) is None
    assert score_mcqs(items, [bad]) == 0.0


def test_score_mcqs_handles_missing_and_extra_answers():
    items = make_items(4)

    assert score_mcqs(items, ["A", "B"]) == 0.5
    assert score_mcqs(items, []) == 0.0
    assert score_mcqs(items, ["A", "B", "C", "D", "A", "A"]) == 1.0


def test_score_mcqs_empty_items_is_zero():
    assert score_mcqs([], ["A"]) == 0.0


# competency

@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "beginner"),
        (0.39, "beginner"),
        (0.4, "intermediate"),
        (0.74, "intermediate"),
        (0.75, "advanced"),
        (1.0, "advanced"),
    ],
)
def test_competency_from_score_default_thresholds(score, expected):
    assert competency_from_score(score) == expected


def test_competency_thresholds_come_from_settings():
    settings = Settings(competency_beginner_max=0.5, competency_intermediate_max=0.9)

    assert competency_from_score(0.45, settings) == "beginner"
    assert competency_from_score(0.8, settings) == "intermediate"
