"""Pre-test planning: which bank questions a new or returning learner gets, and bank top-ups."""
import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.services.pretest_planning import (
    REASON_NEW_LEARNER,
    REASON_RETEST,
    REASON_REVIEW_RIGHT,
    REASON_UNSEEN,
    plan_pretest,
)


def _bank(n, concepts=None):
    concepts = concepts or [f"c{i}" for i in range(n)]
    return [{"question": f"Q{i}?", "options": ["a", "b", "c", "d"], "answer_index": 0, "concept": concepts[i]}
            for i in range(n)]


def _assessment(bank, indices, answers, *, session="s1", type="pre", minutes_ago=10):
    return {"session_id": session, "type": type, "score": 0.0,
            "created_at": datetime(2026, 1, 1) - timedelta(minutes=minutes_ago),
            "items": [bank[i] for i in indices], "answers": answers}


def test_new_learner_gets_first_n_items():
    plan = plan_pretest(_bank(8), [], n=5, retest_max=2)
    assert plan.mode == "new" and plan.previous is None
    assert [p.bank_index for p in plan.items] == [0, 1, 2, 3, 4]
    assert all(p.reason == REASON_NEW_LEARNER for p in plan.items)


def test_returning_learner_retests_misses_then_prefers_unseen_weak_concepts():
    bank = _bank(10, ["cells", "tissues", "cells", "organs", "tissues",
                      "organs", "cells", "nuclei", "tissues", "organs"])
    # Last session: Q0 (cells) and Q1 (tissues) wrong, Q2..Q4 right.
    history = [_assessment(bank, [0, 1, 2, 3, 4], ["B", "B", "A", "A", "A"])]
    plan = plan_pretest(bank, history, n=5, retest_max=2)
    assert plan.mode == "returning"
    assert plan.weak_concepts == ["cells", "tissues"]
    assert [p.bank_index for p in plan.items[:2]] == [0, 1] and all(p.reason == REASON_RETEST for p in plan.items[:2])
    # Unseen items on weak concepts (6: cells, 8: tissues) come before other unseen ones (5: organs).
    assert [p.bank_index for p in plan.items[2:]] == [6, 8, 5]
    assert plan.items[2].reason == "New question on cells, which you found difficult."
    assert plan.items[4].reason == REASON_UNSEEN
    assert plan.unseen_needed == 0


def test_latest_attempt_decides_and_retests_are_capped():
    bank = _bank(6)
    old = _assessment(bank, [0, 1, 2], ["B", "B", "B"], session="s1", minutes_ago=60)  # all wrong
    new = _assessment(bank, [0, 1, 2], ["A", "B", "B"], session="s2", minutes_ago=5)  # Q0 now right
    plan = plan_pretest(bank, [new, old], n=5, retest_max=1)
    assert plan.items[0].bank_index in (1, 2) and plan.items[0].reason == REASON_RETEST
    assert sum(p.reason == REASON_RETEST for p in plan.items) == 1
    assert plan.previous["session_id"] == "s2"


def test_exhausted_bank_reports_shortfall_and_falls_back_to_review():
    bank = _bank(5)
    history = [_assessment(bank, [0, 1, 2, 3, 4], ["A", "A", "A", "A", "B"])]  # only Q4 wrong
    plan = plan_pretest(bank, history, n=5, retest_max=2)
    assert plan.unseen_needed == 4  # one retest, nothing unseen
    assert plan.items[0].bank_index == 4 and plan.items[0].reason == REASON_RETEST
    assert len(plan.items) == 5 and all(p.reason == REASON_REVIEW_RIGHT for p in plan.items[1:])


def test_history_matches_bank_by_question_text():
    bank = _bank(6)
    renamed = {**bank[3], "question": "  q3? "}  # same question, different case/spacing
    history = [{"session_id": "s", "type": "post", "score": 0.0, "created_at": datetime(2026, 1, 1),
                "items": [renamed], "answers": ["C"]}]
    plan = plan_pretest(bank, history, n=3, retest_max=2)
    assert plan.items[0].bank_index == 3 and plan.items[0].reason == REASON_RETEST


@pytest.fixture()
def db_session():
    import app.db.models  # noqa: F401  (registers every table on Base)

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine, autoflush=False, future=True)() as session:
        yield session


class _Retriever:
    def retrieve_context(self, *, resource_id, query, top_k=None):
        return [{"content": "Cells are the basic unit of life."}]


class _LLM:
    """Returns the given question texts as 5 valid MCQs."""

    def __init__(self, questions):
        self.questions = questions
        self.requests = []

    def generate(self, request):
        from app.llm.types import LLMGenerateResult

        self.requests.append(request)
        items = [{"question": q, "options": ["a", "b", "c", "d"], "answer_index": 0, "concept": "cells"}
                 for q in self.questions]
        return LLMGenerateResult(text=json.dumps(items), model_provider="fake", model_name="fake")


def test_extend_topic_items_skips_duplicates_and_focuses_prompt(db_session):
    from app.db.repositories import AdaptationRepository
    from app.services.mcq import extend_topic_items

    repo = AdaptationRepository(db_session)
    repo.save_question_bank(topic_id="cells", items=_bank(5), model_provider="x", model_name="y")
    llm = _LLM(["Q1?", "New A?", "new a?", "New B?", "New C?"])  # Q1 exists; "new a?" repeats "New A?"
    added = extend_topic_items("cells", focus_concepts=["nucleus"], db=db_session, retriever=_Retriever(), llm=llm)
    assert added == 3
    assert [i["question"] for i in repo.get_question_bank("cells")["items"][5:]] == ["New A?", "New B?", "New C?"]
    prompt = llm.requests[0].prompt
    assert "nucleus" in prompt and "Do not repeat" in prompt and "- Q0?" in prompt


def test_extend_topic_items_respects_bank_cap(db_session, monkeypatch):
    from app.core.config import get_settings
    from app.db.repositories import AdaptationRepository
    from app.services.mcq import extend_topic_items

    monkeypatch.setattr(get_settings(), "mcq_bank_max_items", 6)
    repo = AdaptationRepository(db_session)
    repo.save_question_bank(topic_id="cells", items=_bank(5), model_provider="x", model_name="y")
    added = extend_topic_items("cells", db=db_session, retriever=_Retriever(),
                               llm=_LLM(["A?", "B?", "C?", "D?", "E?"]))
    assert added == 1 and len(repo.get_question_bank("cells")["items"]) == 6


def test_concepts_match_regardless_of_case_and_spacing():
    bank = _bank(4, ["Nucleus", "cells", "NUCLEUS ", "nucleus"])
    history = [_assessment(bank, [0, 1], ["B", "A"])]  # "Nucleus" wrong
    plan = plan_pretest(bank, history, n=3, retest_max=1)
    assert plan.weak_concepts == ["Nucleus"]
    assert [p.bank_index for p in plan.items] == [0, 2, 3]
    assert plan.items[1].reason == "New question on NUCLEUS , which you found difficult."


def test_extend_topic_items_retries_when_every_question_is_a_duplicate(db_session):
    from app.db.repositories import AdaptationRepository
    from app.services.mcq import TOPUP_TEMPERATURE, extend_topic_items

    class _RepeatsFirst(_LLM):
        def generate(self, request):
            self.questions = [f"Q{i}?" for i in range(5)] if not self.requests else ["N1?", "N2?", "N3?", "N4?", "N5?"]
            return super().generate(request)

    repo = AdaptationRepository(db_session)
    repo.save_question_bank(topic_id="cells", items=_bank(5), model_provider="x", model_name="y")
    llm = _RepeatsFirst([])
    assert extend_topic_items("cells", db=db_session, retriever=_Retriever(), llm=llm) == 5
    assert len(llm.requests) == 2 and all(r.temperature == TOPUP_TEMPERATURE for r in llm.requests)


def test_explain_items_fills_only_missing_explanations(db_session):
    from app.db.repositories import AdaptationRepository
    from app.services.mcq import explain_items

    class _Explainer:
        def __init__(self):
            self.requests = []

        def generate(self, request):
            from app.llm.types import LLMGenerateResult

            self.requests.append(request)
            return LLMGenerateResult(text=json.dumps(["Because of X.", "Because of Y."]),
                                     model_provider="fake", model_name="fake")

    bank = _bank(4)
    bank[1]["explanation"] = "Already explained."
    repo = AdaptationRepository(db_session)
    repo.save_question_bank(topic_id="cells", items=bank, model_provider="x", model_name="y")
    llm = _Explainer()
    assert explain_items("cells", [2, 1, 0], db=db_session, retriever=_Retriever(), llm=llm) == 2
    items = repo.get_question_bank("cells")["items"]
    assert [i.get("explanation") for i in items] == ["Because of Y.", "Already explained.", "Because of X.", None]
    assert "Correct answer: A. a" in llm.requests[0].prompt and llm.requests[0].json_schema["minItems"] == 2
    assert explain_items("cells", [0, 1, 2], db=db_session, retriever=_Retriever(), llm=llm) == 0  # cached
    assert len(llm.requests) == 1


def test_review_posttest_reports_failed_explanations_without_hiding_answers(db_session):
    from app.db.repositories import AdaptationRepository
    from app.llm.types import LLMProviderError
    from app.services.posttest import review_posttest

    class _Down:
        def generate(self, request):
            raise LLMProviderError("offline")

    AdaptationRepository(db_session).save_question_bank(topic_id="cells", items=_bank(3), model_provider="x",
                                                        model_name="y")
    review = review_posttest("cells", ["A", "C", 0], item_ids=[2, 0, 1], pre_answers=["B", "A", "A"],
                             db=db_session, retriever=_Retriever(), llm=_Down())
    assert "offline" in review["note"]
    assert [(i["question"], i["answer"], i["post_answer"], i["post_correct"], i["pre_correct"], i["explanation"])
            for i in review["items"]] == [("Q2?", "A", "A", True, False, ""), ("Q0?", "A", "C", False, True, ""),
                                          ("Q1?", "A", "A", True, True, "")]
