"""Parallel post-test: pre-test items paired (in parallel_test_items) with a post-test question are re-tested with it;
the rest are repeated. Pairs never reach the pre-test: not in its response, its stored attempt or the bank items."""
import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import ParallelTestItem
from app.db.repositories import AdaptationRepository, LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from app.services.posttest import (
    posttest_items,
    prepare_posttest,
    review_posttest,
    submit_posttest_and_feedback,
)
from app.services.pretest import plan_pretest, submit_pretest
from app.services.pretest_planning import REASON_RETEST, plan_pretest as plan_items

TOPIC = "life_processes"


def pre_item(i, difficulty="easy", concept_id=None):
    concept_id = concept_id or f"c{i}"
    return {"id": f"LP-PRE-0{i}", "question": f"Pre question {i}?", "options": [f"p{i}a", f"p{i}b", f"p{i}c", f"p{i}d"],
            "answer_index": 0, "concept": f"Concept {i}", "concept_id": concept_id, "difficulty": difficulty,
            "explanation": f"Pre explanation {i}."}


def post_item(i, difficulty="easy", concept_id=None):
    return {"id": f"LP-POST-0{i}", "question": f"Post question {i}?",
            "options": [f"SECRET-{i}a", f"SECRET-{i}b", f"SECRET-{i}c", f"SECRET-{i}d"], "answer_index": 2,
            "concept": f"Concept {i}", "concept_id": concept_id or f"c{i}", "difficulty": difficulty,
            "explanation": f"Post explanation {i}."}


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine, autoflush=False, future=True)() as session:
        yield session


@pytest.fixture()
def learner_session(db_session):
    trace = LearningTraceRepository(db_session)
    learner = trace.create_learner(LearnerProfileCreate(display_name="Learner"))
    return learner.id, trace.create_learning_session(LearningSessionCreate(learner_id=learner.id)).id


def save(db_session, pre, pairs, topic=TOPIC):
    """The bank (pre-test items only) and the pairs, as the content loader stores them."""
    AdaptationRepository(db_session).save_question_bank(topic_id=topic, items=pre, model_provider="content-package",
                                                        model_name="test")
    for i, item in pairs.items():
        db_session.add(ParallelTestItem(id=item["id"], topic_id=topic, pre_item_id=f"LP-PRE-0{i}",
                                        item={k: v for k, v in item.items() if k != "id"}, source="test"))
    db_session.flush()


# --- selection ---------------------------------------------------------------------------------------------------

def test_pairs_replace_pre_test_items_and_unpaired_or_mismatched_items_repeat():
    pairs = {"LP-PRE-01": post_item(1), "LP-PRE-03": post_item(3, difficulty="hard"),
             "LP-PRE-04": post_item(4, concept_id="other")}
    items = [pre_item(1), pre_item(2), pre_item(3), pre_item(4)]
    served = posttest_items(TOPIC, items, pairs)
    assert (served[0]["question"], served[0]["form"], served[0]["parallel_to"]) == \
        ("Post question 1?", "parallel", "LP-PRE-01")
    assert served[1] == items[1]  # no pair: unchanged and unmarked
    assert [s["question"] for s in served[2:]] == ["Pre question 3?", "Pre question 4?"]  # difficulty / concept differ
    incomplete = {k: v for k, v in post_item(1).items() if k != "answer_index"}
    assert posttest_items(TOPIC, [pre_item(1)], {"LP-PRE-01": incomplete})[0]["question"] == "Pre question 1?"


def test_learner_sees_paired_questions_without_answers(db_session):
    save(db_session, [pre_item(i) for i in range(1, 5)], {i: post_item(i) for i in range(1, 4)})
    shown = prepare_posttest("learner", TOPIC, item_ids=[3, 0, 1], db=db_session)
    assert [s["question"] for s in shown] == ["Pre question 4?", "Post question 1?", "Post question 2?"]
    assert [s["item_id"] for s in shown] == [3, 0, 1]  # the pre-test bank positions the node passes back
    assert "answer_index" not in json.dumps(shown)


def test_post_test_is_scored_and_stored_against_the_paired_questions(db_session, learner_session):
    student, session = learner_session
    save(db_session, [pre_item(1), pre_item(2), pre_item(3)], {1: post_item(1), 2: post_item(2)})
    submit_pretest(student, session, TOPIC, ["A", "B", "A"], item_ids=[0, 1, 2], db=db_session)
    result = submit_posttest_and_feedback(student, session, TOPIC, ["C", "A", "A"], 4, None,  # paired keys C, C
                                          item_ids=[0, 1, 2], db=db_session)
    assert result["score"] == pytest.approx(2 / 3)
    stored = AdaptationRepository(db_session).get_latest_assessment(student, TOPIC, "post")
    assert [i["question"] for i in stored["items"]] == ["Post question 1?", "Post question 2?", "Pre question 3?"]
    assert [i.get("form") for i in stored["items"]] == ["parallel", "parallel", None]
    assert stored["items"][0]["parallel_to"] == "LP-PRE-01"


def test_review_compares_with_the_matching_pre_test_question(db_session):
    save(db_session, [pre_item(1), pre_item(2), pre_item(3)], {1: post_item(1), 2: post_item(2)})
    review = review_posttest(TOPIC, ["C", "A", "B"], item_ids=[0, 1, 2], pre_answers=["A", "B", "A"], db=db_session)
    first, second, third = review["items"]
    assert (first["form"], first["parallel_to"], first["answer"], first["post_correct"]) == \
        ("parallel", "LP-PRE-01", "C", True)
    assert first["pre_answer"] is None and first["pre_correct"] is True  # pre-test item 1: A was its key
    assert second["pre_correct"] is False and second["post_correct"] is False
    assert (third["form"], third["pre_answer"], third["pre_correct"]) == ("repeat", "A", True)
    assert first["explanation"] == "Post explanation 1." and review["note"] is None  # no LLM call needed


def test_topics_without_pairs_keep_the_old_behaviour(db_session):
    save(db_session, [pre_item(i) for i in range(1, 4)], {})
    shown = prepare_posttest("learner", TOPIC, item_ids=[0, 1, 2], db=db_session)
    assert [s["question"] for s in shown] == ["Pre question 1?", "Pre question 2?", "Pre question 3?"]


def test_missing_pairs_table_is_treated_as_no_pairs():
    engine = create_engine("sqlite:///:memory:", future=True)
    tables = [t for name, t in Base.metadata.tables.items() if name != "parallel_test_items"]
    Base.metadata.create_all(bind=engine, tables=tables)  # a database before migration 0004
    with sessionmaker(bind=engine, future=True)() as db:
        AdaptationRepository(db).save_question_bank(topic_id=TOPIC, items=[pre_item(1)], model_provider="x",
                                                    model_name="y")
        assert prepare_posttest("learner", TOPIC, item_ids=[0], db=db)[0]["question"] == "Pre question 1?"


# --- isolation: post-test questions never reach the pre-test ------------------------------------------------------

def secrets(pairs):
    return [p["question"] for p in pairs.values()] + [o for p in pairs.values() for o in p["options"]]


def test_pre_test_response_and_stored_attempt_contain_no_post_test_content(db_session, learner_session):
    student, session = learner_session
    pairs = {i: post_item(i) for i in range(1, 6)}
    save(db_session, [pre_item(i) for i in range(1, 6)], pairs)

    plan = plan_pretest(student, TOPIC, db=db_session)  # what the learner is shown
    shown = json.dumps(plan, default=str)
    assert not [s for s in secrets(pairs) if s in shown] and "answer_index" not in shown
    assert "LP-POST" not in shown and "parallel" not in shown

    submit_pretest(student, session, TOPIC, ["A"] * 5, item_ids=[i["item_id"] for i in plan["items"]], db=db_session)
    stored = json.dumps(AdaptationRepository(db_session).get_latest_assessment(student, TOPIC, "pre"), default=str)
    assert not [s for s in secrets(pairs) if s in stored]  # neither paired questions nor their options/keys
    assert "LP-POST" not in stored and "parallel" not in stored

    bank = json.dumps(AdaptationRepository(db_session).get_question_bank(TOPIC), default=str)
    assert not [s for s in secrets(pairs) if s in bank]  # the bank row itself holds pre-test items only


# --- planning: paired post-test results count for their pre-test item ---------------------------------------------

def test_missed_paired_question_retests_its_pre_test_item():
    bank = [pre_item(i) for i in range(1, 7)]
    now = datetime(2026, 10, 25)
    paired_missed = {**post_item(2), "form": "parallel", "parallel_to": "LP-PRE-02"}
    paired_right = {**post_item(3), "form": "parallel", "parallel_to": "LP-PRE-03"}
    post = {"session_id": "s1", "type": "post", "score": 0.5, "created_at": now, "items": [paired_missed, paired_right],
            "answers": ["A", "C"]}  # paired keys are C: item 2 missed, item 3 right
    pre = {"session_id": "s1", "type": "pre", "score": 1.0, "created_at": now - timedelta(minutes=30),
           "items": [bank[1], bank[2]], "answers": ["A", "A"]}  # both right in the pre-test
    plan = plan_items(bank, [post, pre], n=5, retest_max=2)
    assert plan.mode == "returning" and plan.weak_concepts == ["Concept 2"]  # the newest attempt (post) decides
    assert plan.items[0].bank_index == 1 and plan.items[0].reason == REASON_RETEST
    assert 2 not in [p.bank_index for p in plan.items if p.reason == REASON_RETEST]


def test_unpaired_post_items_are_still_matched_by_question_text():
    bank = [pre_item(i) for i in range(1, 4)]
    post = {"session_id": "s", "type": "post", "score": 0.0, "created_at": datetime(2026, 10, 25),
            "items": [bank[0]], "answers": ["B"]}  # repeated pre-test item, missed
    plan = plan_items(bank, [post], n=3, retest_max=1)
    assert plan.items[0].bank_index == 0 and plan.items[0].reason == REASON_RETEST


# --- end to end through the agent graph ---------------------------------------------------------------------------

def test_full_session_asks_paired_questions_only_in_the_post_test(runner, learner, services):
    from tests.conftest import pass_pretest

    services.app.ensure_ready()
    pairs = {i: post_item(i) for i in range(1, 6)}
    with services.app._db() as db:
        save(db, [pre_item(i) for i in range(1, 6)], pairs, topic="photosynthesis")
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pre_shown = runner.get_state(tid)["interrupt"]
    assert [i["question"] for i in pre_shown["items"]] == [f"Pre question {i}?" for i in range(1, 6)]
    assert not [s for s in secrets(pairs) if s in json.dumps(pre_shown, default=str)]
    pass_pretest(runner, tid, ["A"] * 5)  # every pre-test key is A
    assert not [s for s in secrets(pairs) if s in json.dumps(runner.get_state(tid)["values"]["test_items"])]
    post = runner.resume(tid, {"end": True})["interrupt"]
    assert [i["question"] for i in post["items"]] == [f"Post question {i}?" for i in range(1, 6)]
    final = runner.resume(tid, {"answers": ["C"] * 5, "rating": 5})["final_state"]  # every paired key is C
    assert final["pre_score"] == 1.0 and final["post_score"] == 1.0
    assert {r["form"] for r in final["post_review"]["items"]} == {"parallel"}


def test_pretest_size_defaults_to_mcq_n_items_and_is_configurable(db_session, learner_session, monkeypatch):
    from app.core.config import get_settings

    student, _ = learner_session
    save(db_session, [pre_item(i % 9 + 1) | {"id": f"LP-PRE-{i:02d}", "question": f"Q{i}?"} for i in range(1, 16)],
         {})
    assert get_settings().pretest_items is None and get_settings().mcq_n_items == 5
    assert len(plan_pretest(student, TOPIC, db=db_session)["items"]) == 5  # unchanged default
    monkeypatch.setattr(get_settings(), "pretest_items", 15)
    assert len(plan_pretest(student, TOPIC, db=db_session)["items"]) == 15  # experiment setting: whole bank
