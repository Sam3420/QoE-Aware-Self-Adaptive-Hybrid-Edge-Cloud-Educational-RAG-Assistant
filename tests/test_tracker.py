"""tracker.py contract, migration 0002, concepts/extract.py mapping and scripts/backfill_tracker.py (in-memory DB,
fixture vocabulary in a temp dir, fake LLM: nothing touches data/app.db, Groq or knowledge_prep/concepts/)."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

import knowledge_prep.concepts as concepts_mod
import tracker
from app.db.base import Base
from app.db.migrations import run_migrations
from app.db.models import Assessment
from app.db.repositories import ConceptTrackerRepository, LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from concepts.extract import map_questions, with_map
from knowledge_prep.concepts import QuestionMapping, Vocabulary, load_vocabulary, save_vocabulary
from scripts.backfill_tracker import backfill

TOPIC = "fixture_topic"
QUESTIONS = ["What does auxin do?", "Where is auxin made?", "What do gibberellins do?", "What is a reflex arc?"]


def vocabulary(**kw) -> Vocabulary:
    concepts = [{"id": f"concept_{i}", "name": f"Concept {i}", "description": f"What concept {i} is about.",
                 "aliases": [f"alias {i}"], "evidence": "Auxin is made at the shoot tip."} for i in range(1, 7)]
    return Vocabulary(topic_id=TOPIC, generated_by="test", generated_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
                      concepts=concepts, **kw)


QMAP = {"0": QuestionMapping(concept_id="concept_1", question=QUESTIONS[0], confidence=0.9),
        "1": QuestionMapping(concept_id="concept_1", question=QUESTIONS[1], confidence=0.85),
        "2": QuestionMapping(concept_id="concept_2", question=QUESTIONS[2], confidence=0.9),
        "3": QuestionMapping(concept_id="concept_3", question=QUESTIONS[3], confidence=0.4, needs_review=True,
                             note="low confidence 0.40")}


@pytest.fixture
def concepts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(concepts_mod, "CONCEPTS_DIR", tmp_path)
    save_vocabulary(vocabulary(question_concept_map=QMAP))
    return tmp_path


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", future=True)
    import app.db.models  # noqa: F401  (registers every table)

    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine, autoflush=False, future=True)() as session:
        yield session


@pytest.fixture
def learner(db):
    return LearningTraceRepository(db).create_learner(LearnerProfileCreate(display_name="Asha")).id


# --- mastery ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("results, mastery", [
    ([], 0.5),  # no attempts: unknown, not weak
    ([True, True, True], 4 / 5),
    ([False, False], 1 / 4),
    ([True, False, True, False, True], 4 / 7),
])
def test_mastery_formula(db, learner, concepts_dir, results, mastery):
    for i, correct in enumerate(results):
        tracker.record_attempt(learner, TOPIC, "concept_1", correct, "formative", source_ref=f"check:{i}", db=db)
    state = tracker.get_mastery(learner, TOPIC, db=db)["concept_1"]
    assert state.mastery == pytest.approx(mastery)
    assert (state.attempts, state.correct) == (len(results), sum(results))


# --- contract --------------------------------------------------------------------------------------------------------

def test_get_concepts_and_get_mastery_cover_the_whole_vocabulary(db, learner, concepts_dir):
    assert [c.id for c in tracker.get_concepts(TOPIC)] == [f"concept_{i}" for i in range(1, 7)]
    tracker.record_attempt(learner, TOPIC, "concept_2", True, "pretest", source_ref="a1:0", db=db)
    states = tracker.get_mastery(learner, TOPIC, db=db)
    assert list(states) == [f"concept_{i}" for i in range(1, 7)]  # every concept, vocabulary order
    assert states["concept_2"].attempts == 1 and states["concept_2"].mastery == pytest.approx(2 / 3)
    assert states["concept_5"] == tracker.ConceptState(mastery=0.5, attempts=0, correct=0, covered=False, asked=0)


def test_unknown_concept_topic_source_and_kind_raise_without_writing(db, learner, concepts_dir):
    with pytest.raises(tracker.UnknownConceptError, match="concept_99"):
        tracker.record_attempt(learner, TOPIC, "concept_99", True, "pretest", db=db)
    with pytest.raises(tracker.UnknownConceptError):
        tracker.mark_covered(learner, TOPIC, "concept_99", "video", db=db)
    with pytest.raises(ValueError, match="source must be one of"):
        tracker.record_attempt(learner, TOPIC, "concept_1", True, "quiz", db=db)
    with pytest.raises(ValueError, match="kind must be one of"):
        tracker.mark_covered(learner, TOPIC, "concept_1", "read", db=db)
    with pytest.raises(tracker.VocabularyNotFoundError):
        tracker.get_mastery(learner, "no_such_topic", db=db)
    assert ConceptTrackerRepository(db).get_mastery(learner, TOPIC) == []  # nothing was created


def test_mark_covered_video_and_asked(db, learner, concepts_dir):
    tracker.mark_covered(learner, TOPIC, "concept_1", "video", source_ref="s1:vidA", db=db)
    tracker.mark_covered(learner, TOPIC, "concept_1", "video", source_ref="s1:vidA", db=db)  # same video again
    tracker.mark_covered(learner, TOPIC, "concept_2", "asked", source_ref="s1:q1", db=db)
    tracker.mark_covered(learner, TOPIC, "concept_2", "asked", source_ref="s1:q2", db=db)
    tracker.mark_covered(learner, TOPIC, "concept_2", "asked", source_ref="s1:q2", db=db)  # node ran twice
    states = tracker.get_mastery(learner, TOPIC, db=db)
    assert states["concept_1"].covered is True and states["concept_1"].asked == 0
    assert states["concept_2"].covered is False and states["concept_2"].asked == 2
    assert states["concept_1"].mastery == 0.5  # exposure is not evidence of learning


def test_same_answer_counts_once_whichever_path_records_it(db, learner, concepts_dir):
    tracker.record_attempt(learner, TOPIC, "concept_1", True, "pretest", source_ref="a1:0", db=db)
    tracker.record_attempt(learner, TOPIC, "concept_1", True, "pretest", source_ref="a1:0", db=db)  # again
    tracker.record_attempt(learner, TOPIC, "concept_2", True, "pretest", source_ref="a1:0", db=db)  # other concept
    # the live test recording (agent/concept_tracking.py) writes the same answer under the stored source name
    ConceptTrackerRepository(db).record_attempt(learner, TOPIC, "concept_1", True, source="pre-test",
                                                source_ref="a1:0")
    states = tracker.get_mastery(learner, TOPIC, db=db)
    assert states["concept_1"].attempts == 1 and states["concept_2"].attempts == 0
    row = db.execute(text("select last_source from learner_topic_strength where concept_id='concept_1'")).scalar_one()
    assert row == "pre-test"


def test_without_source_ref_every_call_counts(db, learner, concepts_dir):
    tracker.record_attempt(learner, TOPIC, "concept_1", False, "formative", db=db)
    tracker.record_attempt(learner, TOPIC, "concept_1", False, "formative", db=db)
    assert tracker.get_mastery(learner, TOPIC, db=db)["concept_1"].attempts == 2


# --- migration 0002 --------------------------------------------------------------------------------------------------

def test_migration_0002_adds_columns_to_a_0001_database():
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:  # the table as 0001 created it, before covered_video/asked_count/last_source
        conn.execute(text("CREATE TABLE learner_topic_strength (id VARCHAR PRIMARY KEY, student_id VARCHAR, "
                          "topic_id VARCHAR, concept_id VARCHAR(64), attempts INTEGER, correct INTEGER, mastery FLOAT, "
                          "covered_count INTEGER)"))
        conn.execute(text("INSERT INTO learner_topic_strength VALUES ('r1','s','t','c',2,1,0.5,0)"))
        conn.execute(text("CREATE TABLE schema_migrations (id VARCHAR(64) PRIMARY KEY, description VARCHAR(200), "
                          "applied_at DATETIME)"))
        conn.execute(text("INSERT INTO schema_migrations VALUES ('0001_concept_tracker', 'x', '2026-10-20')"))
    assert run_migrations(engine) == ["0002_tracker_coverage", "0003_formative_pool", "0004_parallel_test_items"]
    assert run_migrations(engine) == []  # once only
    columns = {c["name"] for c in inspect(engine).get_columns("learner_topic_strength")}
    assert {"covered_video", "asked_count", "last_source"} <= columns
    with engine.connect() as conn:
        assert conn.execute(text("select covered_video, asked_count, last_source from learner_topic_strength")
                            ).one() == (0, 0, None)  # existing rows get the defaults


def test_fresh_database_runs_both_migrations():
    engine = create_engine("sqlite:///:memory:", future=True)
    assert run_migrations(engine) == ["0001_concept_tracker", "0002_tracker_coverage", "0003_formative_pool",
                                     "0004_parallel_test_items"]


# --- vocabulary file keeps the map -----------------------------------------------------------------------------------

def test_question_map_survives_load_and_save_and_rejects_unknown_ids(concepts_dir):
    loaded = load_vocabulary(TOPIC)
    assert loaded.question_concept_map["3"].needs_review is True
    save_vocabulary(loaded)  # e.g. the approve command re-saving the file
    assert load_vocabulary(TOPIC).question_concept_map == loaded.question_concept_map
    with pytest.raises(ValueError, match="unknown concept ids"):
        vocabulary(question_concept_map={"0": QuestionMapping(concept_id="nope", question="q", confidence=1)})


def test_map_questions_flags_low_confidence_and_label_disagreement():
    items = [{"question": QUESTIONS[0], "options": ["a", "b"], "answer_index": 0, "concept": "Concept 1"},
             {"question": QUESTIONS[1], "options": ["a", "b"], "answer_index": 1, "concept": "alias 2"},
             {"question": QUESTIONS[2], "options": ["a", "b"], "answer_index": 0, "concept": "vague label"}]
    replies = [json.dumps({"mappings": [{"question_id": 0, "concept_id": "concept_1", "confidence": 0.95}]}),  # partial
               json.dumps({"mappings": [{"question_id": 0, "concept_id": "concept_1", "confidence": 0.95},
                                        {"question_id": 1, "concept_id": "concept_1", "confidence": 0.9},
                                        {"question_id": 2, "concept_id": "concept_4", "confidence": 0.5}]})]
    prompts = []

    def generate(request):
        prompts.append(request.prompt)
        return NS(text=replies[len(prompts) - 1])

    mapping = map_questions(vocabulary(), items, NS(generate=generate))
    assert len(prompts) == 2 and "previous reply was rejected" in prompts[1]  # incomplete reply retried
    assert mapping["0"].needs_review is False
    assert mapping["1"].needs_review and "label names concept_2" in mapping["1"].note
    assert mapping["2"].needs_review and "low confidence" in mapping["2"].note
    mapped = with_map(vocabulary(status="approved", reviewed_by="T"), mapping)
    assert mapped.status == "approved"  # concepts and status untouched
    assert {c.id: c.source_question_ids for c in mapped.concepts}["concept_1"] == [0, 1]


# --- backfill --------------------------------------------------------------------------------------------------------

def _store_test(db, learner, type_, answers, questions=QUESTIONS, topic=TOPIC):
    session_id = LearningTraceRepository(db).create_learning_session(LearningSessionCreate(learner_id=learner)).id
    items = [{"question": q, "options": ["A0", "A1", "A2", "A3"], "answer_index": 0, "concept": "x"}
             for q in questions]
    db.add(Assessment(session_id=session_id, student_id=learner, topic_id=topic, type=type_, items=items,
                      answers=answers, score=0.5, competency="beginner"))
    db.flush()


def test_backfill_replays_tests_and_is_safe_to_rerun(db, learner, concepts_dir):
    _store_test(db, learner, "pre", ["A", "B", "A", "A"])  # q0 right, q1 wrong (concept_1), q2 right, q3 flagged
    _store_test(db, learner, "post", ["A", "A", "A", "A", "A"],
                questions=[*QUESTIONS, "A question added to the bank later?"])
    _store_test(db, learner, "pre", ["A"], questions=["Old topic question"], topic="old_topic")
    first = backfill(db)
    assert (len(first.learners), first.recorded, first.already) == (1, 6, 0)
    assert first.skipped == {"mapping needs review": 2, "no mapping for the question": 1,
                             "no vocabulary/map (old_topic)": 1}
    before = tracker.get_mastery(learner, TOPIC, db=db)
    assert (before["concept_1"].attempts, before["concept_1"].correct) == (4, 3)
    assert before["concept_2"].attempts == 2 and before["concept_3"].attempts == 0  # needs_review skipped

    second = backfill(db)
    assert (second.recorded, second.already, second.skipped) == (0, 6, first.skipped)
    assert tracker.get_mastery(learner, TOPIC, db=db) == before  # same counts after a re-run

    third = backfill(db, include_unreviewed=True)
    assert third.recorded == 2 and tracker.get_mastery(learner, TOPIC, db=db)["concept_3"].attempts == 2
