"""Concept vocabulary, matcher, tracker repository (LearnerTopicStrength), migration and backfill."""
from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

import knowledge_prep.concepts as concepts_mod
from agent.concept_tracking import AssessmentTracker
from app.db.base import Base
from app.db.errors import ReferencedRecordNotFoundError
from app.db.migrations import MIGRATIONS, applied, run_migrations
from app.db.repositories import ConceptTrackerRepository, LearningTraceRepository
from app.db.repositories.concept_tracker import laplace_mastery
from app.domain.models import LearnerProfileCreate
from knowledge_prep.concepts import (ConceptMatcher, Vocabulary, evidence_problems, load_vocabulary,
                                     overlap_problems, save_vocabulary)
from tests.conftest import ANSWERS, pass_pretest

NOTES = "Auxin is made at the shoot tip. Gibberellins help the stem grow. Cytokinins promote cell division."


def concept(i, **kw):
    return {"id": f"c{i}", "name": f"Concept {i}", "description": f"What concept {i} is about.",
            "aliases": [], "evidence": "Auxin is made at the shoot tip.", **kw}


def vocab(n=6, topic_id="t1", **kw):
    return Vocabulary(topic_id=topic_id, generated_by="test", generated_at=datetime(2026, 10, 20, tzinfo=timezone.utc),
                      concepts=[concept(i) for i in range(1, n + 1)], **kw)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", future=True)
    import app.db.models  # noqa: F401  (registers every table)

    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine, autoflush=False, future=True)() as session:
        yield session


@pytest.fixture
def student(db):
    return LearningTraceRepository(db).create_learner(LearnerProfileCreate(display_name="Asha")).id


# --- mastery formula and repository ------------------------------------------------------------------

def test_laplace_mastery():
    assert laplace_mastery(0, 0) == 0.5 and laplace_mastery(1, 1) == pytest.approx(2 / 3)
    assert laplace_mastery(0, 2) == 0.25 and laplace_mastery(9, 10) == pytest.approx(10 / 12)
    with pytest.raises(ValueError):
        laplace_mastery(3, 2)


def test_record_attempt_updates_mastery_and_is_idempotent(db, student):
    repo = ConceptTrackerRepository(db)
    prior = repo.get_mastery(student, "t1", "c1")
    assert (prior["mastery"], prior["attempts"], prior["tracked"]) == (0.5, 0, False)  # prior, nothing stored
    row = repo.record_attempt(student, "t1", "c1", True, source="pre-test", source_ref="a1:0")
    assert (row["attempts"], row["correct"], row["mastery"]) == (1, 1, pytest.approx(2 / 3))
    row = repo.record_attempt(student, "t1", "c1", False, source="post-test", source_ref="a2:0")
    assert (row["attempts"], row["correct"], row["mastery"]) == (2, 1, 0.5)
    again = repo.record_attempt(student, "t1", "c1", False, source="post-test", source_ref="a2:0")  # same origin
    assert (again["attempts"], again["correct"]) == (2, 1)
    assert [e["source"] for e in repo.list_events(student, "t1")] == ["pre-test", "post-test"]


def test_mark_covered_counts_exposure_without_changing_mastery(db, student):
    repo = ConceptTrackerRepository(db)
    repo.record_attempt(student, "t1", "c1", True, source="pre-test", source_ref="a1:0")
    row = repo.mark_covered(student, "t1", "c1", source="question", source_ref="s1:q1")
    row = repo.mark_covered(student, "t1", "c1", source="question", source_ref="s1:q1")  # idempotent
    assert (row["covered_count"], row["covered"], row["mastery"]) == (1, True, pytest.approx(2 / 3))
    assert row["first_covered_at"] is not None and row["first_covered_at"] == row["last_covered_at"]
    repo.mark_covered(student, "t1", "c2", source="video", source_ref="yt_x")
    assert [r["concept_id"] for r in repo.get_mastery(student, "t1")] == ["c1", "c2"]
    assert repo.get_mastery(student, "t1", "c2")["attempts"] == 0


def test_unknown_learner_is_rejected(db):
    with pytest.raises(ReferencedRecordNotFoundError):
        ConceptTrackerRepository(db).record_attempt("nobody", "t1", "c1", True, source="pre-test", source_ref="x")


# --- migration ---------------------------------------------------------------------------------------------

def test_migration_creates_tracker_tables_once():
    engine = create_engine("sqlite:///:memory:", future=True)
    import app.db.models  # noqa: F401

    Base.metadata.tables["learner_profiles"].create(bind=engine)  # an "old" database without the tracker
    assert run_migrations(engine) == [m.id for m in MIGRATIONS]  # every pending migration, in order
    tables = set(inspect(engine).get_table_names())
    assert {"learner_topic_strength", "concept_events", "schema_migrations"} <= tables
    assert run_migrations(engine) == [] and applied(engine) == [m.id for m in MIGRATIONS]


def test_migration_records_itself_when_create_all_already_made_the_tables():
    engine = create_engine("sqlite:///:memory:", future=True)
    import app.db.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    assert run_migrations(engine) == [m.id for m in MIGRATIONS] and run_migrations(engine) == []


# --- vocabulary ---------------------------------------------------------------------------------------------

def test_vocabulary_validation():
    assert len(vocab().concepts) == 6
    for bad in (dict(n=5), dict(n=13)):
        with pytest.raises(ValidationError, match="6-12 concepts"):
            vocab(**bad)
    with pytest.raises(ValidationError, match="unique"):
        Vocabulary(topic_id="t", generated_by="x", generated_at=datetime.now(timezone.utc),
                   concepts=[concept(1)] * 6)
    with pytest.raises(ValidationError, match="reviewed_by"):
        vocab(status="approved")
    with pytest.raises(ValidationError, match="snake_case"):
        vocab().model_copy().__class__.model_validate({**vocab().model_dump(), "concepts": [
            concept(1, id="Bad Id"), *[concept(i) for i in range(2, 7)]]})


def test_evidence_and_overlap_checks():
    v = vocab()
    assert evidence_problems(v, NOTES) == []
    v.concepts[0].evidence = "Plants dream at night."
    assert evidence_problems(v, NOTES) == ["c1: evidence not found in the lesson notes"]
    v.concepts[1].aliases = ["cerebellum"]
    v.concepts[2].aliases = ["Cerebellum"]
    assert overlap_problems(v) == ["'cerebellum' is used by c2, c3"]


def test_save_and_load_roundtrip(tmp_path):
    save_vocabulary(vocab(), tmp_path)
    assert load_vocabulary("t1", tmp_path) == vocab() and load_vocabulary("missing", tmp_path) is None


# --- matcher ------------------------------------------------------------------------------------------------

class KeywordEmbedder:
    """Deterministic embedder: one dimension per keyword, so similarities are predictable."""

    KEYS = ["hormone", "feedback", "nerve", "reflex", "plant"]

    def _vec(self, text):
        v = np.array([text.lower().count(k) for k in self.KEYS], dtype="float32") + 0.01
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def matcher_vocab():
    names = [("animal_hormones", "Animal Hormones", "Hormone glands in animals.", ["endocrine"]),
             ("hormonal_feedback", "Hormonal Feedback", "Feedback control of hormone levels.", []),
             ("reflex_arc", "Reflex Arc", "Nerve pathway of a reflex.", ["reflex action"]),
             ("neuron", "Neuron", "Structure of a nerve cell.", []),
             ("tropisms", "Plant Tropisms", "Directional plant growth.", []),
             ("auxin", "Auxin", "A plant hormone made at the tip.", [])]
    return Vocabulary(topic_id="t", generated_by="x", generated_at=datetime.now(timezone.utc), concepts=[
        {"id": i, "name": n, "description": d, "aliases": a, "evidence": "Auxin is made at the shoot tip."}
        for i, n, d, a in names])


def test_matcher_exact_alias_embedding_and_ambiguity():
    m = ConceptMatcher(matcher_vocab(), KeywordEmbedder(), min_similarity=0.5, min_margin=0.03)
    assert m.match("Reflex Arc").method == "exact" and m.match("reflex-arc").concept_id == "reflex_arc"
    assert m.match("Reflex action") .concept_id == "reflex_arc" and m.match("ENDOCRINE").method == "alias"
    hit = m.match("nerve cells", context="What does a nerve carry?")
    assert hit.concept_id == "neuron" and hit.method == "embedding" and hit.margin > 0
    class Between:  # the query sits exactly between the first two concepts
        def embed_documents(self, texts):
            return np.eye(len(texts), dtype="float32")

        def embed_query(self, text):
            v = np.zeros(6, dtype="float32")
            v[:2] = 1 / np.sqrt(2)
            return v

    ambiguous = ConceptMatcher(matcher_vocab(), Between(), min_similarity=0.5, min_margin=0.03).match("hormone stuff")
    assert ambiguous.concept_id is None and ambiguous.margin == 0
    assert ambiguous.reason.startswith("ambiguous:") and ambiguous.reason.endswith("by only 0.000")
    far = ConceptMatcher(matcher_vocab(), KeywordEmbedder(), min_similarity=0.99).match("plant nerve")
    assert far.concept_id is None and "similarity" in far.reason
    assert ConceptMatcher(matcher_vocab()).match("something else").reason == "no embedder for a non-exact label"


# --- recording assessments (backfill and live) ---------------------------------------------------------------

def test_assessment_tracker_records_items_once_and_reports_unmapped(db, student, tmp_path, monkeypatch):
    monkeypatch.setattr(concepts_mod, "CONCEPTS_DIR", tmp_path)
    save_vocabulary(matcher_vocab().model_copy(update={"topic_id": "t1"}), tmp_path)
    item = lambda concept_label, key=0: {"question": "Q?", "options": ["a", "b", "c", "d"], "answer_index": key,
                                         "concept": concept_label}
    assessment = {"id": "a1", "student_id": student, "topic_id": "t1", "type": "pre", "created_at": None,
                  "items": [item("Reflex Arc"), item("endocrine", 1), item("Photosynthesis")],
                  "answers": ["A", "A", "B"]}
    tracker, repo = AssessmentTracker(), ConceptTrackerRepository(db)
    report = tracker.record(repo, assessment)
    assert (report.items, report.recorded, report.already) == (3, 2, 0)
    assert dict(report.unmapped) == {"Photosynthesis": 1} and report.vocabulary_status == {"t1": "draft"}
    assert repo.get_mastery(student, "t1", "reflex_arc")["correct"] == 1  # A was right
    assert repo.get_mastery(student, "t1", "animal_hormones")["correct"] == 0  # A was wrong (key B)
    again = tracker.record(repo, assessment)
    assert (again.recorded, again.already) == (0, 2)  # backfill re-run: nothing new
    other = tracker.record(repo, {**assessment, "topic_id": "class8_topic"})
    assert dict(other.skipped_topics) == {"class8_topic": 1}


def test_pre_and_post_tests_feed_the_tracker_during_a_session(runner, learner, services, tmp_path, monkeypatch):
    monkeypatch.setattr(concepts_mod, "CONCEPTS_DIR", tmp_path)
    v = Vocabulary(topic_id="photosynthesis", generated_by="x", generated_at=datetime.now(timezone.utc), concepts=[
        concept(i, id=f"concept_{i}", name=f"concept {i}") for i in range(1, 7)])  # fake MCQs use "concept N"
    save_vocabulary(v, tmp_path)
    tid = runner.start_session(learner, "photosynthesis")["thread_id"]
    pass_pretest(runner, tid, ["A", "A", "A", "A", "A"])  # key A,B,C,D,A: concepts 1 and 5 right
    runner.resume(tid, {"query": "How do plants convert sunlight into chemical energy?"})
    runner.resume(tid, {"end": True})
    runner.resume(tid, {"answers": ANSWERS, "rating": 4})  # all right
    with services.app._db() as db:
        rows = {r["concept_id"]: r for r in ConceptTrackerRepository(db).get_mastery(learner, "photosynthesis")}
    assert set(rows) == {f"concept_{i}" for i in range(1, 6)}
    assert (rows["concept_1"]["attempts"], rows["concept_1"]["correct"]) == (2, 2)  # right in both tests
    assert (rows["concept_2"]["attempts"], rows["concept_2"]["correct"]) == (2, 1)  # wrong, then right
    assert rows["concept_2"]["mastery"] == 0.5
    assert not [e for e in runner.get_state(tid)["values"]["session_errors"] if "concept tracker" in e]
