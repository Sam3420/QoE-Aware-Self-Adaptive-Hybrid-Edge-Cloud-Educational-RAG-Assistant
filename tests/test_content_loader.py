"""scripts/load_content_package.py on a temporary SQLite file: dry run writes nothing, conflicts block --apply,
--apply backs up first and preserves learner data, and re-runs are idempotent. Uses the committed Life Processes
package files; validation (which needs the local, gitignored chapter text) is stubbed as passing."""
import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

import scripts.load_content_package as loader
from app.db.base import Base
from app.db.models import Assessment, FormativeItem, ParallelTestItem, TopicQuestionBank
from app.db.repositories import LearningTraceRepository
from app.domain.models import LearnerProfileCreate, LearningSessionCreate
from config import PROJECT_ROOT

TOPIC = "life_processes"
OLD_ONLY = {"formative_items", "parallel_test_items"}  # tables added by migrations 0003/0004


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A database like the real one before these migrations, with a learner, an attempt and a generated bank."""
    db = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db.as_posix()}", future=True)
    Base.metadata.create_all(bind=engine, tables=[t for n, t in Base.metadata.tables.items() if n not in OLD_ONLY])
    with Session(engine) as s, s.begin():
        trace = LearningTraceRepository(s)
        learner = trace.create_learner(LearnerProfileCreate(display_name="Asha"))
        session = trace.create_learning_session(LearningSessionCreate(learner_id=learner.id))
        s.add(Assessment(session_id=session.id, student_id=learner.id, topic_id="heredity", type="pre",
                         items=[{"question": "Old?"}], answers=["A"], score=1.0, competency="advanced"))
        s.add(TopicQuestionBank(topic_id=TOPIC, items=[{"question": "Generated?", "options": ["a", "b", "c", "d"],
                                                       "answer_index": 0, "concept": "x"}], n_items=1,
                                model_provider="cloud", model_name="openai/gpt-oss-120b"))
    engine.dispose()
    vocab_dir = tmp_path / "concepts"
    vocab_dir.mkdir()
    # an earlier, "approved" vocabulary: the package's concepts with excretion_in_plants still called
    # life_processes_overview (independent of the real knowledge_prep/concepts file, which a load replaces)
    old = json.loads((PROJECT_ROOT / "knowledge_prep" / "packages" / TOPIC / "concepts.json").read_text(encoding="utf-8"))
    old["concepts"] = [{k: v for k, v in c.items() if k != "source"}
                       | ({"id": "life_processes_overview"} if c["id"] == "excretion_in_plants" else {})
                       for c in old["concepts"]]
    old |= {"status": "approved", "reviewed_by": "Your Name", "reviewed_at": "2026-10-20T13:04:34+00:00"}
    (vocab_dir / f"{TOPIC}.json").write_text(json.dumps(old, indent=2), encoding="utf-8")
    monkeypatch.setattr(loader, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(loader, "BACKUPS", tmp_path / "backups")
    monkeypatch.setattr(loader, "VOCAB_DIR", vocab_dir)
    monkeypatch.setattr(loader, "validate", lambda topic_id: {"valid": True, "evidence_checked": True, "errors": [],
                                                             "warnings": ["w"]})
    return {"db": db, "url": f"sqlite:///{db.as_posix()}", "vocab": vocab_dir / f"{TOPIC}.json", "tmp": tmp_path}


def run(env, **kw):
    return loader.run(TOPIC, db_url=env["url"], **kw)


def query(env, model):
    engine = create_engine(env["url"], future=True)
    with Session(engine) as s:
        rows = list(s.scalars(select(model)))
    engine.dispose()
    return rows


def test_dry_run_reports_and_writes_nothing(env):
    before_db, before_vocab = sha(env["db"]), sha(env["vocab"])
    report = run(env)
    assert report["result"] == "dry run: nothing written"
    assert sha(env["db"]) == before_db and sha(env["vocab"]) == before_vocab
    assert report["bank"]["action"].startswith("replace") and report["bank"]["new_items"] == 15
    assert report["parallel_test_items"]["insert"] == 15 and report["formative_items"]["insert"] == 33
    assert report["vocabulary"]["removed_ids"] == ["life_processes_overview"]
    assert {c.split(" ")[0] for c in report["conflicts"]} == {"question", "vocabulary"}
    assert report["pending_migrations"][-1] == "0004_parallel_test_items"
    assert report["untouched"]["learner_profiles"] == 1 and report["untouched"]["assessments"] == 1
    assert json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))["mode"] == "dry-run"


def test_apply_with_conflicts_is_refused_and_writes_nothing(env):
    before = sha(env["db"])
    report = run(env, do_apply=True)
    assert report["result"] == "REFUSED: unresolved conflicts" and len(report["blocking_conflicts"]) == 2
    assert sha(env["db"]) == before and not (env["tmp"] / "backups").exists()
    report = run(env, do_apply=True, replace_bank=True)  # vocabulary conflict still unresolved
    assert report["result"] == "REFUSED: unresolved conflicts" and sha(env["db"]) == before


def test_approved_apply_backs_up_loads_and_preserves_learner_data(env):
    report = run(env, do_apply=True, replace_bank=True, replace_vocabulary=True)
    assert report["result"] == "applied"
    backup = Path(report["backup"])
    assert backup.exists() and backup.parent == loader.BACKUPS
    engine = create_engine(f"sqlite:///{backup.as_posix()}", future=True)
    with engine.connect() as c:  # the backup is the database as it was before the load
        assert c.execute(text("SELECT n_items FROM topic_question_banks WHERE topic_id = :t"), {"t": TOPIC}).scalar() == 1
    engine.dispose()
    assert report["vocabulary_backup"] and json.loads(open(report["vocabulary_backup"], encoding="utf-8").read())[
        "reviewed_by"] == "Your Name"

    bank = [b for b in query(env, TopicQuestionBank) if b.topic_id == TOPIC][0]
    assert bank.n_items == 15 and bank.model_provider == "content-package"
    assert [i["id"] for i in bank.items][:2] == ["LP-PRE-01", "LP-PRE-02"]
    assert "parallel" not in json.dumps(bank.items) and "LP-POST" not in json.dumps(bank.items)
    pairs = query(env, ParallelTestItem)
    assert len(pairs) == 15 and {p.pre_item_id for p in pairs} == {i["id"] for i in bank.items}
    formative = query(env, FormativeItem)
    assert len(formative) == 33 and {f.concept_check for f in formative} == {"pending_review"}
    assert not {f.question_key for f in formative} & {i["question"].lower() for i in bank.items}
    old = query(env, Assessment)
    assert len(old) == 1 and old[0].items == [{"question": "Old?"}]  # attempts untouched
    vocab = json.loads(env["vocab"].read_text(encoding="utf-8"))
    assert vocab["status"] == "draft" and vocab["reviewed_by"] is None and len(vocab["concepts"]) == 10


def test_rerun_after_apply_is_idempotent(env):
    run(env, do_apply=True, replace_bank=True, replace_vocabulary=True)
    report = run(env)
    assert report["conflicts"] == [] and report["bank"]["action"] == "skip (identical)"
    assert report["parallel_test_items"]["skip"] == 15 and report["formative_items"]["skip"] == 33
    assert report["vocabulary"]["action"] == "skip (identical)" and report["pending_migrations"] == []
    assert run(env, do_apply=True)["result"] == "applied"
    assert len(query(env, ParallelTestItem)) == 15 and len(query(env, FormativeItem)) == 33  # nothing twice


def test_existing_generated_formative_item_with_the_same_text_is_a_conflict(env):
    run(env, do_apply=True, replace_bank=True, replace_vocabulary=True)
    engine = create_engine(env["url"], future=True)
    first = json.loads((PROJECT_ROOT / "knowledge_prep" / "packages" / TOPIC / "formative.json")
                       .read_text(encoding="utf-8"))["items"][0]
    with Session(engine) as s, s.begin():
        s.get(FormativeItem, first["id"]).id = "fp_generated01"  # as if the pool generator had made it
    engine.dispose()
    report = run(env, do_apply=True)
    assert report["result"] == "REFUSED: unresolved conflicts"
    assert any("same question text as existing formative item fp_generated01" in c for c in report["conflicts"])


def test_invalid_package_is_refused(env, monkeypatch):
    monkeypatch.setattr(loader, "validate", lambda topic_id: {"valid": False, "evidence_checked": True,
                                                             "errors": ["bad"], "warnings": []})
    before = sha(env["db"])
    report = run(env, do_apply=True, replace_bank=True, replace_vocabulary=True)
    assert report["result"].startswith("REFUSED") and sha(env["db"]) == before
    monkeypatch.setattr(loader, "validate", lambda topic_id: {"valid": True, "evidence_checked": False,
                                                             "errors": [], "warnings": []})
    assert run(env)["result"].startswith("REFUSED")  # grounding must have been checked locally


def test_open_session_blocks_a_bank_replacement_until_named_exactly(env):
    engine = create_engine(env["url"], future=True)
    with Session(engine) as s, s.begin():  # a pre-test on the old bank, no post-test yet
        trace = LearningTraceRepository(s)
        learner = trace.create_learner(LearnerProfileCreate(display_name="Ravi"))
        open_id = trace.create_learning_session(LearningSessionCreate(learner_id=learner.id)).id
        s.add(Assessment(session_id=open_id, student_id=learner.id, topic_id=TOPIC, type="pre",
                         items=[{"question": "Generated?"}], answers=[0], score=1.0, competency="advanced"))
    engine.dispose()
    flags = {"do_apply": True, "replace_bank": True, "replace_vocabulary": True}
    report = run(env, **flags)
    assert report["open_session_ids"] == [open_id] and report["result"] == "REFUSED: unresolved conflicts"
    assert run(env, **flags, accept_open_sessions=["not-that-one"])["result"] == "REFUSED: unresolved conflicts"
    report = run(env, **flags, accept_open_sessions=[open_id])
    assert report["result"] == "applied" and report["accepted_open_sessions"] == [open_id]
    kept = [a for a in query(env, Assessment) if a.session_id == open_id]
    assert len(kept) == 1 and kept[0].items == [{"question": "Generated?"}] and kept[0].score == 1.0


def test_reviewed_vocabulary_with_the_same_concepts_is_kept(env):
    run(env, do_apply=True, replace_bank=True, replace_vocabulary=True)
    vocab = json.loads(env["vocab"].read_text(encoding="utf-8"))
    vocab |= {"status": "approved", "reviewed_by": "Project team", "reviewed_at": "2026-10-26T00:00:00+00:00"}
    env["vocab"].write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    report = run(env)
    assert report["conflicts"] == [] and report["vocabulary"]["action"].startswith("skip (identical concepts")
    assert run(env, do_apply=True)["result"] == "applied"
    assert json.loads(env["vocab"].read_text(encoding="utf-8"))["reviewed_by"] == "Project team"  # not reset
