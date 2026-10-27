"""Load a validated content package into the app: dry-run by default, writes only with --apply.

  .venv\\Scripts\\python scripts\\load_content_package.py life_processes                  # dry run (read-only)
  .venv\\Scripts\\python scripts\\load_content_package.py life_processes --apply          # write, if no conflicts
  .venv\\Scripts\\python scripts\\load_content_package.py life_processes --apply --replace-bank --replace-vocabulary

Targets (all keyed by the package's stable content ids, so a re-run inserts nothing twice):
- topic_question_banks: the topic's bank = the pre-test items, in package order (pairs are NOT stored here)
- parallel_test_items:  one post-test question per pre-test item (read only when a post-test begins)
- formative_items:      the formative questions (kept apart from both test banks)
- knowledge_prep/concepts/<topic_id>.json: the topic vocabulary the coach and tracker use (status stays draft)

Safety:
- The package must validate first, with its local evidence checked (scripts/validate_content_package.py).
- Dry run opens the database read-only and reports inserts, updates, skips and conflicts; it writes a report to
  data/content_loads/ (gitignored) and nothing else.
- An existing, different question bank or vocabulary is a CONFLICT; --apply refuses unless --replace-bank /
  --replace-vocabulary are given. Content ids owned by the package are updated in place; a row that clashes with a
  non-package row (same question text, or another item paired with the same pre-test question) is a conflict.
- --apply backs up the database (SQLite online backup) and any replaced vocabulary file to data/backups/ first,
  applies pending additive migrations, then writes everything in one transaction. Learners, sessions, assessments,
  feedback, experiences and tracker rows are never modified or deleted.
"""
import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, inspect, select, text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.migrations import MIGRATIONS, applied, run_migrations  # noqa: E402
from app.db.models import FormativeItem, ParallelTestItem, TopicQuestionBank  # noqa: E402
from app.services.mcq import question_key  # noqa: E402
from config import PROJECT_ROOT  # noqa: E402
from knowledge_prep.concepts import Vocabulary  # noqa: E402
from scripts.validate_content_package import PACKAGES, validate  # noqa: E402

REPORTS, BACKUPS = PROJECT_ROOT / "data" / "content_loads", PROJECT_ROOT / "data" / "backups"
VOCAB_DIR = PROJECT_ROOT / "knowledge_prep" / "concepts"
BANK_FIELDS = ("id", "question", "options", "answer_index", "concept", "concept_id", "difficulty", "explanation",
               "source")


def sqlite_path(url: str) -> Path:
    if not url.startswith("sqlite:///"):
        raise SystemExit(f"Only SQLite databases are supported by this loader (got {url.split(':')[0]}).")
    return Path(url.removeprefix("sqlite:///"))


def read_only_engine(path: Path):
    """An engine whose connections SQLite itself opens read-only (mode=ro): the dry run cannot write."""
    uri = f"file:{quote(path.as_posix(), safe='/:')}?mode=ro"
    return create_engine("sqlite://", creator=lambda: sqlite3.connect(uri, uri=True))


# --- what the package wants the database to hold ------------------------------------------------------------------

def package_targets(topic_id: str) -> dict[str, Any]:
    topic_dir = PACKAGES / topic_id
    load = lambda name: json.loads((topic_dir / name).read_text(encoding="utf-8"))  # noqa: E731
    manifest, concepts = load("manifest.json"), load("concepts.json")
    names = {c["id"]: c["name"] for c in concepts["concepts"]}
    version = f"package:{topic_id}@{manifest['package_version']}"

    def bank_item(q: dict[str, Any]) -> dict[str, Any]:  # the app's bank-item shape (concept = the concept's name)
        item = {k: q[k] for k in BANK_FIELDS if k in q and k != "concept"}
        return item | {"concept": names[q["concept_id"]]}

    pre = [bank_item(q) for q in load("pretest.json")["items"]]
    post = load("posttest.json")["items"]
    vocabulary = Vocabulary.model_validate(concepts | {"concepts": [{k: v for k, v in c.items() if k != "source"}
                                                                    for c in concepts["concepts"]]})
    return {
        "version": version,
        "bank": pre,
        "pairs": [{"id": q["id"], "pre_item_id": q["parallel_to"],
                   "item": {k: v for k, v in bank_item(q).items() if k != "id"}} for q in post],
        "formative": [{"id": q["id"], "concept_id": q["concept_id"], "position": n, "question": q["question"],
                       "question_key": question_key(q["question"]), "options": q["options"],
                       "answer_index": q["answer_index"], "explanation": q["explanation"]}
                      for n, q in enumerate(load("formative.json")["items"])],
        "vocabulary": json.loads(vocabulary.model_dump_json()),
    }


# --- compare with what the database holds -------------------------------------------------------------------------

def plan(topic_id: str, targets: dict[str, Any], session: Session) -> dict[str, Any]:
    tables = set(inspect(session.connection()).get_table_names())
    report: dict[str, Any] = {"bank": {}, "parallel_test_items": {}, "formative_items": {}, "vocabulary": {},
                              "conflicts": [], "untouched": {}}

    # question bank
    existing = session.scalars(select(TopicQuestionBank).where(TopicQuestionBank.topic_id == topic_id)).first()
    if existing is None:
        report["bank"] = {"action": "insert", "items": len(targets["bank"])}
    elif existing.items == targets["bank"]:
        report["bank"] = {"action": "skip (identical)", "items": len(existing.items)}
    else:
        report["bank"] = {"action": "replace (needs --replace-bank)", "existing_items": existing.n_items,
                          "existing_model": f"{existing.model_provider}/{existing.model_name}",
                          "new_items": len(targets["bank"])}
        report["conflicts"].append(f"question bank for {topic_id!r} exists ({existing.n_items} items from "
                                   f"{existing.model_provider}/{existing.model_name}) and differs")

    # parallel pairs
    rows = {r.id: r for r in session.scalars(select(ParallelTestItem).where(ParallelTestItem.topic_id == topic_id))} \
        if "parallel_test_items" in tables else {}
    by_pre = {r.pre_item_id: r for r in rows.values()}
    counts = {"insert": 0, "update": 0, "skip": 0}
    for pair in targets["pairs"]:
        row = rows.get(pair["id"])
        clash = by_pre.get(pair["pre_item_id"])
        if clash is not None and clash.id != pair["id"]:
            report["conflicts"].append(f"{pair['pre_item_id']} is already paired with {clash.id}, not {pair['id']}")
        elif row is None:
            counts["insert"] += 1
        elif row.item == pair["item"] and row.pre_item_id == pair["pre_item_id"]:
            counts["skip"] += 1
        else:
            counts["update"] += 1
    report["parallel_test_items"] = counts | ({} if "parallel_test_items" in tables
                                              else {"note": "table created by migration 0004 on --apply"})

    # formative pool
    pool = list(session.scalars(select(FormativeItem).where(FormativeItem.topic_id == topic_id))) \
        if "formative_items" in tables else []
    by_id, by_key = {r.id: r for r in pool}, {r.question_key: r for r in pool}
    counts = {"insert": 0, "update": 0, "skip": 0}
    for f in targets["formative"]:
        row, clash = by_id.get(f["id"]), by_key.get(f["question_key"])
        if clash is not None and clash.id != f["id"]:
            report["conflicts"].append(f"{f['id']}: same question text as existing formative item {clash.id}")
        elif row is None:
            counts["insert"] += 1
        elif all(getattr(row, k) == f[k] for k in ("concept_id", "question", "options", "answer_index", "explanation")):
            counts["skip"] += 1
        else:
            counts["update"] += 1
    other = [r.id for r in pool if r.id not in {f["id"] for f in targets["formative"]}]
    report["formative_items"] = counts | {"existing_other_items_kept": len(other)} | (
        {} if "formative_items" in tables else {"note": "table created by migration 0003 on --apply"})

    # vocabulary file
    path = VOCAB_DIR / f"{topic_id}.json"
    new_ids = [c["id"] for c in targets["vocabulary"]["concepts"]]
    if not path.exists():
        report["vocabulary"] = {"action": "create", "path": str(path), "concepts": len(new_ids)}
    else:
        old = json.loads(path.read_text(encoding="utf-8"))
        old_ids = [c["id"] for c in old.get("concepts", [])]
        review = ("status", "reviewed_by", "reviewed_at")
        if old == targets["vocabulary"]:
            report["vocabulary"] = {"action": "skip (identical)", "path": str(path)}
        elif {k: v for k, v in old.items() if k not in review} == \
                {k: v for k, v in targets["vocabulary"].items() if k not in review}:
            # the same concepts, reviewed since they were loaded: keep the review rather than resetting it to draft
            report["vocabulary"] = {"action": f"skip (identical concepts; keeps status {old.get('status')!r})",
                                    "path": str(path), "reviewed_by": old.get("reviewed_by")}
        else:
            report["vocabulary"] = {"action": "replace (needs --replace-vocabulary)", "path": str(path),
                                    "existing_status": old.get("status"), "existing_reviewed_by": old.get("reviewed_by"),
                                    "kept_ids": [i for i in new_ids if i in old_ids],
                                    "added_ids": [i for i in new_ids if i not in old_ids],
                                    "removed_ids": [i for i in old_ids if i not in new_ids]}
            report["conflicts"].append(f"vocabulary {path.name} exists (status {old.get('status')!r}, reviewed_by "
                                       f"{old.get('reviewed_by')!r}) and differs")
        removed = report["vocabulary"].get("removed_ids", [])
        if removed and "learner_topic_strength" in tables:
            n = session.execute(text("SELECT COUNT(*) FROM learner_topic_strength WHERE topic_id = :t AND "
                                     f"concept_id IN ({', '.join(':c' + str(i) for i in range(len(removed)))})"),
                                {"t": topic_id, **{f"c{i}": c for i, c in enumerate(removed)}}).scalar()
            report["vocabulary"]["tracker_rows_on_removed_ids"] = n
            if n:
                report["vocabulary"]["tracker_rows_note"] = ("kept as they are (not deleted or reset); they no longer "
                                                             "match a concept of the vocabulary")

    # records the loader never touches, and attempts that a bank replacement could affect
    for table in ("learner_profiles", "learning_sessions", "assessments", "session_feedback", "experiences",
                  "learner_topic_strength", "concept_events"):
        if table in tables:
            report["untouched"][table] = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    if "assessments" in tables:
        topic_attempts = session.execute(text("SELECT type, COUNT(*) FROM assessments WHERE topic_id = :t GROUP BY type"),
                                         {"t": topic_id}).all()
        open_sessions = sorted(session.scalars(text(
            "SELECT DISTINCT session_id FROM assessments a WHERE topic_id = :t AND type = 'pre' AND NOT EXISTS "
            "(SELECT 1 FROM assessments b WHERE b.session_id = a.session_id AND b.type = 'post')"), {"t": topic_id}))
        report["topic_attempts"] = dict(topic_attempts)
        report["sessions_with_pretest_but_no_posttest"] = len(open_sessions)
        report["open_session_ids"] = open_sessions
        if open_sessions and report["bank"].get("action", "").startswith("replace"):
            report["conflicts"].append(f"open sessions: {len(open_sessions)} session(s) took this topic's pre-test but "
                                       "not the post-test; replacing the bank would change which questions their "
                                       "post-test asks")
    return report


# --- write ------------------------------------------------------------------------------------------------------------

def backup(db_path: Path, topic_id: str, stamp: str) -> Path:
    BACKUPS.mkdir(parents=True, exist_ok=True)
    target = BACKUPS / f"{db_path.stem}-before-{topic_id}-load-{stamp}.db"
    with sqlite3.connect(db_path) as src, sqlite3.connect(target) as dst:
        src.backup(dst)  # consistent online copy
    return target


def apply(topic_id: str, targets: dict[str, Any], db_url: str, stamp: str, *,
          write_vocabulary: bool = True) -> dict[str, Any]:
    engine = create_engine(db_url, future=True)
    ran = run_migrations(engine)
    with Session(engine) as session, session.begin():
        bank = session.scalars(select(TopicQuestionBank).where(TopicQuestionBank.topic_id == topic_id)).first()
        if bank is None:
            session.add(TopicQuestionBank(topic_id=topic_id, items=targets["bank"], n_items=len(targets["bank"]),
                                          model_provider="content-package", model_name=targets["version"]))
        elif bank.items != targets["bank"]:
            bank.items, bank.n_items = targets["bank"], len(targets["bank"])
            bank.model_provider, bank.model_name = "content-package", targets["version"]
        for pair in targets["pairs"]:
            row = session.get(ParallelTestItem, pair["id"])
            if row is None:
                session.add(ParallelTestItem(id=pair["id"], topic_id=topic_id, pre_item_id=pair["pre_item_id"],
                                             item=pair["item"], source=targets["version"]))
            else:
                row.pre_item_id, row.item, row.source = pair["pre_item_id"], pair["item"], targets["version"]
        for f in targets["formative"]:
            row = session.get(FormativeItem, f["id"])
            values = {k: f[k] for k in ("concept_id", "position", "question", "question_key", "options",
                                        "answer_index", "explanation")}
            if row is None:
                session.add(FormativeItem(id=f["id"], topic_id=topic_id, concept_check="pending_review",
                                          model_provider="content-package", model_name=targets["version"], **values))
            else:
                for k, v in values.items():
                    setattr(row, k, v)
    if not write_vocabulary:  # the same concepts, already reviewed: leave the file (and its review) as it is
        return {"migrations_applied": ran, "vocabulary_backup": None}
    path = VOCAB_DIR / f"{topic_id}.json"
    saved = None
    if path.exists():
        saved = BACKUPS / f"{path.stem}-vocabulary-before-load-{stamp}.json"
        shutil.copy2(path, saved)
    path.write_text(json.dumps(targets["vocabulary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"migrations_applied": ran, "vocabulary_backup": str(saved) if saved else None}


def run(topic_id: str, *, do_apply: bool = False, replace_bank: bool = False, replace_vocabulary: bool = False,
        accept_open_sessions: Sequence[str] = (), db_url: str | None = None) -> dict[str, Any]:
    """Validate, compare and (with do_apply) write; returns the report, also saved under data/content_loads/.
    ``accept_open_sessions`` must name exactly the sessions with a pre-test but no post-test before a bank replacement
    may go ahead; their records are left unchanged, and the acknowledgement is kept in the report."""
    validation = validate(topic_id)
    validation.pop("_items", None), validation.pop("_concepts", None)
    if not validation["valid"] or not validation["evidence_checked"]:
        return {"topic_id": topic_id, "result": "REFUSED: the package is not valid with checked evidence",
                "errors": validation["errors"]}
    db_url = db_url or get_settings().resolved_database_url
    db_path = sqlite_path(db_url)
    targets = package_targets(topic_id)
    with Session(read_only_engine(db_path)) as session:
        report = plan(topic_id, targets, session)
        session.rollback()
    done = set(applied(read_only_engine(db_path)))
    report["pending_migrations"] = [m.id for m in MIGRATIONS if m.id not in done]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = {"topic_id": topic_id, "mode": "apply" if do_apply else "dry-run", "at": stamp,
              "database": str(db_path), "package": targets["version"],
              "validation": {"valid": True, "warnings": len(validation["warnings"])}, **report}

    open_ok = bool(report.get("open_session_ids")) and sorted(set(accept_open_sessions)) == report["open_session_ids"]
    if accept_open_sessions:
        report["accepted_open_sessions"] = sorted(set(accept_open_sessions))
    blocking = [c for c in report["conflicts"]
                if not (c.startswith("question bank") and replace_bank)
                and not (c.startswith("vocabulary") and replace_vocabulary)
                and not (c.startswith("open sessions") and open_ok)]
    if do_apply and blocking:
        report["result"] = "REFUSED: unresolved conflicts"
        report["blocking_conflicts"] = blocking
    elif do_apply:
        report["backup"] = str(backup(db_path, topic_id, stamp))
        report["result"] = "applied"
        keep = str(report.get("vocabulary", {}).get("action", "")).startswith("skip (identical concepts")
        report |= apply(topic_id, targets, db_url, stamp, write_vocabulary=not keep)
    else:
        report["result"] = "dry run: nothing written"

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{topic_id}-{report['mode']}-{stamp}.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    report["report_path"] = str(out)
    return report


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Load a content package (dry run unless --apply).")
    parser.add_argument("topic_id")
    parser.add_argument("--apply", action="store_true", help="write to the database (after a reviewed dry run)")
    parser.add_argument("--replace-bank", action="store_true", help="allow replacing an existing, different bank")
    parser.add_argument("--replace-vocabulary", action="store_true", help="allow replacing a different vocabulary")
    parser.add_argument("--accept-open-session", action="append", default=[], metavar="SESSION_ID",
                        help="acknowledge a session with a pre-test but no post-test (repeat; must name all of them)")
    args = parser.parse_args()
    report = run(args.topic_id, do_apply=args.apply, replace_bank=args.replace_bank,
                 replace_vocabulary=args.replace_vocabulary, accept_open_sessions=args.accept_open_session)
    print(json.dumps(report, indent=2, default=str))
    return 1 if str(report["result"]).startswith("REFUSED") else 0


if __name__ == "__main__":
    sys.exit(main())
