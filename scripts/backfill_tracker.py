"""Replay stored pre/post-test results into the concept tracker through each topic's question_concept_map.

  .venv\\Scripts\\python scripts\\backfill_tracker.py                     # migrate the app DB, then backfill
  .venv\\Scripts\\python scripts\\backfill_tracker.py --include-unreviewed  # also use mappings flagged needs_review

Each stored test item is looked up in its topic's question_concept_map by question text (stored items do not keep
their bank position) and its answer recorded with tracker.record_attempt. Safe to re-run: an answer is identified by
``"<assessment id>:<item index>"``, the same id the live test recording uses, so an answer already recorded (by an
earlier run, live, or scripts/backfill_concept_tracker.py) is not counted again.
"""
import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import tracker  # noqa: E402
from app.db.models import ConceptEvent  # noqa: E402
from app.db.repositories import AdaptationRepository  # noqa: E402
from app.services.mcq import normalize_answer, question_key  # noqa: E402
from knowledge_prep.concepts import QuestionMapping, load_vocabulary  # noqa: E402

SOURCES = {"pre": "pretest", "post": "posttest"}


@dataclass
class Summary:
    assessments: int = 0
    learners: set[str] = field(default_factory=set)  # learners with at least one mapped answer
    recorded: int = 0  # new attempts
    already: int = 0  # recorded before (re-run, live recording, the label backfill)
    skipped: Counter = field(default_factory=Counter)  # reason -> questions

    def lines(self) -> list[str]:
        out = [f"assessments read: {self.assessments}", f"learners: {len(self.learners)}",
               f"attempts recorded now: {self.recorded}", f"attempts already recorded: {self.already}",
               f"questions skipped: {sum(self.skipped.values())}"]
        out += [f"  {reason}: {n}" for reason, n in sorted(self.skipped.items())]
        return out


def backfill(db: Session, *, include_unreviewed: bool = False) -> Summary:
    """Record every stored pre/post-test answer that maps to a concept; the caller commits."""
    summary = Summary()
    maps: dict[str, dict[str, QuestionMapping] | None] = {}
    for assessment in AdaptationRepository(db).list_all_assessments():
        summary.assessments += 1
        topic_id, items = assessment["topic_id"], assessment.get("items") or []
        source = SOURCES.get(assessment["type"])
        if source is None:
            summary.skipped[f"test type {assessment['type']!r}"] += len(items)
            continue
        if topic_id not in maps:
            maps[topic_id] = _by_question(topic_id)
        by_question = maps[topic_id]
        if by_question is None:
            summary.skipped[f"no vocabulary/map ({topic_id})"] += len(items)
            continue
        answers = assessment.get("answers") or []
        for index, item in enumerate(items):
            mapping = by_question.get(question_key(str(item.get("question") or "")))
            if mapping is None:
                summary.skipped["no mapping for the question"] += 1
                continue
            if mapping.needs_review and not include_unreviewed:
                summary.skipped["mapping needs review"] += 1
                continue
            ref = f"{assessment['id']}:{index}"
            summary.learners.add(assessment["student_id"])
            if _recorded(db, ref):
                summary.already += 1
                continue
            answer = answers[index] if index < len(answers) else None
            tracker.record_attempt(assessment["student_id"], topic_id, mapping.concept_id,
                                   normalize_answer(answer) == item.get("answer_index"), source,
                                   source_ref=ref, occurred_at=assessment.get("created_at"), db=db)
            summary.recorded += 1
    return summary


def _by_question(topic_id: str) -> dict[str, QuestionMapping] | None:
    """question text key -> mapping, or None when the topic has no vocabulary or no map."""
    vocabulary = load_vocabulary(topic_id)
    if vocabulary is None or not vocabulary.question_concept_map:
        return None
    return {question_key(m.question): m for m in vocabulary.question_concept_map.values()}


def _recorded(db: Session, ref: str) -> bool:
    """True if this answer is already an attempt on some concept (any source spelling of pre/post-test)."""
    return db.scalars(select(ConceptEvent.id).where(ConceptEvent.kind == "attempt",
                                                    ConceptEvent.source_ref == ref)).first() is not None


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Backfill the concept tracker via question_concept_map.")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="also record answers whose mapping is flagged needs_review")
    args = parser.parse_args(argv)
    from app.db.migrations import run_migrations
    from app.db.session import SessionLocal, engine

    ran = run_migrations(engine)
    print(f"migrations applied now: {ran or 'none (up to date)'}")
    with SessionLocal() as db:
        summary = backfill(db, include_unreviewed=args.include_unreviewed)
        db.commit()
    print("\n".join(summary.lines()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
