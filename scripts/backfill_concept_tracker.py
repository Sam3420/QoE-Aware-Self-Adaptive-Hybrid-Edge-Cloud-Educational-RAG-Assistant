"""Backfill the concept tracker from every stored pre/post-test (idempotent: re-running records nothing new).

  .venv\\Scripts\\python scripts\\backfill_concept_tracker.py --dry-run   # show how labels map, write nothing
  .venv\\Scripts\\python scripts\\backfill_concept_tracker.py             # migrate, then record the attempts

Each stored test item's concept label is mapped to the topic's vocabulary (knowledge_prep/concepts/); its answer becomes
one attempt on that concept. Topics without a vocabulary (e.g. the old Class 8 topics) are skipped and reported.
Labels go through name/alias matching, then nomic-embed-text similarity (Ollama must be running; without it only exact
and alias matches are made).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.concept_tracking import AssessmentTracker, TrackReport  # noqa: E402
from agent.deps import make_embedder  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.db.migrations import run_migrations  # noqa: E402
from app.db.repositories import AdaptationRepository, ConceptTrackerRepository  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from config import settings  # noqa: E402


def embedder_or_none():
    try:
        embedder = make_embedder()
        embedder.embed_query("test")
        return embedder
    except Exception as exc:  # Ollama down: exact/alias matching only
        print(f"Embedder unavailable ({type(exc).__name__}); matching by name/alias only.")
        return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Backfill the concept tracker from stored pre/post-tests.")
    parser.add_argument("--dry-run", action="store_true", help="report the label mapping, write nothing")
    parser.add_argument("--topic", help="only this topic")
    args = parser.parse_args()
    print(f"Database: {get_settings().resolved_database_url}")
    print(f"Embedding match: similarity >= {settings.concept_match_min_similarity}, "
          f"margin over the runner-up >= {settings.concept_match_min_margin}")
    if not args.dry_run:
        ran = run_migrations(engine)
        print(f"Migrations applied now: {', '.join(ran) or 'none (up to date)'}")
    tracker, total = AssessmentTracker(embedder_or_none()), TrackReport()
    with SessionLocal() as db:
        assessments = AdaptationRepository(db).list_all_assessments(args.topic)
        repo = ConceptTrackerRepository(db)
        for assessment in assessments:
            if args.dry_run:  # match only
                matcher = tracker.matcher(assessment["topic_id"])
                part = TrackReport(assessments=1)
                if matcher is None:
                    part.skipped_topics[assessment["topic_id"]] += 1
                else:
                    part.vocabulary_status[assessment["topic_id"]] = matcher.vocabulary.status
                    for item in assessment["items"]:
                        part.items += 1
                        m = matcher.match(str(item.get("concept") or ""), context=str(item.get("question") or ""))
                        part.matches[str(item.get("concept"))] = (m.concept_id, m.reason or m.method, m.similarity)
                        (part.methods if m.concept_id else part.unmapped)[m.method if m.concept_id
                                                                          else str(item.get("concept"))] += 1
            else:
                part = tracker.record(repo, assessment)
            total.merge(part)
        if not args.dry_run:
            db.commit()

    print(f"\nAssessments: {total.assessments} | items: {total.items} | attempts recorded now: {total.recorded}"
          f" | already recorded: {total.already}")
    print(f"Matched by: {dict(total.methods)} | unmapped items: {sum(total.unmapped.values())}")
    for topic, status in sorted(total.vocabulary_status.items()):
        print(f"  vocabulary {topic}: {status}")
    for topic, n in sorted(total.skipped_topics.items()):
        print(f"  skipped (no vocabulary): {topic} ({n} assessments)")
    print("\nLabel mapping (label -> concept, method, similarity):")
    for label, (concept, method, sim) in sorted(total.matches.items(), key=lambda kv: (kv[1][0] is None, kv[0])):
        print(f"  {label[:50]:<50} -> {concept or '-':<30} {'' if sim is None else f'{sim:.3f}':<6} {method}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
