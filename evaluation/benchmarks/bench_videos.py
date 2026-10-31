"""Result 6 (+ the coach's "another video" cost): recommendation strategies R1/R2/R3 on every topic, cold cache.

    .venv\\Scripts\\python -m evaluation.benchmarks.bench_videos

For each topic: R1, R2 and R3 on the topic query, and R2 on one concept-focused query (what an accepted "another
video" suggestion runs), each with an empty YouTube cache (a learner nobody has warmed the cache for). Then the top
recommendation is made answerable: its transcript is fetched, chunked and embedded into a video index (what the
learner waits for after "Learn from this video").

Uses ~100 YouTube quota units per run (~2,000 in all). Runs are spaced out (--gap-s) and the benchmark stops when
YouTube starts blocking caption requests, so it does not prolong a block. Nothing is stored as a recommendation run.
Writes evaluation/results/videos/{runs.csv, items.csv}; items.csv has an empty `relevant` column the team can fill.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

from evaluation.benchmarks.isolation import RESULTS, disable_tracing, isolate

isolate("videos", fresh_copy=True)

import pandas as pd  # noqa: E402

from agent.deps import get_services  # noqa: E402
from agent.knowledge import ensure_video_index  # noqa: E402
from agent.nodes.coaching import load_vocabulary  # noqa: E402
from agent.recommendations import STRATEGIES, make_recommender  # noqa: E402
from config import settings  # noqa: E402
from evaluation.benchmarks.harness import TOPIC_IDS  # noqa: E402
from knowledge_prep.catalog import search_text  # noqa: E402
from services.youtube.models import TranscriptStatus  # noqa: E402
from services.youtube.pipeline import RecommendationRequest  # noqa: E402

OUT = RESULTS / "videos"


def run_one(topic_id: str, strategy_id: str, concept: str | None, cache_dir: Path) -> tuple[dict, list[dict]]:
    rec = make_recommender(get_services(), cfg=settings.model_copy(update={"youtube_cache_dir": cache_dir}))
    text = search_text(topic_id)
    request = RecommendationRequest(topic=text, objective=f"{text}: {concept}" if concept else text,
                                    concepts=[concept] if concept else [], language_code="en",
                                    competency="intermediate", competency_num=0.5, count=settings.rec_count)
    started = time.perf_counter()
    result = rec.pipeline.run(request, STRATEGIES[strategy_id], None)
    wall = (time.perf_counter() - started) * 1000
    m, t = result.metrics, result.metrics.timings
    recs = result.recommendations
    blocked = sum(1 for f in result.filtered if "block" in f.reason.lower())
    row = {"topic_id": topic_id, "strategy": strategy_id, "query": "concept" if concept else "topic",
           "concept": concept, "status": result.status, "wall_ms": wall, "search_ms": t.search_ms,
           "metadata_ms": t.metadata_ms, "transcript_ms": t.transcript_ms, "embedding_ms": t.embedding_ms,
           "ranking_ms": t.ranking_ms, "quota_units": m.quota_units, "api_bytes": m.api_bytes,
           "candidates": m.candidates_found, "filtered": len(result.filtered),
           "no_caption_filtered": sum("no transcript track" in f.reason for f in result.filtered),
           "recommended": len(recs), "transcripts_ok": m.transcripts_ok, "transcripts_attempted": m.transcripts_attempted,
           "mean_topic_similarity": (sum((r.score_breakdown.get("topic_similarity") or 0) for r in recs) / len(recs)
                                     if recs else None),
           "with_transcript_evidence": sum(r.transcript_status == TranscriptStatus.OK for r in recs),
           "blocked_signals": blocked, "errors": "; ".join(result.errors)[:300]}
    if recs:  # make the top recommendation answerable, as "Learn from this video" does
        services = get_services()
        started = time.perf_counter()
        try:
            info = ensure_video_index(services.crag.store, rec.transcripts, {"video_id": recs[0].video_id,
                                                                             "title": recs[0].title}, ["en"])
            row.update({"index_ms": (time.perf_counter() - started) * 1000, "index_chunks": info.get("n_chunks"),
                        "index_reused": info.get("reused_index"), "index_error": None})
        except Exception as exc:  # no usable transcript after all, blocked, embedder down
            row.update({"index_ms": (time.perf_counter() - started) * 1000, "index_chunks": None,
                        "index_error": f"{type(exc).__name__}: {exc}"[:200]})
    items = [{"topic_id": topic_id, "strategy": strategy_id, "query": row["query"], "rank": r.rank,
              "video_id": r.video_id, "title": r.title, "channel": r.channel_title, "duration_s": r.duration_s,
              "score": r.score, "evidence": r.evidence_level, "topic_similarity": r.score_breakdown.get("topic_similarity"),
              "transcript_relevance": r.score_breakdown.get("transcript_relevance"), "url": r.url,
              "relevant": ""} for r in recs]
    return row, items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap-s", type=float, default=45.0, help="pause between runs (YouTube rate limiting)")
    parser.add_argument("--topics", nargs="+", default=TOPIC_IDS)
    args = parser.parse_args()
    disable_tracing()
    OUT.mkdir(parents=True, exist_ok=True)
    runs, items = [], []
    plan = [(t, s, None) for t in args.topics for s in ("R1", "R2", "R3")]
    plan += [(t, "R2", load_vocabulary(t).concepts[0].name) for t in args.topics]
    for n, (topic_id, strategy_id, concept) in enumerate(plan, 1):
        with tempfile.TemporaryDirectory(prefix="ytbench-") as cache_dir:
            row, its = run_one(topic_id, strategy_id, concept, Path(cache_dir))
        runs.append(row)
        items += its
        pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)
        pd.DataFrame(items).to_csv(OUT / "items.csv", index=False)
        print(f"{n}/{len(plan)} {topic_id} {strategy_id} {row['query']}: {row['wall_ms'] / 1000:.1f} s, "
              f"{row['recommended']} recs, quota {row['quota_units']}, blocked {row['blocked_signals']}", flush=True)
        if row["blocked_signals"]:
            print("YouTube is blocking caption requests: stopping (re-run later; finished runs are kept)")
            return 2
        if n < len(plan):
            time.sleep(args.gap_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
