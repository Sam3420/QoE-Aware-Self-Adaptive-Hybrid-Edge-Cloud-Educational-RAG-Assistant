"""Find 3-5 YouTube video candidates for a content package and verify transcripts (writes videos.json).

  .venv\\Scripts\\python scripts\\find_package_videos.py life_processes --query "Life Processes class 10" ...

Every field comes from the YouTube Data API (search.list + videos.list) or the transcript fetch; nothing is invented.
Transcripts are fetched in English (youtube-transcript-api) only to verify availability and concept coverage; their
text is NOT stored. Coverage here is lexical: a concept counts as covered when its name or aliases occur in the
transcript at least COVERAGE_MIN_MENTIONS times (auto-captions can misspell terms, so this undercounts). The
embedding-based check (matching.video_coverage) needs Ollama and is recorded as not run.
Selection: greedy, the video adding the most uncovered concepts first, among videos with a usable transcript;
remaining slots (up to --keep) go to the best-covering alternatives, so the reviewer has a choice.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.recommendations import make_transcript_fetcher  # noqa: E402
from config import PROJECT_ROOT, settings  # noqa: E402
from knowledge_prep.concepts import Vocabulary, concept_terms, normalise  # noqa: E402
from services.youtube.cache import JsonFileCache  # noqa: E402
from services.youtube.data_api import QuotaTracker, YouTubeDataClient  # noqa: E402
from services.youtube.models import TranscriptStatus  # noqa: E402
from services.youtube.transcripts import clean_text  # noqa: E402

PACKAGES = PROJECT_ROOT / "knowledge_prep" / "packages"
MIN_DURATION_S, MAX_DURATION_S = 180, 45 * 60


def lexical_coverage(transcript: str, vocabulary: Vocabulary) -> list[dict]:
    text = f" {normalise(transcript)} "
    covered = []
    for concept in vocabulary.concepts:
        mentions = {t: text.count(f" {t} ") for t in concept_terms(concept)}
        mentions = {t: n for t, n in mentions.items() if n}
        if sum(mentions.values()) >= settings.coverage_min_mentions:
            covered.append({"concept_id": concept.id, "method": "lexical", "mentions": mentions})
    return covered


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Find and verify video candidates for a content package.")
    parser.add_argument("topic_id")
    parser.add_argument("--query", action="append", required=True, help="search query (repeatable)")
    parser.add_argument("--per-query", type=int, default=8)
    parser.add_argument("--keep", type=int, default=5)
    args = parser.parse_args()
    topic_dir = PACKAGES / args.topic_id
    raw = json.loads((topic_dir / "concepts.json").read_text(encoding="utf-8"))
    vocabulary = Vocabulary.model_validate(raw | {"concepts": [{k: v for k, v in c.items() if k != "source"}
                                                               for c in raw["concepts"]]})
    if not settings.youtube_api_key:
        raise SystemExit("YOUTUBE_API_KEY is not set.")
    cache = JsonFileCache(settings.youtube_cache_dir, {"search": settings.youtube_search_cache_ttl_s,
                                                       "videos": settings.youtube_video_cache_ttl_s,
                                                       "transcripts": settings.transcript_cache_ttl_s})
    client = YouTubeDataClient(api_key=settings.youtube_api_key.get_secret_value(),
                               base_url=settings.youtube_api_base_url, timeout_s=settings.youtube_timeout_s,
                               cache=cache, quota=QuotaTracker(settings.youtube_cache_dir / "quota.json",
                                                               settings.youtube_daily_quota_units,
                                                               settings.youtube_quota_reserve_units))
    fetcher = make_transcript_fetcher(settings, cache)
    seen, examined, usable = set(), [], []
    for query in args.query:
        found = client.search(query, max_results=args.per_query, region_code=settings.youtube_region_code,
                              safe_search=settings.youtube_safe_search, relevance_language="en")
        details = client.get_videos([c.video_id for c in found])  # duration, captions flag, audio language
        for cand in (client.merge_details(c, details.get(c.video_id)) for c in found):
            if cand.video_id in seen:
                continue
            seen.add(cand.video_id)
            base = {"video_id": cand.video_id, "title": cand.title, "search_query": query,
                    "search_rank": cand.search_rank}
            if cand.duration_s is None or not MIN_DURATION_S <= cand.duration_s <= MAX_DURATION_S:
                examined.append({**base, "rejected": f"duration {cand.duration_s} s outside "
                                                     f"{MIN_DURATION_S}-{MAX_DURATION_S} s"})
                continue
            record = fetcher.fetch(cand.video_id, ["en"])
            checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if record.status != TranscriptStatus.OK:
                examined.append({**base, "rejected": f"transcript {record.status.value}"
                                                     + (f": {record.error[:120]}" if record.error else "")})
                continue
            text = clean_text(record.text)
            usable.append({
                "video_id": cand.video_id, "url": f"https://www.youtube.com/watch?v={cand.video_id}",
                "title": cand.title, "channel_title": cand.channel_title, "channel_id": cand.channel_id,
                "published_at": cand.published_at.isoformat() if cand.published_at else None,
                "duration_s": cand.duration_s, "search_query": query, "search_rank": cand.search_rank,
                "api_caption_flag": cand.caption_flag, "default_audio_language": cand.default_audio_language,
                "transcript": {"status": record.status.value, "language_code": record.language_code,
                               "caption_type": record.caption_type, "words": len(text.split()),
                               "checked_at": checked_at},
                "covers": lexical_coverage(text, vocabulary),
                "review": {"status": "pending", "reviewed_by": None, "reviewed_at": None, "notes": ""}})
    chosen, covered = [], set()
    pool = sorted(usable, key=lambda v: -len(v["covers"]))
    while pool and len(chosen) < args.keep:  # greedy cover first, then alternatives by coverage for the reviewer
        best = max(pool, key=lambda v: (len({c["concept_id"] for c in v["covers"]} - covered), len(v["covers"])))
        if not best["covers"]:
            break
        chosen.append(best)
        covered |= {c["concept_id"] for c in best["covers"]}
        pool.remove(best)
    for v in pool:
        examined.append({"video_id": v["video_id"], "title": v["title"], "search_query": v["search_query"],
                         "search_rank": v["search_rank"], "rejected": "usable, not among the best-covering --keep"})
    all_ids = [c.id for c in vocabulary.concepts]
    out = {"topic_id": args.topic_id, "status": "pending_review",
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "method": {"source": "YouTube Data API v3 (search.list, videos.list) + youtube-transcript-api (en)",
                      "queries": args.query, "duration_s": [MIN_DURATION_S, MAX_DURATION_S],
                      "coverage": f"lexical: concept name/alias mentioned >= {settings.coverage_min_mentions} "
                                  "times in the transcript",
                      "embedding_coverage": "not run: Ollama (nomic-embed-text) unavailable; matching.video_coverage "
                                            "must be run before relying on coverage",
                      "transcript_text_stored": False},
           "concepts_covered": sorted(covered), "concepts_not_covered": [c for c in all_ids if c not in covered],
           "candidates": chosen, "examined": examined}
    path = topic_dir / "videos.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(seen)} videos examined, {len(usable)} with an English transcript, {len(chosen)} kept")
    for v in chosen:
        print(f"  {v['video_id']} | {v['title'][:70]} | {v['transcript']['caption_type']} | "
              f"{len(v['covers'])} concepts: {[c['concept_id'] for c in v['covers']]}")
    print(f"not covered by the kept videos: {out['concepts_not_covered']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
