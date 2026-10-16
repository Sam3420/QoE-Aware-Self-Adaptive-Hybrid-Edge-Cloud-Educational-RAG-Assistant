# YouTube video recommendations

Personalised educational video recommendations for a learner's current topic and knowledge gaps.

- **Search and metadata:** the official YouTube Data API v3. Only `search.list` and `videos.list` are used.
- **Transcripts:** the unofficial `youtube-transcript-api`, behind a replaceable adapter.
- **What the ranking rewards:** what the transcript shows the video actually teaches, more than popularity.

The official `captions.download` endpoint is **not** used. It only serves captions of videos the
authenticated user can manage, so it cannot provide transcripts for arbitrary public videos.

## Where it lives

| Layer | File | Role |
|---|---|---|
| Service | `services/youtube/data_api.py` | `search.list` / `videos.list` client: `part` + `fields`, quota estimate, cache, error classification |
| Service | `services/youtube/transcripts.py` | `youtube-transcript-api` adapter (manual track, then auto; language preference; failure statuses; block circuit breaker), cache wrapper, text cleaning |
| Service | `services/youtube/relevance.py` | Transcript chunking with timestamps; similarity with the **existing RAG embedder**; grounded evidence |
| Service | `services/youtube/ranking.py` | Hard filters, normalised features, weighted score with explicit missing-data rule, explanations |
| Service | `services/youtube/pipeline.py` | Two-stage pipeline, strategy model, instrumentation |
| Service | `services/youtube/metrics.py` | Outcome metrics, QoE input signals, placeholder reward |
| Decision | `agent/recommendations.py` | Learner needs, strategy selection (kNN over past runs), run/event lifecycle |
| DB | `app/db/models/recommendation.py`, `app/db/repositories/recommendation.py` | `recommendation_runs`, `recommendation_items`, `recommendation_events` |
| API | `app/api/routes/recommendations.py` | HTTP endpoints |
| UI | `frontend/views/recommendations.py` | "Video recommendations" page; "Find videos for what I missed" on the session summary |

## Setup

### 1. YouTube Data API v3 key
1. In Google Cloud Console, create or select a project, then go to **APIs & Services → Library**.
   Enable **YouTube Data API v3**.
2. Go to **Credentials → Create credentials → API key**. Restrict the key to *YouTube Data API v3*.
   Add an application restriction (for example an IP restriction) if you deploy it.
3. Put the key in `.env` at the repository root. Never commit `.env`; it is git-ignored.

```dotenv
YOUTUBE_API_KEY=your-key-here
```

Restart Streamlit and/or uvicorn after editing `.env`.

### 2. Transcript library
```powershell
.venv\Scripts\pip install youtube-transcript-api   # tested with 1.2.4
```
The adapter targets the v1.x instance API: `YouTubeTranscriptApi().list(id)` → `TranscriptList.find_manually_created_transcript / find_generated_transcript` → `Transcript.fetch()`.
If the library is missing, or `TRANSCRIPTS_ENABLED=false`, every video is ranked on metadata only and marked as such.

### 3. Embeddings
The ranking reuses the agent's embedder (`EMBEDDING_BACKEND`, default Ollama `nomic-embed-text`), so no second model is loaded.
The similarity calibration (`REC_SIMILARITY_FLOOR=0.40`, `REC_SIMILARITY_CEIL=0.85`) is for `nomic-embed-text`. Re-calibrate it if you change the embedder.

## Configuration (environment variables, `config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `YOUTUBE_API_KEY` | — | Required for search |
| `YOUTUBE_REGION_CODE` | `IN` | `regionCode` for `search.list`; also used for region-restriction filtering |
| `YOUTUBE_SAFE_SEARCH` | `strict` | `safeSearch` |
| `YOUTUBE_SEARCH_MAX_PAGES` | `1` | Pages per search (each costs 100 units) |
| `YOUTUBE_DAILY_QUOTA_UNITS` / `YOUTUBE_QUOTA_RESERVE_UNITS` | `10000` / `500` | Local quota estimate; calls stop when within the reserve |
| `YOUTUBE_SEARCH_CACHE_TTL_S` / `YOUTUBE_VIDEO_CACHE_TTL_S` | 6 h / 24 h | API response cache (max 30 days, enforced) |
| `TRANSCRIPTS_ENABLED` | `true` | Turns transcript retrieval on or off |
| `TRANSCRIPT_LANGUAGES` | `["en","hi"]` | Preference order. A manual track in any listed language is preferred over an auto track |
| `TRANSCRIPT_CACHE_TTL_S` | 24 h | Transcript retention (`0` = never stored; max 30 days) |
| `TRANSCRIPT_MAX_WORKERS` / `TRANSCRIPT_TIME_BUDGET_S` | `4` / `20` | Parallel fetches and the overall deadline (the library sets no request timeout) |
| `TRANSCRIPT_MAX_WORDS`, `TRANSCRIPT_CHUNK_WORDS`, `TRANSCRIPT_CHUNK_OVERLAP_WORDS`, `TRANSCRIPT_MAX_CHUNKS_PER_VIDEO` | 12000, 120, 20, 40 | Bound transcript processing |
| `REC_WEIGHTS` | see below | JSON object of feature weights. Names, signs and sum are validated at startup |
| `REC_COUNT` | `5` | Recommendations returned |
| `REC_MIN_DURATION_S` | `60` | Hard filter (excludes Shorts-length clips) |
| `REC_PREFERRED_DURATION_MIN_S` / `MAX_S` | 240 / 1200 | Duration preference |
| `REC_REQUIRE_EMBEDDABLE`, `REC_REQUIRE_LANGUAGE_MATCH` | `false` | Optional hard filters |
| `REC_MISSING_FEATURE_PRIOR` | `0.5` | Neutral value for missing evidence |
| `REC_DEFAULT_STRATEGY_ID` | `R2` | Baseline strategy during cold start |

## Pipeline

1. **Learner needs** (`agent/recommendations.py`):
   - **Language:** the profile language (`english` → `en`).
   - **Competency:** the latest pre/post-test for the topic, otherwise the profile.
   - **Knowledge gaps:** the concepts of MCQs answered wrongly in that test.
   - **Objective:** the learner's text, otherwise "topic: gap concepts".
2. **Strategy selection:** a kNN over past runs (see below).
3. **Search** (`search.list`, `part=snippet`, `type=video`, `q`, `maxResults`, `relevanceLanguage`, `regionCode`, `safeSearch`, optional `videoDuration`/`videoCaption`) with duplicate removal.
   The query is the objective plus up to two gap concepts plus a level hint ("explained simply" for beginners).
4. **Metadata** (`videos.list`, `part=snippet,contentDetails,statistics,status`, ≤50 ids per call, with `fields` trimming).
   Missing fields stay `None`. Videos that `videos.list` no longer returns (deleted or private) are dropped.
5. **Hard filters:**
   - duplicates
   - videos this learner completed or dismissed (unless *revision* is ticked)
   - non-public videos, or videos not processed
   - live or upcoming broadcasts
   - videos shorter than `REC_MIN_DURATION_S`
   - videos region-blocked for `YOUTUBE_REGION_CODE`
   - optionally, non-embeddable videos and videos in another language

   Unknown values never filter a video out.
6. **Stage 1 (cheap):** a metadata pre-score (topic similarity of title + tags + description, plus the other metadata features).
7. **Stage 2 (bounded):** transcripts only for the top `transcript_budget` pre-ranked videos, fetched in parallel within `TRANSCRIPT_TIME_BUDGET_S`.
   Each transcript is cleaned (caption noise removed; terms and formulas kept), truncated to `TRANSCRIPT_MAX_WORDS`, and split into overlapping word windows with interpolated start/end times.
   Long videos are sampled evenly down to `TRANSCRIPT_MAX_CHUNKS_PER_VIDEO`. All chunks are embedded in one batch with the shared embedder; no transcript is sent to an LLM.
8. **Final ranking and response.**

### Transcript-based relevance
- `transcript_relevance` = the mean of the top-3 normalised chunk similarities to the objective.
- **Evidence:** every item is taken from real transcript text:
  - the best-matching excerpts, with their time ranges
  - each gap concept that is *covered* (some chunk ≥ `REC_CONCEPT_COVER_THRESHOLD`), with its first timestamp
  - concepts *not found*
  - example cues ("for example", "suppose", …) and step cues ("first,", "next,", "step 2", …) inside on-topic chunks
  - prerequisite cues ("recall that", "you should know")
  - on-topic seconds, counted as the union of on-topic chunk time ranges
- A video whose title matches but whose transcript does not is removed: transcript relevance below the strategy's `min_transcript_relevance`.
  In a live check, a video given "Photosynthesis lesson" metadata whose real transcript is a pop song scored 0.11 and was removed.

### Score and missing data
Every feature is normalised to [0, 1] or is `None` (unknown):

| Feature | Definition |
|---|---|
| `topic_similarity` | calibrated cosine(objective, title + tags + description) |
| `transcript_relevance` | see above |
| `difficulty_match` | 1 − \|video level − learner level\|. The video level is a **heuristic** from wording ("basics" 0, "class 10" 0.5, "derivation" 1); `None` when there is no cue |
| `language_match` | defaultAudioLanguage / transcript language vs learner language |
| `duration_match` | 1 inside the preferred range, falling linearly to 0 at 0 s and at 2× the maximum |
| `caption_match` | 1 if a transcript or manual captions exist; 0 only when retrieval confirmed none. `caption=false` alone is unknown, because auto captions may exist |
| `historical_outcome` | Beta(1,1)-smoothed accept ratio and mean rating/5 from past events on this video |
| `popularity` | log views scaled within the candidate set, blended with like/view ratio (secondary) |

```
observed = Σ_available w·x / Σ_available w
coverage = Σ_available w / Σ w
score    = coverage·observed + (1 − coverage)·prior
```

- Missing evidence moves a score toward the neutral prior, not toward 0. A missing transcript or hidden statistics is never treated as low quality.
- The same rule discounts weaker evidence. A metadata-only video with the same observed score as a transcript-backed one ranks lower when that score is above the prior.
- Each recommendation reports `evidence_level` (`transcript` | `metadata`) and `evidence_coverage`.
- **Default weights:** topic 0.25, transcript 0.30, difficulty 0.10, language 0.10, duration 0.08, caption 0.05, history 0.07, popularity 0.05.

### Metadata-only fallback
A video is ranked on metadata only when any of these apply:
- it falls outside the transcript budget
- the strategy is R1
- transcripts are disabled
- the fetch failed or timed out
- the video has no captions

Its explanation says "Ranked on title and description only … no transcript evidence (*reason*)". Metadata-only videos must reach `min_topic_similarity`.

If a whole service fails, the run degrades instead of failing:

| Failure | Effect | Run status |
|---|---|---|
| `videos.list` fails | Ranking uses search snippets only | `degraded` |
| The embedder fails | Topic match uses keyword overlap | `degraded` |
| Search fails | The run is stored with no recommendations and the error kind; the UI shows a Retry button | `failed` |

## Strategies and adaptive selection

The strategies are fixed, predefined, executable configurations:

| id | name | candidates | transcripts | notes |
|---|---|---|---|---|
| R1 | metadata-fast | 10 | 0 | no transcript requests, lowest latency and load |
| R2 | balanced (baseline) | 15 | 4 | default weights |
| R3 | transcript-deep | 25 | 8 | transcript weight 0.40, `min_transcript_relevance` 0.30 |

- **Policy selection (before the run):** the context `{rtt_ms, cpu, mem, competency_num, last_qoe}` is measured exactly as the Q&A agent's `collect_context` measures it. Then `agent.policy_selection.choose_policy` (the same kNN + ε-exploration + tie-to-default code as the Q&A policies) runs over past recommendation runs that have a reward.
  - With fewer than `MIN_EXPERIENCES` rewarded runs, the baseline `REC_DEFAULT_STRATEGY_ID` is used.
  - The decision reason is stored on the run.
- **Observed outcomes (after the run):** learner events plus the topic's assessments around the run.
- **Reward:** recomputed after every event from the measured outcome signals. `metrics.reward_stub` is a **placeholder**:
  - It is `None` until the learner accepts, dismisses or rates something, so runs without judgements never influence selection.
  - It uses acceptance only after an accept/dismiss, and satisfaction only after a rating.
  - The formula is 0.8 × mean of those (plus learning gain when available) + 0.2 × latency quality.
- Recommendation runs are stored in `recommendation_runs`, not in the Q&A `experiences` table. That table's `policy_id` references the Q&A policies P1–P5, so mixing them would let either selector pick the other's actions. The run row has the same experience shape (state, strategy, outcome, reward).
- No adaptive improvement is claimed until runs with real feedback exist and are compared with the R2 baseline.

## Feedback and evaluation

Events (`recommendation_events`):
- `generated`: automatic
- `displayed`: logged once per item when the UI renders it
- `opened`: "Watch here" (embedded player) or "Open on YouTube"
- `accepted` ("Helpful") and `dismissed` ("Not relevant")
- `rated`: 1–5 stars
- `completed`: "I finished it", **self-reported**

The app cannot observe watch time, so `opened` never means "watched".

| Metric | Definition | Available when |
|---|---|---|
| Click-through rate | items opened / items shown | an item was shown (shown = displayed ∪ interacted) |
| Acceptance rate | items accepted / items shown | same |
| Dismissal rate | items dismissed / items shown | same |
| Completion rate | self-reported completions / items opened | an item was opened |
| Learner satisfaction | mean latest rating per item / 5 | a rating exists |
| Precision@K | accepted / explicitly labelled items in the top K (unlabelled excluded) | a label exists |
| Learning gain | post − pre score for the topic | a pre-test ≤ run time and a post-test after it |
| Normalised learning gain | (post − pre) / (1 − pre) | same, with pre < 1 |
| Normalised post-test | post score | a post-test after the run |

Learning gain is measured around the run and is **not attributed to a single video**.

## QoE integration

The project has no QoE computation service yet: session evaluation is `evaluate_session_stub`. This feature therefore produces normalised QoE **inputs** per component and does not introduce a competing overall QoE formula. They are stored on each run as `qoe_signals`:

- **EQ:** learning gain ((normalised gain + 1) / 2), normalised post-test
- **AQ:** latency quality (1 − total latency / `REC_LATENCY_BUDGET_MS`), API reliability (1 − failure rate), transcript success rate
- **NQ:** RTT quality (1 − RTT / `RTT_MAX_MS`)
- **UX:** acceptance rate, satisfaction, CTR, completion rate. Feedback sentiment is `None`, because no sentiment analyser exists.

Pipeline instrumentation per run (`metrics`):
- latency of search, metadata, transcripts, embedding, ranking and the total
- API calls, failures, quota units and bytes
- cache hits and misses
- candidates found and filtered
- transcripts attempted, OK and from cache
- the number of embeddings
- CPU and memory at the start

Derived rates: API failure rate, transcript success rate, candidate filtering rate and cache hit rate.

## HTTP API (FastAPI)

```http
POST /learners/{student_id}/recommendations
{"topic_id": "photosynthesis", "objective": "how plants make glucose", "concepts": ["chlorophyll"], "revision": false}
```
`objective` and `concepts` are optional; the defaults come from the learner's latest test. `session_id` links the run to a learning session.

Response (abridged, from the synthetic test catalogue):
```json
{
  "id": "74ecba28-…", "status": "ok", "strategy_id": "R2",
  "decision_reason": "cold start: 0 experience(s) < MIN_EXPERIENCES=3 → default R2",
  "query": "photosynthesis chlorophyll light energy glucose",
  "context": {"rtt_ms": 180.0, "cpu": 0.35, "mem": 0.55, "competency_num": 0.0, "last_qoe": 0.5},
  "recommendations": [{
    "rank": 1, "item_id": "db8823b2-…", "video_id": "goodTeach01", "title": "Photosynthesis explained simply",
    "url": "https://www.youtube.com/watch?v=goodTeach01", "thumbnail_url": "https://i.ytimg.com/vi/goodTeach01/hq.jpg",
    "channel_title": "Fake Channel", "duration_s": 570, "published_at": "2024-01-02T03:04:05Z", "embeddable": true,
    "score": 0.8556, "evidence_level": "transcript", "evidence_coverage": 0.83,
    "score_breakdown": {"topic_similarity": 1.0, "transcript_relevance": 0.9156, "difficulty_match": null,
                        "language_match": 1.0, "duration_match": 1.0, "caption_match": 1.0,
                        "historical_outcome": null, "popularity": 0.318},
    "transcript_status": "ok", "caption_language": "en", "caption_type": "manual",
    "explanation": "Transcript (en, manual captions) addresses the objective at 00:33: “Next, carbon dioxide and light energy make glucose…”. Covers 2 of 2 target concepts (chlorophyll at 00:00, glucose at 00:33). Includes examples and step-by-step explanation in the relevant parts. 10 min, within the preferred 4–20 min.",
    "transcript_evidence": {"relevance": 0.9156,
      "excerpts": [{"start": 33.33, "end": 50.0, "similarity": 1.0, "text": "Next, carbon dioxide and light energy make glucose. …"}],
      "concepts_covered": {"chlorophyll": 0.0, "glucose": 33.33}, "concepts_missing": [],
      "example_timestamps": [0.0, 20.0], "step_timestamps": [20.0, 33.33], "on_topic_seconds": 50.0, "n_chunks": 3}
  }, {
    "rank": 2, "video_id": "popularNoTr", "score": 0.71, "evidence_level": "metadata", "evidence_coverage": 0.53,
    "transcript_status": "transcripts_disabled",
    "explanation": "Ranked on title and description only (match 1.00); no transcript evidence (transcripts disabled). 12 min, within the preferred 4–20 min."
  }],
  "metrics": {"timings": {"search_ms": 0.8, "metadata_ms": 0.5, "transcript_ms": 1.3, "embedding_ms": 0.5,
                          "ranking_ms": 0.4, "total_ms": 5.1},
              "quota_units": 101, "rates": {"api_failure_rate": 0.0, "transcript_success_rate": 0.5,
                                            "candidate_filtering_rate": 0.6667, "cache_hit_rate": null}}
}
```

```http
POST /recommendations/{run_id}/events      {"event_type": "rated", "item_id": "db8823b2-…", "value": {"rating": 4}}
→ 201 {"event": {...}, "outcome": {"metrics": {...}, "qoe_signals": {...}, "reward": 0.84, "reward_is_placeholder": true}}
GET  /recommendations/{run_id}             the stored run with items
GET  /recommendations/{run_id}/metrics     recompute and return the outcome metrics
GET  /learners/{student_id}/recommendations   past runs
```

The API returns 404 for an unknown learner, session, run or item, and 422 for invalid input (an empty topic, a rating outside 1–5, an unknown event type).
A search failure (bad key, quota, network) returns 200 with `status: "failed"` and `errors`, so the stored run can be inspected and the request retried.

## Tests

`pytest` covers this feature with no network, credentials or quota. YouTube calls go through `httpx.MockTransport`, and transcripts come from scripted fakes.

| File | What it covers |
|---|---|
| `tests/test_youtube_data_api.py` | Supported parameters, `fields`, pagination and duplicate removal, 50-id batching, ISO-8601 parsing, error classification (quota/key/forbidden/bad request/timeout/network/non-JSON), local quota stop, cache and TTL |
| `tests/test_youtube_transcripts.py` | Manual-over-auto preference, language order, every failure status, block circuit breaker, a real library exception class, transcript cache, cleaning, truncation, chunk timestamps, overlap and bounds |
| `tests/test_youtube_ranking.py` | Weight validation, missing-feature rule, every feature, hard filters, query building, and a synthetic catalogue with a known correct ranking. Also the metadata-only strategy, transcripts disabled, exclusions, history lookup, failure degradation, the transcript time budget and instrumentation |
| `tests/test_recommendation_coordinator.py` | Persistence, gap concepts from the pre-test, events → outcomes → reward, dismissed exclusion and revision, kNN strategy choice, learning gain around the run, validation, failed runs, metric definitions |
| `tests/test_recommendation_api.py` | HTTP round trip and error codes |
| `tests/test_recommendations_page.py` | Streamlit `AppTest`: search → cards → "Helpful" event → reward; the missing-key state |

## Limitations

- **Transcripts are unofficial.** `youtube-transcript-api` depends on YouTube internals that can change without notice, and YouTube may block requests (for example from cloud IPs).
  - Blocks, age gates and login requirements are reported and never bypassed.
  - After a block, retrieval stops for the rest of the process.
  - Check the library's licence (MIT) and YouTube's Terms of Service before deploying.
- **Quota.**
  - **Cost:** each search costs 100 units and each `videos.list` call 1 unit. The default daily quota is 10,000 units, about 90 uncached searches after the reserve.
  - **Tracking:** the local tracker is an estimate. It resets at midnight UTC−8 and ignores daylight saving; the API's own `quotaExceeded` error is still handled.
- **Retention.** API responses and transcripts are cached only up to their TTLs (≤ 30 days, enforced) and deleted when read after expiry. Set `TRANSCRIPT_CACHE_TTL_S=0` to keep no transcripts.
  Stored recommendation items keep titles, descriptions and evidence excerpts for evaluation, so review retention for deployment.
- **Heuristics.**
  - Difficulty and example/step/prerequisite cues are lexical and English-centred.
  - Similarity calibration is tuned for `nomic-embed-text`, and Hindi transcripts with an English objective will score lower.
- **Feedback has limits.** Watch time and completion cannot be observed: completion is self-reported, and "opened" is not "watched".
- **Placeholder reward.** The reward is a placeholder, and QoE is not combined into an overall score until Person B's metrics/QoE service exists.
- **Live latency.** The first request after Ollama starts includes model load (about 30 s); after that, embedding 36 chunks took about 0.9 s.
  Transcript fetches varied between 1.3 s and 12 s per video, which is why fetches run in parallel within a time budget.
