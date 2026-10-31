"""Two-stage recommendation pipeline (service layer: does the work, makes no strategy decisions).

1. search.list → candidates; videos.list → metadata (one call per 50 ids); hard filters.
2. Cheap metadata pre-ranking; transcripts only for the top `transcript_budget` candidates.
3. Transcript chunks → shared embedder → evidence; final weighted ranking with explanations.

Failures degrade instead of aborting: no metadata → search snippets only; no embedder → keyword
overlap; no transcript → metadata-only evidence. Only a failed search returns no recommendations.
"""
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

from services.vector_store import EmbeddingError
from services.youtube.data_api import VIDEO_CAPTIONS, VIDEO_DURATIONS, YouTubeApiError, YouTubeDataClient
from services.youtube.models import (
    FilteredVideo,
    PipelineMetrics,
    Recommendation,
    RecommendationResult,
    StageTimings,
    TranscriptRecord,
    TranscriptStatus,
)
from services.youtube.ranking import (
    LearnerSignals,
    RankingWeights,
    ScoredVideo,
    caption_match,
    combine,
    difficulty_match,
    duration_match,
    estimate_video_level,
    explain,
    hard_filter,
    historical_outcome,
    language_match,
    popularity_scores,
)
from services.youtube.relevance import RelevanceScorer, chunk_transcript, keyword_overlap, metadata_text
from services.youtube.transcripts import TranscriptFetcher, timed_fetch, truncate_record
from telemetry import tracing

logger = logging.getLogger(__name__)
_LEVEL_HINTS = {"beginner": "explained simply", "intermediate": "", "advanced": "in depth"}
# no track the session could index for answers: such a video cannot be taught from
UNUSABLE_TRANSCRIPT = {TranscriptStatus.NO_TRANSCRIPT, TranscriptStatus.TRANSCRIPTS_DISABLED}


class RecommendationStrategy(BaseModel):
    """Runtime configuration one recommendation run executes with (every field is an executable action)."""

    strategy_id: str
    name: str
    max_candidates: int = Field(ge=1, le=50)  # search.list maxResults (one page = 100 quota units)
    transcript_budget: int = Field(ge=0, le=20)  # transcripts fetched for the best pre-ranked candidates
    metadata_only: bool = False
    video_duration: str = "any"  # search.list videoDuration
    video_caption: str = "any"  # search.list videoCaption
    weights: dict[str, float] | None = None  # None → REC_WEIGHTS
    min_topic_similarity: float = Field(default=0.15, ge=0.0, le=1.0)  # relevance threshold (metadata-only)
    min_transcript_relevance: float = Field(default=0.20, ge=0.0, le=1.0)  # relevance threshold (transcript)
    duration_range_s: tuple[int, int] | None = None  # None → learner/config preference
    description: str = ""

    @field_validator("weights")
    @classmethod
    def _valid_weights(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        """Reject unknown features or invalid weights when the strategy is defined."""
        if value is not None:
            RankingWeights.from_mapping(value)
        return value

    @field_validator("video_duration")
    @classmethod
    def _valid_duration(cls, value: str) -> str:
        if value not in VIDEO_DURATIONS:
            raise ValueError(f"video_duration must be one of {sorted(VIDEO_DURATIONS)}")
        return value

    @field_validator("video_caption")
    @classmethod
    def _valid_caption(cls, value: str) -> str:
        if value not in VIDEO_CAPTIONS:
            raise ValueError(f"video_caption must be one of {sorted(VIDEO_CAPTIONS)}")
        return value


class RecommendationRequest(BaseModel):
    """What the learner needs right now."""

    topic: str = Field(min_length=1)
    objective: str = Field(min_length=1)  # text the videos are compared with
    concepts: list[str] = Field(default_factory=list)  # knowledge-gap concepts to look for in transcripts
    language_code: str | None = None  # "en", "hi"
    competency: str | None = None
    competency_num: float | None = None
    excluded: dict[str, str] = Field(default_factory=dict)  # video_id → "completed" | "dismissed"
    history: dict[str, dict[str, Any]] = Field(default_factory=dict)  # video_id → past feedback stats
    count: int = Field(default=5, ge=1, le=20)


@dataclass
class PipelineSettings:
    """Static settings (from config.py), separate from the per-run strategy."""

    region_code: str | None
    safe_search: str
    max_pages: int
    transcript_languages: list[str]
    transcripts_enabled: bool
    transcript_max_words: int
    chunk_words: int
    chunk_overlap: int
    max_chunks_per_video: int
    min_duration_s: int
    duration_min_s: int
    duration_max_s: int
    require_embeddable: bool
    require_language_match: bool
    prior: float
    default_weights: dict[str, float] = field(default_factory=dict)
    transcript_workers: int = 4
    transcript_time_budget_s: float = 20.0
    require_transcript: bool = False  # recommend only videos with a track the session can index (lists tracks)
    # The track listings for require_transcript: all candidates in one parallel round, started together with the
    # stage-2 transcript fetches; listings not answered within the budget keep their video (cannot tell).
    caption_check_workers: int = 6
    caption_check_budget_s: float = 6.0


def build_query(request: RecommendationRequest) -> str:
    """Search query: objective (or topic) plus up to two gap concepts and a level hint."""
    base = request.objective if len(request.objective) <= 80 else request.topic
    extra = [c for c in request.concepts[:2] if c.lower() not in base.lower()]
    hint = _LEVEL_HINTS.get(request.competency or "", "")
    return " ".join(p for p in [base, *extra, hint] if p)[:200]


class RecommendationPipeline:
    """Runs one recommendation request with one strategy."""

    def __init__(self, *, data_client: YouTubeDataClient, transcripts: TranscriptFetcher,
                 scorer: RelevanceScorer, settings: PipelineSettings, monitor: Any | None = None) -> None:
        self.data_client = data_client
        self.transcripts = transcripts
        self.scorer = scorer
        self.settings = settings
        self.monitor = monitor

    def run(self, request: RecommendationRequest, strategy: RecommendationStrategy,
            history_lookup: Callable[[list[str]], dict[str, dict[str, Any]]] | None = None) -> RecommendationResult:
        """Search, enrich, filter, rank; never raises for external failures.

        history_lookup(video_ids) returns past feedback per video once the candidates are known.
        """
        s = self.settings
        started = time.perf_counter()
        metrics, timings, errors = PipelineMetrics(), StageTimings(), []
        cache = self.data_client.cache
        hits0, misses0 = (cache.hits, cache.misses) if cache else (0, 0)
        self.data_client.reset_stats()
        self.scorer.embed_ms, self.scorer.embeddings = 0.0, 0
        if self.monitor is not None:
            try:
                metrics.cpu, metrics.mem = self.monitor.cpu_mem()
            except Exception:  # monitoring is optional
                pass
        query = build_query(request)
        lo, hi = strategy.duration_range_s or (s.duration_min_s, s.duration_max_s)
        learner = LearnerSignals(request.language_code, request.competency_num, lo, hi)
        weights = RankingWeights.from_mapping(strategy.weights or s.default_weights)

        def finish(status: str, recs: list[Recommendation], filtered: list, extra: dict | None = None) -> RecommendationResult:
            stats = self.data_client.stats
            timings.search_ms = round(stats.search_ms, 1) if stats.calls or stats.search_ms else timings.search_ms
            timings.metadata_ms = round(stats.metadata_ms, 1) if stats.metadata_ms else timings.metadata_ms
            timings.embedding_ms = round(self.scorer.embed_ms, 1) if self.scorer.embeddings else None
            timings.total_ms = round((time.perf_counter() - started) * 1000, 1)
            metrics.timings = timings
            metrics.api_calls, metrics.api_failures = stats.calls, stats.failures
            metrics.quota_units, metrics.api_bytes = stats.units, stats.bytes
            metrics.embeddings = self.scorer.embeddings
            if cache:
                metrics.cache_hits, metrics.cache_misses = cache.hits - hits0, cache.misses - misses0
            return RecommendationResult(query=query, objective=request.objective, recommendations=recs,
                                        filtered=filtered, metrics=metrics, errors=errors, status=status,
                                        extra=extra or {})

        # ---- stage 1: search + metadata ------------------------------------------------------------
        try:
            candidates = self.data_client.search(
                query, max_results=strategy.max_candidates, max_pages=s.max_pages,
                relevance_language=request.language_code, region_code=s.region_code,
                video_duration=strategy.video_duration, video_caption=strategy.video_caption,
                safe_search=s.safe_search)
        except YouTubeApiError as exc:
            errors.append(f"search failed: {exc}")
            return finish("failed", [], [], {"error_kind": exc.kind})
        metrics.candidates_found = len(candidates)
        if not candidates:
            return finish("empty", [], [])

        degraded = False
        try:
            details = self.data_client.get_videos([c.video_id for c in candidates])
            candidates = [self.data_client.merge_details(c, details.get(c.video_id)) for c in candidates]
            missing = [c.video_id for c in candidates if not c.has_details]
            if missing:  # deleted/private videos are omitted by videos.list
                errors.append(f"videos.list returned no details for {len(missing)} video(s); they were excluded")
                candidates = [c for c in candidates if c.has_details]
        except YouTubeApiError as exc:
            degraded = True
            errors.append(f"metadata unavailable, ranking on search snippets only: {exc}")

        kept, filtered = hard_filter(
            candidates, region_code=s.region_code, min_duration_s=s.min_duration_s,
            require_embeddable=s.require_embeddable,
            require_language=request.language_code if s.require_language_match else None,
            excluded=request.excluded)
        if history_lookup is not None and kept:
            request = request.model_copy(update={"history": {**request.history,
                                                             **history_lookup([c.video_id for c in kept])}})

        # ---- similarity to the objective -------------------------------------------------------------
        ranking_started = time.perf_counter()
        objective_vec, concept_vecs, topic_sim = None, {}, {}
        try:
            objective_vec = self.scorer.embed_query(request.objective)
            topic_sim = self.scorer.topic_similarity(objective_vec, kept)
            if request.concepts:
                vecs = self.scorer.embed_documents(request.concepts)
                concept_vecs = dict(zip(request.concepts, vecs))
        except EmbeddingError as exc:
            degraded = True
            errors.append(f"embedder unavailable, using keyword overlap for topic match: {exc}")
            topic_sim = {c.video_id: round(keyword_overlap(request.objective, metadata_text(c)), 4) for c in kept}
        popularity = popularity_scores(kept)

        scored = []
        for c in kept:
            level = estimate_video_level(metadata_text(c))
            features = {
                "topic_similarity": topic_sim.get(c.video_id),
                "transcript_relevance": None,
                "difficulty_match": difficulty_match(level, request.competency_num),
                "language_match": language_match(c, None, request.language_code),
                "duration_match": duration_match(c.duration_s, lo, hi),
                "caption_match": caption_match(c, None),
                "historical_outcome": historical_outcome(request.history.get(c.video_id)),
                "popularity": popularity.get(c.video_id),
            }
            video = ScoredVideo(c, features, notes={"video_level": level})
            video.score, video.coverage = combine(features, weights, s.prior)
            scored.append(video)
        scored.sort(key=lambda v: v.score, reverse=True)
        ranking_ms = (time.perf_counter() - ranking_started) * 1000 - self.scorer.embed_ms

        # ---- stage 2: transcripts for the most promising candidates -----------------------------------
        budget = 0 if strategy.metadata_only or not s.transcripts_enabled else strategy.transcript_budget
        not_attempted = TranscriptStatus.NOT_ATTEMPTED if s.transcripts_enabled else TranscriptStatus.DISABLED_BY_CONFIG
        selected_chunks: list[tuple[ScoredVideo, list]] = []
        for video in scored[budget:]:
            video.transcript = TranscriptRecord(video_id=video.candidate.video_id, status=not_attempted)
        # the learner's language only, as when the chosen video is indexed for answers (agent.knowledge.
        # transcript_languages): a transcript the session cannot use must not count as transcript evidence
        languages = [request.language_code] if request.language_code else list(s.transcript_languages)
        checking_tracks = s.require_transcript and s.transcripts_enabled
        # the other candidates' track listings run meanwhile (the final ranking needs them, not stage 2)
        prechecked = (self._start_track_checks([v.candidate.video_id for v in scored[budget:]], languages)
                      if checking_tracks else dict)
        records =self._fetch_transcripts([v.candidate.video_id for v in scored[:budget]], languages) if budget else {}
        timings.transcript_ms = records.pop("__elapsed_ms__", None) if budget else None
        for video in scored[:budget]:
            record = records[video.candidate.video_id]
            metrics.transcripts_attempted += 1
            metrics.transcripts_from_cache += int(record.from_cache)
            if record.status == TranscriptStatus.OK:
                metrics.transcripts_ok += 1
                record = truncate_record(record, s.transcript_max_words)
                chunks = chunk_transcript(record, words_per_chunk=s.chunk_words, overlap=s.chunk_overlap,
                                          max_chunks=s.max_chunks_per_video)
                if chunks:
                    selected_chunks.append((video, chunks))
            video.transcript = record

        if selected_chunks and objective_vec is not None:
            try:
                all_chunks = [ch for _, chunks in selected_chunks for ch in chunks]
                all_vecs = self.scorer.embed_documents([ch.text for ch in all_chunks])  # one batch for all videos
                offset = 0
                for video, chunks in selected_chunks:
                    vecs = all_vecs[offset:offset + len(chunks)]
                    offset += len(chunks)
                    video.evidence = self.scorer.transcript_evidence(objective_vec, chunks, vecs, concept_vecs)
                    video.features["transcript_relevance"] = video.evidence.relevance
            except EmbeddingError as exc:
                degraded = True
                errors.append(f"transcript embedding failed; transcripts not used for ranking: {exc}")

        # ---- final ranking --------------------------------------------------------------------------
        with tracing.observation("rank-videos", as_type="span",
                                 input={"candidates": len(scored), "weights": weights.weights},
                                 metadata={"strategy_id": strategy.strategy_id}) as rank_obs:
            final_started = time.perf_counter()
            results = []
            for video in scored:
                video.features["caption_match"] = caption_match(video.candidate, video.transcript)
                video.features["language_match"] = language_match(video.candidate, video.transcript, request.language_code)
                video.score, video.coverage = combine(video.features, weights, s.prior)
                rel = video.features["transcript_relevance"]
                topic = video.features["topic_similarity"]
                if rel is not None and rel < strategy.min_transcript_relevance:
                    filtered.append(_filtered(video, f"transcript relevance {rel:.2f} < {strategy.min_transcript_relevance}"))
                elif rel is None and topic is not None and topic < strategy.min_topic_similarity:
                    filtered.append(_filtered(video, f"topic similarity {topic:.2f} < {strategy.min_topic_similarity}"))
                else:
                    results.append(video)
            results.sort(key=lambda v: (v.score, v.coverage), reverse=True)
            best = (self._with_usable_transcripts(results, languages, request.count, filtered, prechecked())
                    if checking_tracks else results[:request.count])
            recs = [self._to_recommendation(rank, v, request, learner) for rank, v in enumerate(best, 1)]
            rank_obs.update(output={"ranked": [{"video_id": r.video_id, "score": r.score,
                                                "evidence": r.evidence_level} for r in recs],
                                    "filtered": [f.model_dump() for f in filtered]})
        timings.ranking_ms = round(max(ranking_ms, 0.0) + (time.perf_counter() - final_started) * 1000, 1)
        metrics.candidates_filtered = len(filtered)

        status = "empty" if not recs else "degraded" if degraded else "ok"
        return finish(status, recs, filtered, {"weights": weights.weights, "duration_range_s": [lo, hi]})

    def warm(self, request: RecommendationRequest, strategy: RecommendationStrategy) -> dict[str, Any]:
        """Fill the caches a later run() of this request reads (search, metadata, caption verdicts), without
        ranking or storing anything; returns counts for the log. Raises YouTubeApiError (e.g. no quota)."""
        s = self.settings
        query = build_query(request)
        candidates = self.data_client.search(
            query, max_results=strategy.max_candidates, max_pages=s.max_pages,
            relevance_language=request.language_code, region_code=s.region_code,
            video_duration=strategy.video_duration, video_caption=strategy.video_caption, safe_search=s.safe_search)
        details = self.data_client.get_videos([c.video_id for c in candidates]) if candidates else {}
        candidates = [self.data_client.merge_details(c, details.get(c.video_id)) for c in candidates]
        kept, _ = hard_filter(
            [c for c in candidates if c.has_details], region_code=s.region_code, min_duration_s=s.min_duration_s,
            require_embeddable=s.require_embeddable,
            require_language=request.language_code if s.require_language_match else None, excluded=request.excluded)
        verdicts, downloaded = {}, 0
        if s.require_transcript and s.transcripts_enabled and kept:
            # in the background, so a longer wait than a learner's run allows; the few usable transcripts are
            # downloaded too (stage 2 and indexing a chosen video then read them from the cache)
            languages = [request.language_code] if request.language_code else list(s.transcript_languages)
            verdicts = self._check_tracks([c.video_id for c in kept], languages, budget_s=4 * s.caption_check_budget_s)
            usable = [vid for vid, status in verdicts.items() if status == TranscriptStatus.OK][:strategy.max_candidates]
            if usable:
                pool = ThreadPoolExecutor(max_workers=min(s.transcript_workers, len(usable)), thread_name_prefix="warm")
                futures = [pool.submit(self.transcripts.fetch, vid, languages) for vid in usable]
                done, _ = wait(futures, timeout=2 * s.transcript_time_budget_s)
                pool.shutdown(wait=False, cancel_futures=True)
                downloaded = sum(f.exception() is None and f.result().status == TranscriptStatus.OK for f in done)
        return {"query": query, "candidates": len(candidates), "kept": len(kept), "captions_checked": len(verdicts),
                "usable": sum(v not in UNUSABLE_TRANSCRIPT for v in verdicts.values()),
                "transcripts_downloaded": downloaded}

    def _fetch_transcripts(self, video_ids: list[str], languages: Sequence[str]) -> dict[str, Any]:
        """Fetch transcripts in parallel within TRANSCRIPT_TIME_BUDGET_S; late ones count as failed.

        The transcript library sets no request timeout, so the budget is what bounds latency. A
        late fetch keeps running in its worker thread, but its result is ignored.
        """
        s = self.settings
        started = time.perf_counter()
        pool = ThreadPoolExecutor(max_workers=min(s.transcript_workers, len(video_ids)), thread_name_prefix="transcript")
        # Worker threads do not inherit the tracing context: bind each task to a copy of it, so the
        # transcript observations nest under this request instead of starting orphan traces.
        futures = {pool.submit(tracing.with_current_context(self._traced_fetch), vid, languages): vid
                   for vid in video_ids}
        done, _ = wait(futures, timeout=s.transcript_time_budget_s)
        pool.shutdown(wait=False, cancel_futures=True)
        out: dict[str, Any] = {}
        for future, vid in futures.items():
            if future in done and future.exception() is None:
                out[vid] = future.result()[0]
            else:
                reason = (f"exceeded the {s.transcript_time_budget_s:.0f}s transcript budget" if future not in done
                          else f"{type(future.exception()).__name__}: {future.exception()}")
                out[vid] = TranscriptRecord(video_id=vid, status=TranscriptStatus.FAILED, error=reason)
        out["__elapsed_ms__"] = round((time.perf_counter() - started) * 1000, 1)
        return out

    def _with_usable_transcripts(self, ranked: list[ScoredVideo], languages: Sequence[str], count: int,
                                 filtered: list[FilteredVideo],
                                 prechecked: dict[str, TranscriptStatus] | None = None) -> list[ScoredVideo]:
        """The best `count` videos, in rank order, that have a transcript track the session can index for answers.

        Videos whose transcript was fetched in stage 2 are already known, the others' tracks were listed during
        stage 2 (`prechecked`; nothing downloaded); any still unknown are listed now, all in one parallel round.
        Videos with no track in `languages` go to `filtered`; a check that cannot tell (network error, block, time
        budget) keeps the video.
        """
        verdicts = dict(prechecked or {})
        unknown = [v.candidate.video_id for v in ranked if v.candidate.video_id not in verdicts
                   and (v.transcript is None or v.transcript.status == TranscriptStatus.NOT_ATTEMPTED)]
        if unknown:
            verdicts.update(self._check_tracks(unknown, languages))
        kept: list[ScoredVideo] = []
        for video in ranked:
            status = verdicts.get(video.candidate.video_id, video.transcript.status if video.transcript else None)
            if status in UNUSABLE_TRANSCRIPT:
                filtered.append(_filtered(video, f"no transcript track in {'/'.join(languages)} "
                                                 f"({status.value.replace('_', ' ')})"))
            elif len(kept) < count:
                kept.append(video)
        return kept

    def _check_tracks(self, video_ids: list[str], languages: Sequence[str],
                      budget_s: float | None = None) -> dict[str, TranscriptStatus]:
        """Track listings in parallel within CAPTION_CHECK_BUDGET_S; videos not answered in time are left out."""
        return self._start_track_checks(video_ids, languages, budget_s)()

    def _start_track_checks(self, video_ids: list[str], languages: Sequence[str],
                            budget_s: float | None = None) -> Callable[[], dict[str, TranscriptStatus]]:
        """Start the track listings in the background; the returned function collects the verdicts, waiting until
        CAPTION_CHECK_BUDGET_S (or budget_s) after the start at most (listings still queued then are cancelled;
        running ones finish and their cached verdicts serve the next run)."""
        check = getattr(self.transcripts, "available", None)
        if check is None or not video_ids:
            return dict
        s = self.settings
        budget = s.caption_check_budget_s if budget_s is None else budget_s
        started = time.perf_counter()
        pool = ThreadPoolExecutor(max_workers=min(s.caption_check_workers, len(video_ids)), thread_name_prefix="captions")
        futures = {pool.submit(tracing.with_current_context(check), vid, languages): vid for vid in video_ids}
        pool.shutdown(wait=False)

        def collect() -> dict[str, TranscriptStatus]:
            done, pending = wait(futures, timeout=max(0.0, budget - (time.perf_counter() - started)))
            for future in pending:
                future.cancel()
            return {vid: f.result() for f, vid in futures.items() if f in done and not f.exception()}

        return collect

    def _traced_fetch(self, video_id: str, languages: Sequence[str]) -> tuple[TranscriptRecord, float]:
        """One transcript fetch as a `tool` observation (WARNING when no usable transcript)."""
        with tracing.observation("fetch-transcript", as_type="tool",
                                 input={"video_id": video_id, "languages": list(languages)},
                                 metadata={"source": "youtube-transcript-api"}) as obs:
            record, ms = timed_fetch(self.transcripts, video_id, languages)
            ok = record.status == TranscriptStatus.OK
            obs.update(output={"status": record.status.value, "language": record.language_code,
                               "caption_type": record.caption_type, "segments": len(record.segments),
                               "from_cache": record.from_cache, "error": record.error},
                       **({} if ok else {"level": "WARNING", "status_message": record.error or record.status.value}))
            return record, ms

    @staticmethod
    def _to_recommendation(rank: int, video: ScoredVideo, request: RecommendationRequest,
                           learner: LearnerSignals) -> Recommendation:
        c, t = video.candidate, video.transcript
        ok = t is not None and t.status == TranscriptStatus.OK
        return Recommendation(
            rank=rank, video_id=c.video_id, title=c.title, description=c.description[:1000],
            thumbnail_url=c.thumbnail_url, channel_title=c.channel_title, duration_s=c.duration_s,
            published_at=c.published_at, url=c.url, embeddable=c.embeddable, score=video.score,
            score_breakdown=dict(video.features), evidence_level=video.evidence_level,
            evidence_coverage=video.coverage, matched_objective=request.objective,
            explanation=explain(video, learner),
            transcript_status=t.status if t else TranscriptStatus.NOT_ATTEMPTED,
            caption_language=t.language_code if ok else None, caption_type=t.caption_type if ok else None,
            transcript_evidence=video.evidence,
        )


def _filtered(video: ScoredVideo, reason: str) -> FilteredVideo:
    return FilteredVideo(video_id=video.candidate.video_id, title=video.candidate.title, reason=reason)
