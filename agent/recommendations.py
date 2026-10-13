"""Video-recommendation decisions and run lifecycle (decision layer; the pipeline does the work).

Policy selection: before a run, the context vector {rtt_ms, cpu, mem, competency_num, last_qoe} is
measured exactly as collect_context does, and agent.policy_selection.choose_policy picks one of the
predefined strategies R1–R3 from past runs that already have a reward (cold start → REC_DEFAULT_STRATEGY_ID).
Observed outcomes: learner events (displayed/opened/accepted/dismissed/rated/completed) and the
topic's assessments around the run. Reward: computed from those outcomes after the fact
(services.youtube.metrics.reward_stub, a placeholder until Person B's reward function exists).
"""
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy.orm import Session

from agent.deps import Services, get_services
from agent.nodes.context import measure_context
from agent.policy_selection import choose_policy
from app.db.repositories import AdaptationRepository, RecommendationRepository
from app.services.mcq import normalize_answer
from config import Settings, settings
from services.vector_store import HashingEmbedder
from services.youtube.cache import JsonFileCache
from services.youtube.data_api import QuotaTracker, YouTubeDataClient
from services.youtube.metrics import count_explicit_feedback, qoe_signals, recommendation_metrics, reward_stub
from services.youtube.pipeline import (
    PipelineSettings,
    RecommendationPipeline,
    RecommendationRequest,
    RecommendationStrategy,
)
from services.youtube.ranking import LANGUAGE_CODES
from services.youtube.relevance import RelevanceScorer
from services.youtube.transcripts import (
    CachedTranscriptFetcher,
    DisabledTranscriptFetcher,
    TranscriptFetcher,
    YouTubeTranscriptApiAdapter,
)

# Fixed, predefined strategies; every field is an action the pipeline can execute.
STRATEGIES: dict[str, RecommendationStrategy] = {
    "R1": RecommendationStrategy(
        strategy_id="R1", name="metadata-fast", max_candidates=10, transcript_budget=0, metadata_only=True,
        description="Title/description only; no transcript requests. Lowest latency and load."),
    "R2": RecommendationStrategy(
        strategy_id="R2", name="balanced", max_candidates=15, transcript_budget=4,
        description="Transcripts for the 4 best metadata matches."),
    "R3": RecommendationStrategy(
        strategy_id="R3", name="transcript-deep", max_candidates=25, transcript_budget=8,
        weights={"topic_similarity": 0.20, "transcript_relevance": 0.40, "difficulty_match": 0.10,
                 "language_match": 0.10, "duration_match": 0.06, "caption_match": 0.04,
                 "historical_outcome": 0.06, "popularity": 0.04},
        min_transcript_relevance=0.30,
        description="More candidates, 8 transcripts, transcript evidence weighted highest, stricter relevance."),
}
RATING_RANGE = (1, 5)


class RecommendationCoordinator:
    """Chooses a strategy, runs the pipeline, stores the run, and turns learner events into outcomes."""

    def __init__(self, *, pipeline: RecommendationPipeline, session_factory: Callable[[], Session],
                 services: Services, rng: random.Random, cfg: Settings = settings) -> None:
        self.pipeline = pipeline
        self.session_factory = session_factory
        self.services = services
        self.rng = rng
        self.cfg = cfg

    @contextmanager
    def _db(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ---- learner needs ----------------------------------------------------------------------------

    def learner_needs(self, student_id: str, topic_id: str) -> dict[str, Any]:
        """Language, competency and knowledge-gap concepts (MCQs answered wrongly in the latest test)."""
        profile = self.services.app.get_learner_profile(student_id)  # raises LearnerNotFound
        language = (profile.get("language") or "").lower()
        language_code = LANGUAGE_CODES.get(language, language if len(language) == 2 else None)
        with self._db() as db:
            repo = AdaptationRepository(db)
            tests = [t for t in (repo.get_latest_assessment(student_id, topic_id, "post"),
                                 repo.get_latest_assessment(student_id, topic_id, "pre")) if t]
        latest = max(tests, key=lambda t: t["created_at"]) if tests else None
        gaps: list[str] = []
        if latest:
            for i, item in enumerate(latest["items"]):
                answer = latest["answers"][i] if i < len(latest["answers"]) else None
                if normalize_answer(answer) != item.get("answer_index") and item.get("concept") not in gaps:
                    gaps.append(item.get("concept"))
        competency = latest["competency"] if latest else profile.get("competency_level")
        return {"language_code": language_code, "competency": competency,
                "competency_num": self.cfg.competency_levels.get(competency) if competency else None,
                "gap_concepts": [g for g in gaps if g], "assessment_type": latest["type"] if latest else None}

    # ---- policy selection + run -----------------------------------------------------------------------

    def choose_strategy(self, context: dict[str, Any]) -> tuple[RecommendationStrategy, str]:
        """kNN over past runs with a reward; returns (strategy, decision reason)."""
        with self._db() as db:
            history = RecommendationRepository(db).get_experience_runs(self.cfg.experience_fetch_limit)
        decision = choose_policy(
            context, history, policy_ids=list(STRATEGIES), default_policy_id=self.cfg.rec_default_strategy_id,
            k=self.cfg.knn_k, min_experiences=self.cfg.min_experiences, epsilon=self.cfg.epsilon, rng=self.rng,
            rtt_max_ms=self.cfg.rtt_max_ms, default_last_qoe=self.cfg.default_last_qoe)
        return STRATEGIES[decision.policy_id], decision.reason

    def recommend(self, student_id: str, topic_id: str, *, objective: str | None = None,
                  concepts: list[str] | None = None, session_id: str | None = None, revision: bool = False,
                  strategy_id: str | None = None, count: int | None = None) -> dict[str, Any]:
        """Run one recommendation request end to end and store it; returns the stored run with items."""
        needs = self.learner_needs(student_id, topic_id)
        context, context_errors = measure_context(self.services, student_id, needs["competency"] or "")
        if strategy_id is not None:
            if strategy_id not in STRATEGIES:
                raise ValueError(f"Unknown strategy {strategy_id!r}; choose from {sorted(STRATEGIES)}.")
            strategy, reason = STRATEGIES[strategy_id], f"{strategy_id}: requested explicitly (no selection)"
        else:
            strategy, reason = self.choose_strategy(context)

        concepts = needs["gap_concepts"] if concepts is None else [c.strip() for c in concepts if c.strip()]
        objective = (objective or "").strip() or (f"{topic_id}: {', '.join(concepts[:4])}" if concepts else topic_id)
        with self._db() as db:
            excluded = {} if revision else RecommendationRepository(db).learner_video_states(student_id)
        request = RecommendationRequest(
            topic=topic_id, objective=objective, concepts=concepts[:8], language_code=needs["language_code"],
            competency=needs["competency"], competency_num=needs["competency_num"], excluded=excluded,
            count=count or self.cfg.rec_count)

        def history_lookup(video_ids: list[str]) -> dict[str, dict[str, Any]]:
            with self._db() as db:
                return RecommendationRepository(db).video_history(video_ids)

        result = self.pipeline.run(request, strategy, history_lookup)
        metrics = {**result.metrics.model_dump(mode="json"), "rates": result.metrics.rates()}
        with self._db() as db:
            repo = RecommendationRepository(db)
            run = repo.create_run(
                student_id=student_id, session_id=session_id, topic_id=topic_id, objective=objective,
                query=result.query, strategy_id=strategy.strategy_id, decision_reason=reason, context=context,
                strategy=strategy.model_dump(mode="json"),
                request={**request.model_dump(mode="json", exclude={"history"}), "revision": revision,
                         "needs": needs, "filtered": [f.model_dump() for f in result.filtered]},
                status=result.status, errors=context_errors + result.errors, metrics=metrics,
                items=[r.model_dump(mode="json") for r in result.recommendations])
            for item in run["items"]:
                repo.add_event(run_id=run["id"], event_type="generated", item_id=item["id"])
        return self.get_run(run["id"])

    # ---- outcomes ---------------------------------------------------------------------------------

    def record_event(self, run_id: str, event_type: str, *, item_id: str | None = None,
                     value: dict[str, Any] | None = None) -> dict[str, Any]:
        """Store a learner event and recompute the run's outcome metrics, QoE signals and reward."""
        if event_type == "rated":
            rating = (value or {}).get("rating")
            if not isinstance(rating, int) or isinstance(rating, bool) or not RATING_RANGE[0] <= rating <= RATING_RANGE[1]:
                raise ValueError("a 'rated' event needs value {'rating': 1-5}")
        if event_type in {"opened", "accepted", "dismissed", "rated", "completed", "displayed"} and item_id is None:
            raise ValueError(f"a '{event_type}' event needs item_id")
        with self._db() as db:
            event = RecommendationRepository(db).add_event(run_id=run_id, event_type=event_type, item_id=item_id,
                                                           value=value)
        return {"event": event, "outcome": self.evaluate(run_id)}

    def evaluate(self, run_id: str) -> dict[str, Any]:
        """Recompute metrics from all events and the assessments around the run; store and return them."""
        with self._db() as db:
            repo = RecommendationRepository(db)
            run = repo.get_run(run_id)
            if run is None:
                raise LookupError(f"Recommendation run {run_id!r} does not exist.")
            events = repo.list_events(run_id)
            pre, post = repo.assessments_around(run["student_id"], run["topic_id"], run["created_at"])
            metrics = recommendation_metrics(run["items"], events, pre, post, k=self.cfg.rec_count)
            signals = qoe_signals(metrics, run["metrics"], latency_budget_ms=self.cfg.rec_latency_budget_ms,
                                  rtt_ms=(run["context"] or {}).get("rtt_ms"), rtt_max_ms=self.cfg.rtt_max_ms)
            reward = reward_stub(signals, count_explicit_feedback(events))
            repo.update_outcome(run_id, outcome=metrics, qoe_signals=signals, reward=reward)
        return {"metrics": metrics, "qoe_signals": signals, "reward": reward, "reward_is_placeholder": True}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Stored run with items; each recommendation payload carries its item_id."""
        with self._db() as db:
            run = RecommendationRepository(db).get_run(run_id)
        if run is None:
            return None
        run["recommendations"] = [{**item["payload"], "item_id": item["id"]} for item in run["items"]]
        return run

    def list_runs(self, student_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._db() as db:
            return RecommendationRepository(db).list_runs(student_id=student_id, limit=limit)


def make_transcript_fetcher(cfg: Settings, cache: JsonFileCache | None) -> TranscriptFetcher:
    """The youtube-transcript-api adapter (cached), or a disabled fetcher with the reason."""
    if not cfg.transcripts_enabled:
        return DisabledTranscriptFetcher()
    try:
        adapter = YouTubeTranscriptApiAdapter()
    except ImportError:
        return DisabledTranscriptFetcher("youtube-transcript-api is not installed")
    return CachedTranscriptFetcher(adapter, cache)


def make_recommender(services: Services | None = None, *, data_client: YouTubeDataClient | None = None,
                     transcripts: TranscriptFetcher | None = None, embedder: Any = None,
                     cfg: Settings = settings) -> RecommendationCoordinator:
    """Wire the coordinator to the shared services: the RAG embedder, the monitor and the app DB."""
    services = services or get_services()
    cache = JsonFileCache(cfg.youtube_cache_dir, {"search": cfg.youtube_search_cache_ttl_s,
                                                  "videos": cfg.youtube_video_cache_ttl_s,
                                                  "transcripts": cfg.transcript_cache_ttl_s})
    if data_client is None:
        data_client = YouTubeDataClient(
            api_key=cfg.youtube_api_key.get_secret_value() if cfg.youtube_api_key else None,
            base_url=cfg.youtube_api_base_url, timeout_s=cfg.youtube_timeout_s, cache=cache,
            quota=QuotaTracker(cfg.youtube_cache_dir / "quota.json", cfg.youtube_daily_quota_units,
                               cfg.youtube_quota_reserve_units))
    # Reuse the RAG embedder (no second model); fakes without one fall back to the hashing embedder.
    embedder = embedder or getattr(services.crag.store, "embedder", None) or HashingEmbedder()
    scorer = RelevanceScorer(embedder, floor=cfg.rec_similarity_floor, ceil=cfg.rec_similarity_ceil,
                             top_chunks=cfg.rec_top_chunks, concept_threshold=cfg.rec_concept_cover_threshold)
    pipeline = RecommendationPipeline(
        data_client=data_client, transcripts=transcripts or make_transcript_fetcher(cfg, cache), scorer=scorer,
        monitor=services.monitor,
        settings=PipelineSettings(
            region_code=cfg.youtube_region_code, safe_search=cfg.youtube_safe_search,
            max_pages=cfg.youtube_search_max_pages, transcript_languages=cfg.transcript_languages,
            transcripts_enabled=cfg.transcripts_enabled, transcript_max_words=cfg.transcript_max_words,
            chunk_words=cfg.transcript_chunk_words, chunk_overlap=cfg.transcript_chunk_overlap_words,
            max_chunks_per_video=cfg.transcript_max_chunks_per_video, min_duration_s=cfg.rec_min_duration_s,
            duration_min_s=cfg.rec_preferred_duration_min_s, duration_max_s=cfg.rec_preferred_duration_max_s,
            require_embeddable=cfg.rec_require_embeddable, require_language_match=cfg.rec_require_language_match,
            prior=cfg.rec_missing_feature_prior, default_weights=cfg.rec_weights,
            transcript_workers=cfg.transcript_max_workers, transcript_time_budget_s=cfg.transcript_time_budget_s))
    return RecommendationCoordinator(pipeline=pipeline, session_factory=services.app.session_factory,
                                     services=services, rng=services.rng, cfg=cfg)
