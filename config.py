"""Project-wide settings: paths, thresholds, limits and model endpoints (validated by pydantic).

Values can be overridden with environment variables or a .env file at the project root,
e.g. CRAG_RETRIEVAL_THRESHOLD=0.7 or HF_TOKEN=hf_xxx.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """All tunable values in one validated object."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- Paths (relative paths resolve against the project root) -------------
    data_dir: Path = Path("data")
    # LangGraph checkpoints live in their own SQLite file, separate from the app DB.
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    vector_index_dir: Path = Path("data/vector_indexes")
    knowledge_samples_dir: Path = Path("knowledge_prep/samples")
    docs_dir: Path = Path("docs")

    # --- CRAG ----------------------------------------------------------------
    # Calibrated for nomic-embed-text on the photosynthesis sample: on-topic questions scored
    # 0.66–0.84 and off-topic ones 0.41–0.55. Re-calibrate per embedding model in experiments/.
    # Minimum top retrieval score for CRAG to accept the retrieved chunks.
    crag_retrieval_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    # Chunks scoring below this are discarded as irrelevant (the CRAG "incorrect" grade).
    crag_min_chunk_score: float = Field(default=0.50, ge=0.0, le=1.0)

    # --- Knowledge preparation -----------------------------------------------
    chunk_size_words: int = Field(default=120, gt=0)
    chunk_overlap_words: int = Field(default=25, ge=0)

    # --- Embeddings ------------------------------------------------------------
    # "ollama" uses a local embedding model; "hashing" is a no-model offline fallback for tests.
    embedding_backend: Literal["ollama", "hashing"] = "ollama"
    embedding_model: str = "nomic-embed-text"

    # --- LLMs ----------------------------------------------------------------
    # 127.0.0.1 rather than localhost: on Windows, "localhost" tries IPv6 first and adds ~2 s per call.
    ollama_base_url: str = "http://127.0.0.1:11434"
    local_llm_model: str = "gemma3:1b"
    local_llm_timeout_s: float = Field(default=180.0, gt=0)
    hf_token: SecretStr | None = None
    hf_inference_url: str = "https://router.huggingface.co/v1/chat/completions"
    cloud_llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    cloud_llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_tokens: int = Field(default=384, gt=0)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # Provider health checks are cached for this long, so each question doesn't re-probe.
    llm_health_ttl_s: float = Field(default=30.0, ge=0)
    llm_health_timeout_s: float = Field(default=2.0, gt=0)

    # --- Session -------------------------------------------------------------
    # Hard cap on questions per session, even if the learner asks to continue.
    max_questions_per_session: int = Field(default=10, gt=0)
    pretest_num_items: int = Field(default=5, gt=0)
    posttest_num_items: int = Field(default=5, gt=0)

    # --- Competency ----------------------------------------------------------
    # Numeric encoding of competency levels used in the policy-selection context.
    competency_levels: dict[str, float] = {"beginner": 0.0, "intermediate": 0.5, "advanced": 1.0}
    # Minimum pre-test score (0..1) for each level, checked from highest to lowest.
    competency_bands: list[tuple[str, float]] = [("advanced", 0.8), ("intermediate", 0.5), ("beginner", 0.0)]

    # --- Policies ------------------------------------------------------------
    default_policy_id: str = "P1"

    # --- Context collection ----------------------------------------------------
    # RTT is probed against the cloud LLM host; any HTTP response counts as a round trip.
    probe_url: str = "https://router.huggingface.co"
    probe_timeout_s: float = Field(default=1.5, gt=0)
    rtt_max_ms: float = Field(default=2000.0, gt=0)  # RTT above this normalises to 1.0
    rtt_fail_ms: float = Field(default=5000.0, gt=0)  # RTT recorded when the probe fails
    default_last_qoe: float = Field(default=0.5, ge=0.0, le=1.0)  # learner with no past session

    # --- Policy selection (case-based, not RL) ---------------------------------
    knn_k: int = Field(default=5, gt=0)
    min_experiences: int = Field(default=3, ge=0)  # fewer stored experiences than this → default policy
    epsilon: float = Field(default=0.2, ge=0.0, le=1.0)  # chance of trying a non-greedy policy
    policy_rng_seed: int | None = None  # set for reproducible experiments
    experience_fetch_limit: int = Field(default=1000, gt=0)  # newest experiences considered by kNN

    # --- YouTube recommendations: Data API v3 (search.list + videos.list only) ---------------
    youtube_api_key: SecretStr | None = None  # env YOUTUBE_API_KEY; never commit it
    youtube_api_base_url: str = "https://www.googleapis.com/youtube/v3"
    youtube_timeout_s: float = Field(default=10.0, gt=0)
    youtube_region_code: str | None = "IN"  # ISO 3166-1 alpha-2, biases search.list results
    youtube_safe_search: Literal["none", "moderate", "strict"] = "strict"
    youtube_search_max_pages: int = Field(default=1, ge=1, le=3)  # each page costs 100 quota units
    youtube_daily_quota_units: int = Field(default=10_000, gt=0)  # the project's default daily quota
    youtube_quota_reserve_units: int = Field(default=500, ge=0)  # stop calling the API below this estimate
    youtube_cache_dir: Path = Path("data/youtube_cache")
    # YouTube API Services policies require refreshing or deleting stored API data within 30 days.
    youtube_search_cache_ttl_s: float = Field(default=6 * 3600, ge=0, le=30 * 86400)
    youtube_video_cache_ttl_s: float = Field(default=24 * 3600, ge=0, le=30 * 86400)

    # --- Transcripts (youtube-transcript-api, unofficial; optional) ----------------------------
    transcripts_enabled: bool = True
    transcript_languages: list[str] = ["en", "hi"]  # preference order; manual tracks before auto
    transcript_cache_ttl_s: float = Field(default=24 * 3600, ge=0, le=30 * 86400)  # 0 = never stored
    transcript_max_words: int = Field(default=12_000, gt=0)  # longer transcripts are truncated
    transcript_chunk_words: int = Field(default=120, gt=10)
    transcript_chunk_overlap_words: int = Field(default=20, ge=0)
    transcript_max_chunks_per_video: int = Field(default=40, gt=0)
    transcript_max_workers: int = Field(default=4, ge=1, le=8)  # parallel fetches (the library has no timeout)
    transcript_time_budget_s: float = Field(default=20.0, gt=0)  # unfinished fetches are dropped after this

    # --- Recommendation ranking ------------------------------------------------------------------
    rec_count: int = Field(default=5, gt=0, le=20)  # recommendations returned
    # Cosine similarity → [0, 1]: values ≤ floor map to 0, ≥ ceil to 1 (calibrated for nomic-embed-text).
    rec_similarity_floor: float = Field(default=0.40, ge=-1.0, le=1.0)
    rec_similarity_ceil: float = Field(default=0.85, ge=-1.0, le=1.0)
    rec_top_chunks: int = Field(default=3, gt=0)  # transcript relevance = mean of the best chunks
    rec_concept_cover_threshold: float = Field(default=0.6, ge=0.0, le=1.0)  # normalised chunk similarity
    rec_min_duration_s: int = Field(default=60, ge=0)  # hard filter: shorter videos are excluded
    rec_preferred_duration_min_s: int = Field(default=240, ge=0)
    rec_preferred_duration_max_s: int = Field(default=1200, gt=0)
    rec_require_embeddable: bool = False
    rec_require_language_match: bool = False
    rec_missing_feature_prior: float = Field(default=0.5, ge=0.0, le=1.0)  # neutral value for missing evidence
    rec_weights: dict[str, float] = {
        "topic_similarity": 0.25, "transcript_relevance": 0.30, "difficulty_match": 0.10,
        "language_match": 0.10, "duration_match": 0.08, "caption_match": 0.05,
        "historical_outcome": 0.07, "popularity": 0.05,
    }
    rec_default_strategy_id: str = "R2"  # baseline when there is too little history for kNN
    rec_latency_budget_ms: float = Field(default=15_000.0, gt=0)  # latency at which latency quality hits 0

    # --- QoE / SRE / utility / reward (metrics/) --------------------------------------------------
    # Each group's weights must sum to 1 (validated). Keys: see metrics/weights.py GROUP_KEYS.
    metric_weights: dict[str, dict[str, float]] = {
        "qoe": {"EQ": 0.35, "AQ": 0.25, "NQ": 0.20, "UX": 0.20},
        "eq": {"LG": 0.35, "PT": 0.30, "RA": 0.15, "LS": 0.20},
        "aq": {"RQ": 0.40, "LQ": 0.30, "SQ": 0.15, "TQ": 0.15},
        "nq": {"PL": 0.35, "NR": 0.25, "RU": 0.15, "AV": 0.25},
        "ux": {"SR": 0.45, "FS": 0.20, "CR": 0.35},
        "sre": {"CPU": 0.35, "MEM": 0.30, "EN": 0.20, "NE": 0.15},
        "utility": {"QOE": 0.70, "SRE": 0.30},  # lambda1, lambda2
        "reward": {"QOE": 0.70, "SRE": 0.30},  # mu1, mu2
    }
    # "fixed": the weights above. "ewm": the active entropy-weight set per group (fixed where none is active).
    metric_weights_source: Literal["fixed", "ewm"] = "fixed"
    # Missing metrics: "renormalize" over the available ones (coverage reported), "neutral" (impute 0.5),
    # or "strict" (raise). Applied the same way inside groups and across groups.
    metric_missing_policy: Literal["renormalize", "neutral", "strict"] = "renormalize"
    qoe_latency_good_ms: float = Field(default=1500.0, gt=0)  # mean answer latency at or below → PL = 1
    qoe_latency_bad_ms: float = Field(default=15000.0, gt=0)  # at or above → PL = 0
    qoe_retrieval_floor: float = Field(default=0.40, ge=0.0, le=1.0)  # retrieval score → RQ = 0
    qoe_retrieval_ceil: float = Field(default=0.85, ge=0.0, le=1.0)  # retrieval score → RQ = 1
    qoe_utilization_target: float = Field(default=0.70, gt=0.0, le=1.0)  # load up to here → RU = 1
    qoe_utilization_saturation: float = Field(default=0.95, gt=0.0, le=1.0)  # load from here → RU = 0
    sentiment_lexicon_enabled: bool = True  # FS from a small word lexicon (labelled estimate)
    sre_cpu_max: float = Field(default=1.0, gt=0.0, le=1.0)  # CPU budget as a utilisation fraction
    sre_memory_max: float = Field(default=1.0, gt=0.0, le=1.0)
    sre_energy_max_wh: float = Field(default=2.0, gt=0)  # per-session energy budget
    sre_bandwidth_max_bytes: float = Field(default=50e6, gt=0)  # per-session network budget
    energy_estimate_enabled: bool = True  # EN from a linear CPU power model (labelled estimate)
    energy_idle_power_w: float = Field(default=8.0, ge=0)
    energy_max_power_w: float = Field(default=45.0, gt=0)
    baseline_qoe: float = Field(default=0.5, ge=0.0, le=1.0)  # QoE_old on a learner's first session
    baseline_sre: float = Field(default=0.5, ge=0.0, le=1.0)  # SRE_old on a learner's first session
    ewm_min_observations: int = Field(default=10, ge=2)

    # --- Adaptive policy selection (FAISS) ---------------------------------------------------------
    experience_index_dir: Path = Path("data/experience_index")
    policy_kernel_sigma: float = Field(default=0.25, gt=0)  # similarity weight exp(-d²/2σ²)
    policy_min_support: int = Field(default=1, ge=1)  # neighbours needed before an action has an estimate
    # Total similarity weight needed for an estimate: one neighbour at distance d has weight exp(−d²/2σ²);
    # 0.1 ≈ a single neighbour within 2.1σ. Stops very dissimilar states from producing confident estimates.
    policy_min_weight_sum: float = Field(default=0.1, ge=0.0)
    policy_estimate_target: Literal["utility", "reward"] = "utility"

    # --- Optional Langfuse tracing (telemetry/tracing.py; no-op unless both keys are set) ---------------
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str | None = None  # e.g. https://cloud.langfuse.com (EU) or https://us.cloud.langfuse.com
    langfuse_host: str | None = None  # older name for the same setting; LANGFUSE_BASE_URL wins
    langfuse_environment: str = "development"  # keeps test/dev traces apart from production data
    langfuse_release: str | None = None  # e.g. a git commit or version tag
    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    # False: learner questions, answers, comments and lesson text are replaced before export; scores,
    # models, token usage, latencies and the trace structure are still sent. PII redaction always applies.
    langfuse_capture_content: bool = True

    @field_validator("data_dir", "checkpoint_db_path", "vector_index_dir", "knowledge_samples_dir", "docs_dir",
                     "youtube_cache_dir", "experience_index_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Resolve relative paths against the project root."""
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("competency_bands")
    @classmethod
    def _bands_descending(cls, value: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Keep bands sorted from highest to lowest minimum score."""
        return sorted(value, key=lambda band: band[1], reverse=True)

    @model_validator(mode="after")
    def _check_recommendation_settings(self) -> "Settings":
        """Reject inconsistent recommendation settings at startup rather than mid-request."""
        if self.rec_similarity_ceil <= self.rec_similarity_floor:
            raise ValueError("REC_SIMILARITY_CEIL must be greater than REC_SIMILARITY_FLOOR")
        if self.rec_preferred_duration_max_s <= self.rec_preferred_duration_min_s:
            raise ValueError("REC_PREFERRED_DURATION_MAX_S must exceed REC_PREFERRED_DURATION_MIN_S")
        if self.transcript_chunk_overlap_words >= self.transcript_chunk_words:
            raise ValueError("TRANSCRIPT_CHUNK_OVERLAP_WORDS must be smaller than TRANSCRIPT_CHUNK_WORDS")
        from services.youtube.ranking import RankingWeights  # validates names, signs and the sum

        RankingWeights.from_mapping(self.rec_weights)
        from metrics.weights import WeightSet  # every group: exact keys, non-negative, sums to 1

        WeightSet.from_config(self.metric_weights)
        if self.qoe_latency_bad_ms <= self.qoe_latency_good_ms:
            raise ValueError("QOE_LATENCY_BAD_MS must exceed QOE_LATENCY_GOOD_MS")
        if self.qoe_retrieval_ceil <= self.qoe_retrieval_floor:
            raise ValueError("QOE_RETRIEVAL_CEIL must exceed QOE_RETRIEVAL_FLOOR")
        if self.qoe_utilization_saturation <= self.qoe_utilization_target:
            raise ValueError("QOE_UTILIZATION_SATURATION must exceed QOE_UTILIZATION_TARGET")
        if self.energy_max_power_w < self.energy_idle_power_w:
            raise ValueError("ENERGY_MAX_POWER_W must be at least ENERGY_IDLE_POWER_W")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()


settings = get_settings()
