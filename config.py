"""Project-wide settings: paths, thresholds, limits and model endpoints (validated by pydantic).

Values can be overridden with environment variables or a .env file at the project root,
e.g. CRAG_RETRIEVAL_THRESHOLD=0.7 or HF_TOKEN=hf_xxx.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
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

    @field_validator("data_dir", "checkpoint_db_path", "vector_index_dir", "knowledge_samples_dir", "docs_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Resolve relative paths against the project root."""
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("competency_bands")
    @classmethod
    def _bands_descending(cls, value: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Keep bands sorted from highest to lowest minimum score."""
        return sorted(value, key=lambda band: band[1], reverse=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()


settings = get_settings()
