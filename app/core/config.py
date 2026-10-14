from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "QoE Educational Assistant"
    app_env: str = "local"
    log_level: str = "INFO"

    data_root: Path = Field(default=Path("data"))
    database_path: Path | None = Field(default=Path("data/app.db"))
    database_url: str | None = None
    knowledge_storage_path: Path = Field(default=Path("data/knowledge"))
    vector_index_storage_path: Path = Field(default=Path("data/vector_indexes"))
    metrics_storage_path: Path = Field(default=Path("data/metrics"))
    experience_storage_path: Path = Field(default=Path("data/experiences"))
    hf_token: SecretStr | None = None
    llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_inference_url: str = "https://router.huggingface.co/v1/chat/completions"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_tokens: int = Field(default=512, gt=0)
    llm_temperature: float = Field(default=0.2, ge=0)
    default_runtime_configuration_name: str = "huggingface-qwen-text"
    competency_beginner_max: float = Field(default=0.4, ge=0, le=1)
    competency_intermediate_max: float = Field(default=0.75, ge=0, le=1)
    mcq_n_items: int = Field(default=5, gt=0)
    # A returning learner's pre-test re-asks at most this many previously missed questions.
    mcq_retest_max: int = Field(default=2, ge=0)
    # The topic bank grows with new questions when a returning learner has seen them all; it stops here.
    mcq_bank_max_items: int = Field(default=40, gt=0)
    mcq_max_retries: int = Field(default=1, ge=0)
    mcq_context_chunks: int = Field(default=5, gt=0)
    mcq_max_tokens: int = Field(default=1024, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @computed_field
    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        if self.database_path is None:
            raise ValueError("DATABASE_PATH is required when DATABASE_URL is not set.")
        return f"sqlite:///{self._resolve_path(self.database_path).as_posix()}"

    def resolved_data_root(self) -> Path:
        return self._resolve_path(self.data_root)

    def resolved_storage_paths(self) -> dict[str, Path]:
        return {
            "data_root": self.resolved_data_root(),
            "knowledge": self._resolve_path(self.knowledge_storage_path),
            "vector_indexes": self._resolve_path(self.vector_index_storage_path),
            "metrics": self._resolve_path(self.metrics_storage_path),
            "experiences": self._resolve_path(self.experience_storage_path),
        }

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        return path if path.is_absolute() else Path.cwd() / path


# Predefined runtime policies (top_k x max_retries x route) used to seed the policies table.
# FLAG FOR SAMI: agent/policies.py should be the single source of truth. Once it is committed,
# seed from it and delete this copy. The P1-P4 mapping and threshold below are assumptions.
POLICY_SEED: tuple[dict, ...] = (
    {
        "policy_id": "P1",
        "name": "local-light",
        "top_k": 3,
        "threshold": 0.75,
        "max_retries": 0,
        "route": "local",
        "is_default": True,
        "description": "Local SLM, 3 chunks, no corrective retry.",
    },
    {
        "policy_id": "P2",
        "name": "local-thorough",
        "top_k": 5,
        "threshold": 0.75,
        "max_retries": 1,
        "route": "local",
        "is_default": False,
        "description": "Local SLM, 5 chunks, one corrective retry.",
    },
    {
        "policy_id": "P3",
        "name": "cloud-light",
        "top_k": 3,
        "threshold": 0.75,
        "max_retries": 0,
        "route": "cloud",
        "is_default": False,
        "description": "Cloud LLM, 3 chunks, no corrective retry.",
    },
    {
        "policy_id": "P4",
        "name": "cloud-thorough",
        "top_k": 5,
        "threshold": 0.75,
        "max_retries": 1,
        "route": "cloud",
        "is_default": False,
        "description": "Cloud LLM, 5 chunks, one corrective retry.",
    },
)


@lru_cache
def get_settings() -> Settings:
    return Settings()
