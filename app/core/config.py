from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    knowledge_storage_path: Path = Field(default=Path("data/knowledge"))
    faiss_storage_path: Path = Field(default=Path("data/faiss"))
    hf_token: SecretStr | None = None
    llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_inference_url: str = "https://router.huggingface.co/v1/chat/completions"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_tokens: int = Field(default=512, gt=0)
    llm_temperature: float = Field(default=0.2, ge=0)
    default_runtime_configuration_name: str = "huggingface-qwen-text"
    youtube_api_key: SecretStr | None = None
    youtube_api_base_url: str = "https://www.googleapis.com/youtube/v3/search"
    youtube_default_language: str = "en"
    youtube_max_results: int = Field(default=5, gt=0, le=20)
    youtube_timeout_seconds: float = Field(default=20.0, gt=0)
    knowledge_chunk_size: int = Field(default=120, gt=0)
    knowledge_chunk_overlap: int = Field(default=25, ge=0)
    knowledge_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    knowledge_retrieval_top_k: int = Field(default=3, gt=0, le=10)
    local_llm_enabled: bool = Field(default=False)
    local_llm_model_id: str = "local-model"
    crag_quality_min_score: float = Field(default=0.75, ge=0.0, le=1.0)
    stt_provider_name: str = "passthrough"
    stt_provider_enabled: bool = Field(default=True)
    tts_provider_name: str = "passthrough"
    tts_provider_enabled: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
