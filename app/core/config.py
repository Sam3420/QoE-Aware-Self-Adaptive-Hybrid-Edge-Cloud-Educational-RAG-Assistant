from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
