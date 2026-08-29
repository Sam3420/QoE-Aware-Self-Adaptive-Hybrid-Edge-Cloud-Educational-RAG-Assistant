from pathlib import Path

from app.core.config import Settings


def test_resolves_local_edge_storage_paths_from_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    settings = Settings(
        data_root=Path("edge-data"),
        database_path=Path("edge-data/app.db"),
        knowledge_storage_path=Path("edge-data/knowledge"),
        vector_index_storage_path=Path("edge-data/vector-indexes"),
        metrics_storage_path=Path("edge-data/metrics"),
        experience_storage_path=Path("edge-data/experiences"),
    )

    assert settings.resolved_database_url == f"sqlite:///{(tmp_path / 'edge-data/app.db').as_posix()}"
    assert settings.resolved_storage_paths()["data_root"] == tmp_path / "edge-data"
    assert settings.resolved_storage_paths()["vector_indexes"] == tmp_path / "edge-data/vector-indexes"


def test_database_url_overrides_database_path(tmp_path):
    settings = Settings(
        database_url="sqlite:///:memory:",
        database_path=tmp_path / "ignored.db",
    )

    assert settings.resolved_database_url == "sqlite:///:memory:"
