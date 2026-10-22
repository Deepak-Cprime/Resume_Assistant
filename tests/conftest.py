import pytest

from src.config import Settings, load_settings


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    """Real config, but with data/ redirected to tmp_path so tests never touch ./data."""
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    return load_settings()
