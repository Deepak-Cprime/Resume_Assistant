import pytest

from src.config import Settings, load_settings
from tests.fixtures.builder import load_specs, write_all


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    """Real config, but with data/ redirected to tmp_path so tests never touch ./data."""
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    return load_settings()


@pytest.fixture(scope="session")
def specs() -> dict:
    """Fixture specs (content + ground truth) keyed by id."""
    return load_specs()


@pytest.fixture(scope="session")
def fixture_files(tmp_path_factory) -> dict:
    """Synthetic resume files generated once per test session, keyed by spec id."""
    return write_all(tmp_path_factory.mktemp("fixtures"))
