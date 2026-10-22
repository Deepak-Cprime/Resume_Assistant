from datetime import date
from pathlib import Path

import pytest

from src.config import Settings, load_settings
from tests.fixtures.builder import load_specs, write_all

FIXTURE_AS_OF = date(2025, 6, 1)  # what "Present" means in the fixture ground truth


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


@pytest.fixture(scope="session")
def parsed(fixture_files) -> dict[str, str]:
    """Cleaned Markdown of every fixture, parsed once per session (OCR makes one slow)."""
    pytest.importorskip("pymupdf4llm")
    from src.providers.parsers import router

    settings = load_settings()
    out = {}
    for spec_id, path in fixture_files.items():
        doc_type = "docx" if Path(path).suffix == ".docx" else "pdf"
        out[spec_id] = router.parse_resume(path, doc_type, settings).markdown
    return out


@pytest.fixture(scope="session")
def profiles(parsed) -> dict:
    """Extracted CandidateProfile of every fixture (rules only, no LLM)."""
    from src.metadata.profile import build_profile

    return {k: build_profile(md, as_of=FIXTURE_AS_OF) for k, md in parsed.items()}
