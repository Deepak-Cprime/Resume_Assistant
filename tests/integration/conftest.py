"""One real-parser, fake-embedder run over every synthetic resume, shared by integration tests."""

import os

import pytest

from src.config import load_settings
from src.pipeline import report, uploads
from src.pipeline import run as pipeline
from src.stores import db
from tests.conftest import FIXTURE_AS_OF
from tests.unit.fakes import FakeEmbedder
from tests.unit.pipeline_helpers import configure


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, fixture_files, specs):
    os.environ["RAG_DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    settings = load_settings()
    configure(settings)
    settings.metadata.as_of = FIXTURE_AS_OF
    conn = db.connect(settings.paths.db)
    uploads.ingest_files(conn, settings, [(p.name, p.read_bytes()) for p in fixture_files.values()])
    result = pipeline.run_pipeline(conn, settings, embedders=[FakeEmbedder("fake-embed")])
    expected = {k: v["expected"]["total_yoe"] for k, v in specs.items()}
    reports = report.document_reports(conn, settings, expected_yoe=expected)
    yield conn, settings, result, reports
    conn.close()


