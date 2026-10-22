"""Setup for the Streamlit smoke tests: a throwaway config and data folder, and a fake embedder
so no model is downloaded. The pages run for real (Streamlit's AppTest); only the embedding
model is swapped."""

from types import SimpleNamespace

import pytest
import yaml

from src.config import PROJECT_ROOT, load_settings
from src.pipeline import run as pipeline
from src.pipeline import uploads
from src.providers import factory
from src.stores import db
from tests.unit.fakes import FakeEmbedder

APP_DIR = PROJECT_ROOT / "app"


@pytest.fixture
def env(tmp_path, monkeypatch):
    st = pytest.importorskip("streamlit")
    pytest.importorskip("streamlit.testing.v1")

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    cfg["vector_store"]["provider"] = "numpy"
    cfg["embedding"]["models"] = [{"provider": "sentence_transformers", "name": "fake-embed"}]
    cfg["embedding"]["default"] = "fake-embed"
    cfg["metadata"]["as_of"] = "2025-06-01"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("RAG_CONFIG", str(config_path))
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(factory, "create_embedder", lambda name, settings: FakeEmbedder(name))

    st.cache_resource.clear()  # the pages cache the connection and embedder per data folder
    settings = load_settings()
    conn = db.connect(settings.paths.db)
    yield SimpleNamespace(settings=settings, conn=conn)
    conn.close()
    st.cache_resource.clear()


@pytest.fixture
def open_page(env):
    from streamlit.testing.v1 import AppTest

    def open_(relative: str, **kwargs):
        return AppTest.from_file(str(APP_DIR / relative), default_timeout=60).run(**kwargs)

    return open_


@pytest.fixture
def add_resumes(env, fixture_files):
    """Upload fixture files the way the Upload page does, then optionally run the pipeline."""

    def add(*specs: str, process: bool = True, build_indexes: bool = True) -> list[str]:
        results = uploads.ingest_files(
            env.conn, env.settings,
            [(fixture_files[s].name, fixture_files[s].read_bytes()) for s in specs],
        )
        if process:
            pipeline.run_pipeline(
                env.conn, env.settings, embedders=[FakeEmbedder("fake-embed")],
                build_indexes=build_indexes,
            )
        return [r.doc_id for r in results]

    return add
