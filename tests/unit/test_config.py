import pytest
import yaml
from pydantic import ValidationError

from src.config import PROJECT_ROOT, load_settings


def test_default_config_loads():
    s = load_settings()
    assert s.use_case == "resume"
    assert s.embedding.default == "BAAI/bge-small-en-v1.5"
    assert [m.name for m in s.embedding.models] == ["BAAI/bge-small-en-v1.5", "BAAI/bge-m3"]
    assert s.chunking.budgets.prefix_reserve == 32
    assert s.chunking.budgets.max_clamp == (256, None)
    assert (s.metadata.use_llm, s.metadata.as_of, s.metadata.gap_months) == (False, None, 3)
    assert s.parser.fallback is None and s.parser.docx == "python_docx"


def test_relative_paths_anchor_at_project_root():
    s = load_settings()
    assert s.paths.data_dir == PROJECT_ROOT / "data"
    assert s.paths.db == PROJECT_ROOT / "data" / "app.db"


def test_data_dir_env_override(settings, tmp_path):
    assert settings.paths.data_dir == tmp_path / "data"
    assert settings.paths.db == tmp_path / "data" / "app.db"
    assert settings.raw_dir == tmp_path / "data" / "raw"
    assert settings.indexes_dir == tmp_path / "data" / "indexes"


def _write(tmp_path, mutate):
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    mutate(raw)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_default_must_be_a_listed_model(tmp_path):
    p = _write(tmp_path, lambda r: r["embedding"].update(default="nope"))
    with pytest.raises(ValidationError):
        load_settings(p)


def test_unknown_key_rejected(tmp_path):
    p = _write(tmp_path, lambda r: r.update(typo_key=1))
    with pytest.raises(ValidationError):
        load_settings(p)


def test_percentiles_must_be_ordered(tmp_path):
    p = _write(tmp_path, lambda r: r["chunking"]["budgets"].update(p_low=95))
    with pytest.raises(ValidationError):
        load_settings(p)


def test_unknown_embedding_model_lookup(settings):
    with pytest.raises(KeyError):
        settings.embedding.get("missing")
