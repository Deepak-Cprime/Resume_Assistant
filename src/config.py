"""Typed settings loaded from configs/config.yaml, with environment overrides.

Environment variables:
    RAG_CONFIG    path to an alternative config file
    RAG_DATA_DIR  overrides paths.data_dir (and moves paths.db under it)
Secrets (paid API keys) are read from the environment by the provider that needs them,
never from the YAML file.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    data_dir: Path = Path("data")
    db: Path = Path("data/app.db")


class IngestionConfig(_Section):
    max_file_mb: int = Field(10, gt=0)
    max_pages: int = Field(10, gt=0)


class VlmFallbackConfig(_Section):
    enabled: bool = False
    model: str = "qwen2.5vl:7b"


class ParserConfig(_Section):
    default: str = "pymupdf4llm"
    fallback: str = "docling"
    ocr: str = "tesseract"
    vlm_fallback: VlmFallbackConfig = VlmFallbackConfig()
    min_chars_per_page: int = Field(50, ge=0)


class LLMConfig(_Section):
    provider: str = "ollama"
    model: str = "qwen2.5:7b"
    temperature: float = 0


class EmbeddingModelConfig(_Section):
    provider: str
    name: str


class EmbeddingConfig(_Section):
    models: list[EmbeddingModelConfig]
    default: str

    @model_validator(mode="after")
    def _default_is_listed(self) -> EmbeddingConfig:
        if self.default not in {m.name for m in self.models}:
            raise ValueError(f"embedding.default '{self.default}' is not in embedding.models")
        return self

    def get(self, name: str) -> EmbeddingModelConfig:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"unknown embedding model '{name}'")


class BudgetsConfig(_Section):
    mode: str = "adaptive_per_section"
    p_low: int = Field(20, ge=0, le=100)
    p_high: int = Field(90, ge=0, le=100)
    min_clamp: tuple[int, int] = (64, 128)
    max_clamp: tuple[int, int | None] = (256, None)
    prefix_reserve: int = Field(32, ge=0)

    @model_validator(mode="after")
    def _percentiles_ordered(self) -> BudgetsConfig:
        if self.p_low >= self.p_high:
            raise ValueError("budgets.p_low must be smaller than budgets.p_high")
        return self


class ChunkingConfig(_Section):
    strategy: str = "section_entry"
    budgets: BudgetsConfig = BudgetsConfig()


class VectorStoreConfig(_Section):
    provider: str = "faiss"
    index: str = "flat_ip"


class KeywordConfig(_Section):
    provider: str = "bm25s"
    tokenizer: str = "tech_aware"


class LoggingConfig(_Section):
    level: str = "INFO"


class Settings(_Section):
    use_case: str = "resume"
    paths: PathsConfig = PathsConfig()
    ingestion: IngestionConfig = IngestionConfig()
    parser: ParserConfig = ParserConfig()
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig
    chunking: ChunkingConfig = ChunkingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    keyword: KeywordConfig = KeywordConfig()
    logging: LoggingConfig = LoggingConfig()

    # Derived directories under paths.data_dir.
    @property
    def raw_dir(self) -> Path:
        return self.paths.data_dir / "raw"

    @property
    def markdown_dir(self) -> Path:
        return self.paths.data_dir / "markdown"

    @property
    def profiles_dir(self) -> Path:
        return self.paths.data_dir / "profiles"

    @property
    def chunks_dir(self) -> Path:
        return self.paths.data_dir / "chunks"

    @property
    def indexes_dir(self) -> Path:
        return self.paths.data_dir / "indexes"


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from `path`, $RAG_CONFIG, or configs/config.yaml (in that order)."""
    config_path = Path(path or os.environ.get("RAG_CONFIG") or DEFAULT_CONFIG_PATH)
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    data_dir_override = os.environ.get("RAG_DATA_DIR")
    if data_dir_override:
        raw.setdefault("paths", {})
        raw["paths"]["data_dir"] = data_dir_override
        raw["paths"]["db"] = str(Path(data_dir_override) / "app.db")

    settings = Settings.model_validate(raw)
    # Relative paths are anchored at the project root so the app works from any cwd.
    paths = settings.paths
    if not paths.data_dir.is_absolute():
        paths.data_dir = PROJECT_ROOT / paths.data_dir
    if not paths.db.is_absolute():
        paths.db = PROJECT_ROOT / paths.db
    return settings
