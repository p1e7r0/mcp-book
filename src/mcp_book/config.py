from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

CONFIG_ENV = "MCP_BOOK_CONFIG"
DATABASE_NAME = "library.sqlite"


@dataclass
class LibraryConfig:
    pdf_dir: Path = Path("library")
    index_dir: Path = Path("data")


@dataclass
class EmbeddingConfig:
    backend: str = "sentence-transformers"
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 16
    query_prefix: str = ""
    passage_prefix: str = ""


@dataclass
class ChunkingConfig:
    target_tokens: int = 650
    max_tokens: int = 800
    overlap_tokens: int = 100


@dataclass
class IngestConfig:
    ocr: bool = True
    ocr_languages: str = "ita+eng"
    ocr_min_chars: int = 25
    ocr_dpi: int = 300
    extract_tables: bool = True


@dataclass
class SearchConfig:
    top_k: int = 8
    candidates: int = 50
    rrf_k: int = 60


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class Config:
    library: LibraryConfig = field(default_factory=LibraryConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @property
    def database_path(self) -> Path:
        return self.library.index_dir / DATABASE_NAME

    def validate(self) -> None:
        chunking = self.chunking
        if not 0 < chunking.target_tokens <= chunking.max_tokens:
            raise ValueError("chunking: target_tokens must be > 0 and <= max_tokens")
        if not 0 <= chunking.overlap_tokens < chunking.target_tokens:
            raise ValueError("chunking: overlap_tokens must be >= 0 and < target_tokens")
        if self.embedding.backend not in {"sentence-transformers", "hash", "none"}:
            raise ValueError(f"embedding: unknown backend {self.embedding.backend!r}")


def _build_section(section_type: type, values: dict[str, Any], section_name: str) -> Any:
    known = {item.name: item.type for item in fields(section_type)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"[{section_name}]: unknown keys {sorted(unknown)}")
    return section_type(**values)


def find_config_path(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get(CONFIG_ENV):
        candidates.append(Path(os.environ[CONFIG_ENV]).expanduser())
    candidates.append(Path.cwd() / "config.toml")
    candidates.append(Path.home() / ".config" / "mcp-book" / "config.toml")
    return next((path for path in candidates if path.is_file()), None)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    config_path = find_config_path(path)
    raw: dict[str, Any] = {}
    base_dir = Path.cwd()
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        base_dir = config_path.resolve().parent

    sections = {item.name: item.default_factory for item in fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")

    config = Config(
        **{
            name: _build_section(type(factory()), raw.get(name, {}), name)
            for name, factory in sections.items()
        }
    )
    config.library.pdf_dir = _resolve(base_dir, config.library.pdf_dir)
    config.library.index_dir = _resolve(base_dir, config.library.index_dir)
    config.validate()
    return config


def _resolve(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()
