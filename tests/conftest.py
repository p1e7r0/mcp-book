from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from sample_pdf import build_sample_pdf

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Tesseract is not installed"
)


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("library")
    return build_sample_pdf(folder / "manuale-cucina.pdf")


@pytest.fixture(scope="session")
def indexed_config(tmp_path_factory: pytest.TempPathFactory, sample_pdf: Path):
    from mcp_book.config import load_config
    from mcp_book.embeddings import HashEmbedder
    from mcp_book.ingest import ingest_folder
    from mcp_book.store import Store

    root = tmp_path_factory.mktemp("indexed")
    (root / "library").mkdir()
    shutil.copy(sample_pdf, root / "library" / sample_pdf.name)
    (root / "config.toml").write_text('[embedding]\nbackend = "hash"\n')
    config = load_config(root / "config.toml")
    config.ingest.ocr = shutil.which("tesseract") is not None
    store = Store(config.database_path)
    report = ingest_folder(config.library.pdf_dir, config, store, HashEmbedder())
    store.close()
    assert report.added == ["manuale-cucina"]
    return config


@pytest.fixture
def library(indexed_config):
    from mcp_book.library import Library

    library = Library(indexed_config)
    yield library
    library.close()
