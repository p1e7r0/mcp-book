import shutil
from pathlib import Path

import pytest

from mcp_book.cli import main
from mcp_book.config import Config, load_config
from mcp_book.embeddings import HashEmbedder
from mcp_book.ingest import ingest_folder, slugify
from mcp_book.store import Store

from conftest import requires_tesseract
from sample_pdf import TITLE, build_sample_pdf


@pytest.fixture
def config(tmp_path: Path) -> Config:
    config_file = tmp_path / "config.toml"
    config_file.write_text('[embedding]\nbackend = "hash"\n')
    return load_config(config_file)


@pytest.fixture
def library(tmp_path: Path, sample_pdf: Path) -> Path:
    folder = tmp_path / "library"
    (folder / "sub").mkdir(parents=True)
    shutil.copy(sample_pdf, folder / "Manuale Cucina.pdf")
    build_sample_pdf(folder / "sub" / "secondo.PDF", with_scanned_page=False)
    return folder


def test_slugify() -> None:
    assert slugify("Il Cucchiaio d'Argento (2ª ed.)") == "il-cucchiaio-d-argento-2a-ed"
    assert slugify("###") == "book"


@requires_tesseract
def test_ingest_is_incremental(config: Config, library: Path) -> None:
    store = Store(config.database_path)
    embedder = HashEmbedder()

    first = ingest_folder(library, config, store, embedder)
    second = ingest_folder(library, config, store, embedder)

    assert sorted(first.added) == ["manuale-cucina", "secondo"]
    assert sorted(second.skipped) == ["manuale-cucina", "secondo"]
    book = store.get_book("manuale-cucina")
    assert book is not None and book.title == TITLE and book.page_count == 5
    assert store.query("SELECT ocr FROM pages WHERE book_id = 'manuale-cucina' AND page = 5")[0][0]
    assert store.tables("manuale-cucina", 4)
    assert store.has_vector_table


def test_changed_file_is_reingested_with_same_id(config: Config, library: Path) -> None:
    config.ingest.ocr = False
    store = Store(config.database_path)
    ingest_folder(library, config, store, HashEmbedder())

    build_sample_pdf(library / "sub" / "secondo.PDF", with_scanned_page=True)
    report = ingest_folder(library, config, store, HashEmbedder())

    assert report.updated == ["secondo"]
    assert report.skipped == ["manuale-cucina"]


def test_prune_and_rebuild(config: Config, library: Path) -> None:
    config.ingest.ocr = False
    store = Store(config.database_path)
    ingest_folder(library, config, store, HashEmbedder())

    (library / "sub" / "secondo.PDF").unlink()
    pruned = ingest_folder(library, config, store, HashEmbedder(), prune=True)
    rebuilt = ingest_folder(library, config, store, HashEmbedder(dimension=64), rebuild=True)

    assert pruned.removed == ["secondo"]
    assert rebuilt.added == ["manuale-cucina"]
    assert store.embedding_model == "hash-64"


def test_model_mismatch_stops_ingest(config: Config, library: Path) -> None:
    config.ingest.ocr = False
    store = Store(config.database_path)
    ingest_folder(library, config, store, HashEmbedder())

    with pytest.raises(ValueError, match="--rebuild"):
        ingest_folder(library, config, store, HashEmbedder(dimension=64))


def test_keyword_only_index(config: Config, library: Path) -> None:
    config.ingest.ocr = False
    store = Store(config.database_path)

    report = ingest_folder(library, config, store, None)

    assert len(report.added) == 2
    assert not store.has_vector_table


def test_cli_ingest_and_books(
    tmp_path: Path, library: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('[embedding]\nbackend = "hash"\n')

    assert main(["-c", str(config_file), "ingest", str(library), "--no-ocr"]) == 0
    assert main(["-c", str(config_file), "books"]) == 0

    output = capsys.readouterr()
    assert "2 added" in output.err
    assert "manuale-cucina\tManuale di cucina tecnica\tMario Rossi\t5 pages" in output.out


def test_cli_reports_missing_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('[embedding]\nbackend = "none"\n')

    assert main(["-c", str(config_file), "ingest", str(tmp_path / "missing")]) == 1
    assert "PDF folder not found" in capsys.readouterr().err
