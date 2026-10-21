from pathlib import Path

from mcp_book.config import IngestConfig
from mcp_book.extract import extract_book, rows_to_markdown

from conftest import requires_tesseract
from sample_pdf import AUTHOR, TITLE, build_sample_pdf


def test_metadata_toc_and_labels(sample_pdf: Path) -> None:
    book = extract_book(sample_pdf, IngestConfig(ocr=False, extract_tables=False))

    assert book.metadata.title == TITLE
    assert book.metadata.author == AUTHOR
    assert book.metadata.page_count == 5
    assert [(entry.level, entry.title, entry.page) for entry in book.toc][:3] == [
        (1, "Introduzione", 1),
        (1, "Le salse", 2),
        (2, "Maionese", 2),
    ]
    assert [page.label for page in book.pages] == ["i", "1", "2", "3", "4"]
    assert "lecitina" in book.pages[1].text


def test_tables_are_extracted_as_markdown(sample_pdf: Path) -> None:
    book = extract_book(sample_pdf, IngestConfig(ocr=False))

    assert len(book.tables) == 1
    table = book.tables[0]
    assert table.page == 4
    assert table.rows[0] == ["Alimento", "Temperatura", "Tempo"]
    assert "| Salmone | 45 °C | 20 min |" in table.markdown


@requires_tesseract
def test_scanned_page_is_ocred(sample_pdf: Path) -> None:
    book = extract_book(sample_pdf, IngestConfig(extract_tables=False))

    scanned = book.pages[4]
    assert scanned.ocr
    assert "controfiletto" in scanned.text
    assert not book.pages[1].ocr


def test_ocr_disabled_leaves_scanned_page_empty(sample_pdf: Path) -> None:
    book = extract_book(sample_pdf, IngestConfig(ocr=False, extract_tables=False))
    assert book.pages[4].text == ""


def test_sidecar_overrides_metadata(tmp_path: Path) -> None:
    pdf = build_sample_pdf(tmp_path / "libro.pdf", with_scanned_page=False)
    (tmp_path / "libro.toml").write_text('title = "Titolo corretto"\nauthor = "Anna Bianchi"\n')

    book = extract_book(pdf, IngestConfig(ocr=False, extract_tables=False))

    assert book.metadata.title == "Titolo corretto"
    assert book.metadata.author == "Anna Bianchi"


def test_markdown_escapes_pipes() -> None:
    assert rows_to_markdown([["a|b", "c"], ["1", "2"]]).splitlines() == [
        "| a\\|b | c |",
        "| --- | --- |",
        "| 1 | 2 |",
    ]
