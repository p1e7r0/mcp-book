import pytest

from mcp_book.library import Library, LibraryError

from conftest import requires_tesseract
from sample_pdf import TITLE

BOOK = "manuale-cucina"


def test_list_books(library: Library) -> None:
    [book] = library.list_books()["books"]
    assert book == {
        "book_id": BOOK,
        "title": TITLE,
        "author": "Mario Rossi",
        "pages": 5,
        "has_toc": True,
        "citation": TITLE,
    }


def test_toc_citations_use_printed_labels(library: Library) -> None:
    entries = library.get_toc(BOOK)["entries"]
    besciamella = next(entry for entry in entries if entry["title"] == "Besciamella")
    assert besciamella["page"] == 3
    assert besciamella["page_label"] == "2"
    assert besciamella["citation"] == f"{TITLE}, Le salse, p. 2"


def test_search_result_shape(library: Library) -> None:
    result = library.search("come stabilizzare la maionese", top_k=3)

    assert result["mode"] == "hybrid"
    top = result["results"][0]
    assert top["chapter"] == "Le salse" and top["section"] == "Maionese"
    assert top["book_id"] == BOOK and isinstance(top["chunk_id"], int)
    assert "maionese" in top["excerpt"].lower()
    assert top["citation"] == f"{TITLE}, Le salse, pp. 1–2"


def test_search_validation(library: Library) -> None:
    with pytest.raises(LibraryError, match="Unknown book_id"):
        library.search("salsa", ["nessuno"])
    with pytest.raises(LibraryError):
        library.search("   ")
    with pytest.raises(LibraryError):
        library.search("salsa", top_k=0)


def test_read_pages_clamps_and_cites_range(library: Library) -> None:
    result = library.read_pages(BOOK, 2, 3)

    assert [page["page"] for page in result["pages"]] == [2, 3]
    assert result["citation"] == f"{TITLE}, Le salse, pp. 1–2"
    assert result["pages"][1]["citation"] == f"{TITLE}, Le salse, p. 2"
    assert not result["truncated"]
    assert library.read_pages(BOOK, 4, 99)["end_page"] == 5
    with pytest.raises(LibraryError):
        library.read_pages(BOOK, 9, 10)
    with pytest.raises(LibraryError):
        library.read_pages(BOOK, 3, 2)


def test_read_pages_limit(library: Library, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mcp_book.library.MAX_PAGES_PER_READ", 2)
    result = library.read_pages(BOOK, 1, 5)
    assert result["end_page"] == 2
    assert result["truncated"] and result["next_start_page"] == 3


def test_read_chunk_with_context(library: Library) -> None:
    top = library.search("besciamella roux", top_k=1)["results"][0]

    result = library.read_chunk(top["chunk_id"], context=1)

    flags = [chunk["is_target"] for chunk in result["chunks"]]
    assert flags.count(True) == 1 and len(flags) == 3
    assert result["citation"] == top["citation"]
    assert library.read_chunk(top["chunk_id"], context=0)["chunks"][0]["is_target"]
    with pytest.raises(LibraryError, match="Unknown chunk_id"):
        library.read_chunk(999_999)


def test_get_tables(library: Library) -> None:
    result = library.get_tables(BOOK, 4)
    assert result["citation"] == f"{TITLE}, Tabelle, p. 3"
    assert "| Uova | 64 °C | 45 min |" in result["tables"][0]["markdown"]
    assert library.get_tables(BOOK, 2)["tables"] == []


@requires_tesseract
def test_ocr_text_is_searchable(library: Library) -> None:
    top = library.search("controfiletto", top_k=1)["results"][0]
    assert top["page_start"] == 5
    assert top["citation"] == f"{TITLE}, Cotture, p. 4"
