from pathlib import Path

import pymupdf

from mcp_book.config import IngestConfig
from mcp_book.extract import extract_book
from mcp_book.margins import merge_variants, running_key

CHAPTERS = {range(1, 6): "SALSE", range(7, 12): "COTTURE"}


def build_scanned_book(path: Path, *, toc: list[list] | None = None) -> Path:
    document = pymupdf.open()
    for number in range(1, 13):
        page = document.new_page(width=595, height=842)
        page.insert_text((60, 100), f"Testo della pagina {number} con una ricetta.", fontsize=11)
        footer = "MANUALE DI PROVA"
        if number % 2:
            footer = next((title for pages, title in CHAPTERS.items() if number in pages), footer)
        if number == 3:
            footer = "SAlSE"
        page.insert_text((250, 825), footer, fontsize=9)
        page.insert_text((60, 825), str(number), fontsize=9)
        if number == 4:
            page.insert_text((400, 825), "Nota a margine unica", fontsize=9)
    if toc:
        document.set_toc(toc)
    document.save(path)
    return path


def test_running_footers_become_chapters_and_leave_the_text(tmp_path: Path) -> None:
    book = extract_book(
        build_scanned_book(tmp_path / "libro.pdf"), IngestConfig(ocr=False, extract_tables=False)
    )

    assert [(entry.level, entry.title, entry.page) for entry in book.toc] == [
        (1, "Salse", 1),
        (1, "Cotture", 7),
    ]
    assert book.pages[0].text == "Testo della pagina 1 con una ricetta."
    assert book.pages[3].text == "Testo della pagina 4 con una ricetta.\n\nNota a margine unica"
    joined = "\n".join(page.text for page in book.pages)
    assert "MANUALE" not in joined and "SAlSE" not in joined and "COTTURE" not in joined


def test_ocr_variants_of_a_running_line_are_merged() -> None:
    pages_by_key = {
        running_key("GElS"): set(range(1, 30)),
        running_key("GEL 5"): {3, 5, 7},
        running_key("FOAMS"): set(range(40, 60)),
    }

    canonical = merge_variants(pages_by_key)

    assert canonical == {"gels": "gels", "gel": "gels", "foams": "foams"}


def test_outline_wins_over_running_footers(tmp_path: Path) -> None:
    toc = [[1, "Le salse madri", 1], [1, "Le cotture", 7]]
    pdf = build_scanned_book(tmp_path / "libro.pdf", toc=toc)

    book = extract_book(pdf, IngestConfig(ocr=False, extract_tables=False))

    assert [entry.title for entry in book.toc] == ["Le salse madri", "Le cotture"]
    assert "SALSE" not in book.pages[0].text
