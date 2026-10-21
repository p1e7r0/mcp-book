from pathlib import Path

import pymupdf

from mcp_book.config import IngestConfig
from mcp_book.extract import extract_book
from mcp_book.recipes import QUANTITY

COLUMNS = {"INGREDIENT": 40, "QUANTITY": 200, "SCALING": 270, "PROCEDURE": 340}


def write_recipe(page: pymupdf.Page, top: float, title: str, rows: list, steps: list) -> float:
    page.insert_text((40, top), title, fontsize=12, fontname="hebo")
    top += 24
    for header, x in COLUMNS.items():
        page.insert_text((x, top), header, fontsize=8, fontname="hebo")
    for index, step in enumerate(steps):
        page.insert_text((COLUMNS["PROCEDURE"], top + 14 + index * 12), step, fontsize=9)
    for ingredient, quantity, scaling in rows:
        top += 14
        lines = ingredient.split("\n")
        for offset, line in enumerate(lines):
            page.insert_text((COLUMNS["INGREDIENT"], top + offset * 11), line, fontsize=9)
        page.insert_text((COLUMNS["QUANTITY"], top), quantity, fontsize=9)
        page.insert_text((COLUMNS["SCALING"], top), scaling, fontsize=9)
        top += 11 * (len(lines) - 1)
    return top


def build_recipe_page(path: Path) -> Path:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    top = write_recipe(
        page,
        80,
        "MAIONESE",
        [
            ("Tuorlo d'uovo,\npastorizzato", "20 g", "10%"),
            ("Olio di semi", "200 g", "100%"),
            ("Succo di limone", "10 g", "5%"),
            ("Sale", "to taste", ""),
        ],
        ["Frullare il tuorlo con il limone.", "Aggiungere l'olio a filo a 20 °C / 68 °F."],
    )
    write_recipe(
        page,
        top + 60,
        "BESCIAMELLA",
        [("Latte intero", "500 g", "100%"), ("Burro", "50 g", "10%")],
        ["Sciogliere il burro e cuocere il roux."],
    )
    page.insert_text(
        (40, 700), "La maionese si conserva in frigorifero per due giorni.", fontsize=10
    )
    document.save(path)
    return path


def test_recipe_tables_are_read_row_by_row(tmp_path: Path) -> None:
    book = extract_book(
        build_recipe_page(tmp_path / "ricette.pdf"), IngestConfig(ocr=False, extract_tables=False)
    )

    assert [table.rows for table in book.tables] == [
        [
            ["INGREDIENT", "QUANTITY", "SCALING"],
            ["Tuorlo d'uovo, pastorizzato", "20 g", "10%"],
            ["Olio di semi", "200 g", "100%"],
            ["Succo di limone", "10 g", "5%"],
            ["Sale", "to taste", ""],
        ],
        [
            ["INGREDIENT", "QUANTITY", "SCALING"],
            ["Latte intero", "500 g", "100%"],
            ["Burro", "50 g", "10%"],
        ],
    ]
    assert [(table.page, table.index) for table in book.tables] == [(1, 0), (1, 1)]


def test_recipe_text_keeps_title_table_procedure_and_prose(tmp_path: Path) -> None:
    book = extract_book(
        build_recipe_page(tmp_path / "ricette.pdf"), IngestConfig(ocr=False, extract_tables=False)
    )

    text = book.pages[0].text
    order = [
        "MAIONESE",
        "| Tuorlo d'uovo, pastorizzato | 20 g | 10% |",
        "PROCEDURE\nFrullare il tuorlo con il limone.",
        "BESCIAMELLA",
        "| Burro | 50 g | 10% |",
        "La maionese si conserva",
    ]
    assert [text.index(fragment) for fragment in order] == sorted(
        text.index(fragment) for fragment in order
    )
    assert text.count("Olio di semi") == 1
    assert "20 °C / 68 °F" in text


def test_quantity_pattern_accepts_ocr_digits_and_rejects_prose() -> None:
    accepted = ["100g", "1.5 kg", "SOg", "lOOg", "as needed", "75g (four large)", "3 sheets"]
    rejected = ["4, and wait until", "5 · PLATED-DISH", "2010)", "Salt", "Large eggs"]

    assert all(QUANTITY.match(value) for value in accepted)
    assert not any(QUANTITY.match(value) for value in rejected)
