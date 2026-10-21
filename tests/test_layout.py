from pathlib import Path

import pymupdf

from mcp_book.config import IngestConfig
from mcp_book.extract import drop_ocr_noise_lines, extract_book
from mcp_book.layout import Line, reading_order

LEFT = [
    "Primo passo della maionese: separare i tuorli e portarli a temperatura ambiente.",
    "Secondo passo della maionese: aggiungere l'olio a filo, mescolando di continuo.",
    "Terzo passo della maionese: regolare di sale e conservare in frigorifero.",
]
RIGHT = [
    "Primo passo della besciamella: sciogliere il burro e unire la farina setacciata.",
    "Secondo passo della besciamella: versare il latte caldo poco alla volta.",
    "Terzo passo della besciamella: cuocere dieci minuti finché vela il cucchiaio.",
]


def test_side_by_side_lines_are_split_into_columns() -> None:
    lines = [
        Line(40, 100, 260, 112, "sinistra uno"),
        Line(300, 100, 540, 112, "destra uno"),
        Line(40, 114, 250, 126, "sinistra due"),
        Line(300, 114, 530, 126, "destra due"),
    ]

    paragraphs = reading_order([lines])

    assert [[line.text for line in paragraph] for paragraph in paragraphs] == [
        ["sinistra uno", "sinistra due"],
        ["destra uno", "destra due"],
    ]


def test_single_column_is_left_alone() -> None:
    lines = [Line(40, 100, 540, 112, "riga uno"), Line(40, 114, 540, 126, "riga due")]

    assert reading_order([lines]) == [lines]


def test_ocr_noise_is_dropped_before_columns_are_split() -> None:
    groups = [
        [
            Line(40, 100, 260, 112, "Floral essences can be distilled"),
            Line(300, 100, 340, 112, "basil"),
            Line(40, 114, 260, 126, "in a rotary evaporator at low heat"),
            Line(300, 114, 330, 126, "# EE"),
            Line(300, 128, 330, 140, "7 | EI"),
            Line(40, 128, 260, 140, "that anything you can smell is volatile"),
        ]
    ]

    kept = drop_ocr_noise_lines(groups)

    assert [line.text for line in kept[0]] == [
        "Floral essences can be distilled",
        "basil",
        "in a rotary evaporator at low heat",
        "that anything you can smell is volatile",
    ]


def test_columns_written_row_by_row_are_read_column_by_column(tmp_path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((50, 80), "DUE SALSE A CONFRONTO", fontsize=14, fontname="hebo")
    for row, (left, right) in enumerate(zip(LEFT, RIGHT, strict=True)):
        top = 110 + row * 70
        page.insert_textbox(pymupdf.Rect(50, top, 280, top + 60), left, fontsize=10)
        page.insert_textbox(pymupdf.Rect(315, top, 545, top + 60), right, fontsize=10)
    document.save(tmp_path / "colonne.pdf")

    book = extract_book(tmp_path / "colonne.pdf", IngestConfig(ocr=False, extract_tables=False))

    text = " ".join(book.pages[0].text.split())
    positions = [text.index(" ".join(step.split()[:4])) for step in LEFT + RIGHT]
    assert positions == sorted(positions)
