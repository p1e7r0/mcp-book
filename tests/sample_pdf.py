"""Generates the sample cookbook used by the tests.

Layout (PDF page index → content):
1. Introduction, roman label "i"
2. "Le salse" › "Maionese", label "1"
3. end of "Maionese", then "Besciamella" starts mid-page
4. "Tabelle": a ruled table of cooking temperatures
5. "Cotture" › "Cottura sottovuoto": scanned page (image only, needs OCR)
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

TITLE = "Manuale di cucina tecnica"
AUTHOR = "Mario Rossi"

MAYONNAISE_PARAGRAPHS = [
    "La maionese è un'emulsione di olio in tuorlo d'uovo. Il tuorlo contiene "
    "lecitina, che stabilizza le goccioline di olio disperse nella fase acquosa.",
    "Per una maionese stabile tutti gli ingredienti devono essere a temperatura "
    "ambiente, circa 20 gradi. L'olio va aggiunto a filo, mescolando di continuo.",
    "Se la maionese impazzisce, si riparte da un tuorlo nuovo e si incorpora "
    "lentamente l'emulsione separata, un cucchiaio alla volta.",
]

BECHAMEL_TEXT = (
    "La besciamella si prepara con un roux bianco di burro e farina in parti "
    "uguali, a cui si aggiunge latte caldo. Si cuoce per dieci minuti a fuoco "
    "dolce finché la salsa vela il cucchiaio."
)

SOUS_VIDE_TEXT = (
    "Cottura sottovuoto\n"
    "Il controfiletto di manzo cuoce a 56 gradi per due ore.\n"
    "Il salmone cuoce a 45 gradi per venti minuti."
)

TABLE_ROWS = [
    ["Alimento", "Temperatura", "Tempo"],
    ["Manzo", "56 °C", "2 h"],
    ["Salmone", "45 °C", "20 min"],
    ["Uova", "64 °C", "45 min"],
]

TOC = [
    [1, "Introduzione", 1],
    [1, "Le salse", 2],
    [2, "Maionese", 2],
    [2, "Besciamella", 3],
    [1, "Tabelle", 4],
    [1, "Cotture", 5],
    [2, "Cottura sottovuoto", 5],
]


def _write_paragraphs(page: pymupdf.Page, paragraphs: list[str], top: float) -> float:
    for paragraph in paragraphs:
        rect = pymupdf.Rect(60, top, 540, top + 120)
        page.insert_textbox(rect, paragraph, fontsize=11, fontname="helv")
        top += 80
    return top


def _draw_table(page: pymupdf.Page, rows: list[list[str]], left: float, top: float) -> None:
    column_width, row_height = 140, 24
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            rect = pymupdf.Rect(
                left + column_index * column_width,
                top + row_index * row_height,
                left + (column_index + 1) * column_width,
                top + (row_index + 1) * row_height,
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 6, rect.y1 - 8), value, fontsize=10, fontname="helv")


def _scanned_page_image(text: str) -> bytes:
    source = pymupdf.open()
    page = source.new_page(width=595, height=842)
    page.insert_textbox(pymupdf.Rect(60, 80, 540, 400), text, fontsize=16, fontname="helv")
    image = page.get_pixmap(dpi=200).tobytes("png")
    source.close()
    return image


def build_sample_pdf(path: Path, *, with_scanned_page: bool = True) -> Path:
    document = pymupdf.open()

    page = document.new_page(width=595, height=842)
    page.insert_text((60, 70), "Introduzione", fontsize=18, fontname="hebo")
    _write_paragraphs(
        page,
        [
            "Questo manuale raccoglie tecniche di base della cucina professionale: "
            "salse madri, emulsioni e cotture a bassa temperatura."
        ],
        100,
    )

    page = document.new_page(width=595, height=842)
    page.insert_text((60, 70), "Le salse", fontsize=20, fontname="hebo")
    page.insert_text((60, 105), "Maionese", fontsize=15, fontname="hebo")
    _write_paragraphs(page, MAYONNAISE_PARAGRAPHS, 125)

    page = document.new_page(width=595, height=842)
    _write_paragraphs(
        page,
        ["La maionese si conserva in frigorifero per un massimo di due giorni."],
        70,
    )
    page.insert_text((60, 190), "Besciamella", fontsize=15, fontname="hebo")
    _write_paragraphs(page, [BECHAMEL_TEXT], 210)

    page = document.new_page(width=595, height=842)
    page.insert_text((60, 70), "Tabelle", fontsize=20, fontname="hebo")
    page.insert_text((60, 100), "Temperature al cuore", fontsize=12, fontname="helv")
    _draw_table(page, TABLE_ROWS, 60, 120)

    page = document.new_page(width=595, height=842)
    if with_scanned_page:
        page.insert_image(page.rect, stream=_scanned_page_image("Cotture\n\n" + SOUS_VIDE_TEXT))
    else:
        page.insert_textbox(
            pymupdf.Rect(60, 80, 540, 400), "Cotture\n\n" + SOUS_VIDE_TEXT, fontsize=12
        )

    document.set_toc(TOC)
    document.set_metadata({"title": TITLE, "author": AUTHOR})
    document.set_page_labels(
        [
            {"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1},
            {"startpage": 1, "prefix": "", "style": "D", "firstpagenum": 1},
        ]
    )
    document.save(path)
    document.close()
    return path
