from __future__ import annotations

import logging
import re
import tomllib
from pathlib import Path

import pymupdf

from mcp_book.config import IngestConfig
from mcp_book.models import BookMetadata, ExtractedBook, PageText, Table, TocEntry

logger = logging.getLogger(__name__)

TEXT_FLAGS = pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_PRESERVE_WHITESPACE
JUNK_TITLE = re.compile(r"^(untitled|microsoft word\b|document\d*$)|\.(docx?|indd|pdf)$", re.I)
HYPHENATED_LINE_END = re.compile(r"(?<=[^\W\d_])[-\u00ad\u2010]$")
PAGE_LABEL_TITLE = re.compile(r"^(\d+|[ivxlcdm]+)$", re.I)


class OcrUnavailableError(RuntimeError):
    pass


def extract_book(path: Path, config: IngestConfig) -> ExtractedBook:
    with pymupdf.open(path) as document:
        metadata = read_metadata(path, document)
        toc = read_toc(document)
        labels = outline_page_labels(toc, document.page_count)
        if labels:
            logger.info("%s: the outline holds page numbers, not chapters", path.name)
            toc = []
        pages = [extract_page(page, config) for page in document]
    for page in pages:
        page.label = page.label or labels.get(page.page)
    tables = extract_tables(path) if config.extract_tables else []
    return ExtractedBook(metadata=metadata, pages=pages, toc=toc, tables=tables)


def read_metadata(path: Path, document: pymupdf.Document) -> BookMetadata:
    pdf_metadata = document.metadata or {}
    title = (pdf_metadata.get("title") or "").strip()
    author = (pdf_metadata.get("author") or "").strip() or None
    if not title or JUNK_TITLE.search(title):
        title = re.sub(r"[_-]+", " ", path.stem).strip()

    sidecar = path.with_suffix(".toml")
    if sidecar.is_file():
        overrides = tomllib.loads(sidecar.read_text(encoding="utf-8"))
        title = overrides.get("title", title)
        author = overrides.get("author", author)

    return BookMetadata(title=title, author=author, page_count=document.page_count)


def read_toc(document: pymupdf.Document) -> list[TocEntry]:
    entries = []
    for level, title, page in document.get_toc(simple=True):
        title = " ".join(str(title).split())
        if title and 1 <= page <= document.page_count:
            entries.append(TocEntry(level=level, title=title, page=page))
    return entries


def join_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if text and HYPHENATED_LINE_END.search(text) and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text}\n{line}" if text else line
    return text


def outline_page_labels(entries: list[TocEntry], page_count: int) -> dict[int, str]:
    pages = {entry.page for entry in entries}
    labels = {entry.page: entry.title for entry in entries if PAGE_LABEL_TITLE.match(entry.title)}
    if (
        len(entries) >= 0.8 * page_count
        and len(pages) == len(entries)
        and all(entry.level == 1 for entry in entries)
        and len(labels) >= 0.8 * len(entries)
    ):
        return labels
    return {}


def _blocks_to_text(blocks: list[tuple]) -> str:
    paragraphs = []
    for block in blocks:
        if block[6] != 0:
            continue
        lines = [line.rstrip() for line in block[4].splitlines()]
        paragraph = join_lines([line for line in lines if line.strip()])
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def extract_page(page: pymupdf.Page, config: IngestConfig) -> PageText:
    text = _blocks_to_text(page.get_text("blocks", flags=TEXT_FLAGS))
    used_ocr = False
    if config.ocr and len(text.strip()) < config.ocr_min_chars:
        try:
            ocr_text = ocr_page(page, config)
        except OcrUnavailableError as error:
            logger.warning("Page %d: OCR skipped (%s)", page.number + 1, error)
        else:
            if len(ocr_text.strip()) > len(text.strip()):
                text, used_ocr = ocr_text, True
    return PageText(page=page.number + 1, text=text, label=page.get_label() or None, ocr=used_ocr)


def ocr_page(page: pymupdf.Page, config: IngestConfig) -> str:
    try:
        textpage = page.get_textpage_ocr(
            flags=TEXT_FLAGS, language=config.ocr_languages, dpi=config.ocr_dpi, full=True
        )
    except Exception as error:
        raise OcrUnavailableError(
            f"Tesseract not available or language data missing: {error}"
        ) from error
    return _blocks_to_text(page.get_text("blocks", textpage=textpage, flags=TEXT_FLAGS))


def extract_tables(path: Path) -> list[Table]:
    import pdfplumber

    tables: list[Table] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                raw_tables = page.extract_tables()
            except Exception as error:
                logger.warning("Page %d: table extraction failed (%s)", page_number, error)
                continue
            index = 0
            for raw in raw_tables:
                rows = normalize_rows(raw)
                if len(rows) < 2 or max(len(row) for row in rows) < 2:
                    continue
                tables.append(
                    Table(page=page_number, index=index, rows=rows, markdown=rows_to_markdown(rows))
                )
                index += 1
            page.flush_cache()
    return tables


def normalize_rows(raw: list[list[str | None]]) -> list[list[str]]:
    rows = [[" ".join((cell or "").split()) for cell in row] for row in raw]
    rows = [row for row in rows if any(row)]
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def rows_to_markdown(rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"

    header, *body = rows
    return "\n".join([line(header), line(["---"] * len(header)), *(line(row) for row in body)])
