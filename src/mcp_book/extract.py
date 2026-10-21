from __future__ import annotations

import logging
import re
import tomllib
from pathlib import Path

import pymupdf

from mcp_book.config import IngestConfig
from mcp_book.margins import PageParts, join_parts, remove_running_lines
from mcp_book.models import BookMetadata, ExtractedBook, PageText, Table, TocEntry

logger = logging.getLogger(__name__)

TEXT_FLAGS = pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_PRESERVE_WHITESPACE
JUNK_TITLE = re.compile(r"^(untitled|microsoft word\b|document\d*$)|\.(docx?|indd|pdf)$", re.I)
HYPHENATED_LINE_END = re.compile(r"(?<=[^\W\d_])[-\u00ad\u2010]$")
PAGE_LABEL_TITLE = re.compile(r"^(\d+|[ivxlcdm]+)$", re.I)
OCR_WORD = re.compile(r"[^\W\d_]{3,}")
OCR_QUANTITY = re.compile(r"^[-—]?\d+(?:[.,·]\d+)?\s?(?:%|°[CF]|[a-z]{1,3})?$")
MARGIN = 0.05
NUMBER = r"[-—–]?\d+(?:[.,]\d+)?(?:\s?[-–]\s?\d+(?:[.,]\d+)?)?"
CELSIUS_MARK = r"(?:[°º•·'\"’”`]|o(?=\s?[cC]))?"
FAHRENHEIT_MARK = r"(?:[°º•·'\"’”`]|[oO](?=[FfPp]))?"
TEMPERATURE_PAIR = re.compile(
    rf"(?P<celsius>{NUMBER})\s?{CELSIUS_MARK}\s?[cC]\s*(?:[I/l|]|(?<=\s)1(?=\s))\s*"
    rf"(?P<fahrenheit>{NUMBER})\s?{FAHRENHEIT_MARK}\s?[FfPpr](?P<after>[a-z]?)"
)


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
        parts = [extract_page_parts(page, config) for page in document]
    for part in parts:
        part.page.label = part.page.label or labels.get(part.page.page)
    pages, running_chapters = remove_running_lines(parts)
    if not toc and running_chapters:
        logger.info("%s: %d chapters read from running heads", path.name, len(running_chapters))
        toc = running_chapters
    tables = extract_tables(path) if config.extract_tables else []
    return ExtractedBook(metadata=metadata, pages=pages, toc=toc, tables=tables)


def read_metadata(path: Path, document: pymupdf.Document) -> BookMetadata:
    pdf_metadata = document.metadata or {}
    title = (pdf_metadata.get("title") or "").strip()
    author = (pdf_metadata.get("author") or "").strip() or None
    if not title or JUNK_TITLE.search(title):
        title = " ".join(re.sub(r"[_-]+", " ", path.stem).split())

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


def _block_lines(blocks: list[tuple]) -> list[str]:
    return [line.strip() for block in blocks for line in block[4].splitlines() if line.strip()]


def split_margins(blocks: list[tuple], height: float) -> tuple[list[str], str, list[str]]:
    header, body, footer = [], [], []
    for block in blocks:
        if block[6] != 0:
            continue
        if block[3] <= MARGIN * height:
            header.append(block)
        elif block[1] >= (1 - MARGIN) * height:
            footer.append(block)
        else:
            body.append(block)
    return _block_lines(header), _blocks_to_text(body), _block_lines(footer)


def _length(header: list[str], body: str, footer: list[str]) -> int:
    return len(body.strip()) + sum(len(line) for line in header + footer)


def extract_page_parts(page: pymupdf.Page, config: IngestConfig) -> PageParts:
    header, body, footer = split_margins(
        page.get_text("blocks", flags=TEXT_FLAGS), page.rect.height
    )
    used_ocr = False
    if config.ocr and _length(header, body, footer) < config.ocr_min_chars:
        try:
            ocr_header, ocr_body, ocr_footer = ocr_page(page, config)
        except OcrUnavailableError as error:
            logger.warning("Page %d: OCR skipped (%s)", page.number + 1, error)
        else:
            if _length(ocr_header, ocr_body, ocr_footer) > _length(header, body, footer):
                header, body, footer, used_ocr = ocr_header, ocr_body, ocr_footer, True
    body = normalize_temperatures(body)
    text = PageText(page=page.number + 1, text=body, label=page.get_label() or None, ocr=used_ocr)
    return PageParts(page=text, header=header, footer=footer)


def extract_page(page: pymupdf.Page, config: IngestConfig) -> PageText:
    return join_parts(extract_page_parts(page, config))


def ocr_page(page: pymupdf.Page, config: IngestConfig) -> tuple[list[str], str, list[str]]:
    try:
        textpage = page.get_textpage_ocr(
            flags=TEXT_FLAGS, language=config.ocr_languages, dpi=config.ocr_dpi, full=True
        )
    except Exception as error:
        raise OcrUnavailableError(
            f"Tesseract not available or language data missing: {error}"
        ) from error
    blocks = page.get_text("blocks", textpage=textpage, flags=TEXT_FLAGS)
    header, body, footer = split_margins(blocks, page.rect.height)
    return (
        drop_ocr_noise("\n".join(header)).splitlines(),
        drop_ocr_noise(body),
        drop_ocr_noise("\n".join(footer)).splitlines(),
    )


def _clean_number(number: str) -> str:
    return re.sub(r"\s", "", number).replace("—", "-").replace("–", "-")


def normalize_temperatures(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        after = f" {match['after']}" if match["after"] else ""
        celsius, fahrenheit = _clean_number(match["celsius"]), _clean_number(match["fahrenheit"])
        return f"{celsius} °C / {fahrenheit} °F{after}"

    return TEMPERATURE_PAIR.sub(replace, text)


def is_text_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) >= 2 and OCR_QUANTITY.match(stripped):
        return True
    words = OCR_WORD.findall(stripped)
    visible = len("".join(stripped.split()))
    return len(words) >= 2 and sum(len(word) for word in words) >= 0.6 * visible


def drop_ocr_noise(text: str) -> str:
    paragraphs = [paragraph.splitlines() for paragraph in text.split("\n\n")]
    lines = [line for paragraph in paragraphs for line in paragraph]
    strong = [is_text_line(line) for line in lines]
    kept = []
    for index, line in enumerate(lines):
        previous = strong[index - 1] if index > 0 else True
        following = strong[index + 1] if index + 1 < len(lines) else True
        weak = any(character.isalnum() for character in line) and previous and following
        kept.append(strong[index] or weak)

    cleaned, position = [], 0
    for paragraph in paragraphs:
        kept_lines = [line for offset, line in enumerate(paragraph) if kept[position + offset]]
        position += len(paragraph)
        if kept_lines:
            cleaned.append("\n".join(kept_lines))
    return "\n\n".join(cleaned)


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
