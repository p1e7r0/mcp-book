from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from mcp_book.config import ChunkingConfig
from mcp_book.models import Chunk, PageText, TocEntry

TokenCounter = Callable[[str], int]

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+(?=[\"«(\[A-ZÀ-ÖØ-Ý0-9])")


def approximate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


@dataclass
class Unit:
    text: str
    page: int
    tokens: int
    separator: str


@dataclass
class Segment:
    chapter: str | None
    section: str | None
    units: list[Unit] = field(default_factory=list)


def normalize_heading(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def heading_positions(paragraphs: list[str], entries: list[TocEntry]) -> list[int]:
    """Index of the paragraph where each TOC entry starts on its page (0 when not found)."""
    normalized = [normalize_heading(paragraph) for paragraph in paragraphs]
    positions = []
    cursor = 0
    for entry in entries:
        title = normalize_heading(entry.title)
        found = next(
            (
                index
                for index in range(cursor, len(normalized))
                if title and normalized[index].startswith(title)
            ),
            None,
        )
        if found is not None:
            cursor = found
        positions.append(cursor)
    return positions


def split_sentences(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    pieces = []
    for sentence in SENTENCE_BREAK.split(text):
        if not sentence.strip():
            continue
        tokens = count_tokens(sentence)
        if tokens <= max_tokens:
            pieces.append(sentence)
            continue
        words = sentence.split()
        window = max(1, int(max_tokens * len(words) / tokens * 0.9))
        pieces.extend(" ".join(words[i : i + window]) for i in range(0, len(words), window))
    return pieces


def build_segments(
    pages: Iterable[PageText], toc: list[TocEntry], count_tokens: TokenCounter, max_tokens: int
) -> list[Segment]:
    entries_by_page: dict[int, list[TocEntry]] = {}
    for entry in toc:
        entries_by_page.setdefault(entry.page, []).append(entry)

    path: list[str] = []
    segments: list[Segment] = []

    def current_segment() -> Segment:
        chapter = path[0] if path else None
        section = " › ".join(path[1:]) or None
        if not segments or (segments[-1].chapter, segments[-1].section) != (chapter, section):
            segments.append(Segment(chapter=chapter, section=section))
        return segments[-1]

    for page in pages:
        paragraphs = [p.strip() for p in PARAGRAPH_BREAK.split(page.text) if p.strip()]
        entries = entries_by_page.get(page.page, [])
        positions = heading_positions(paragraphs, entries)
        pending = list(zip(positions, entries, strict=True))

        for index in range(max(len(paragraphs), 1)):
            while pending and pending[0][0] <= index:
                _, entry = pending.pop(0)
                level = max(1, entry.level)
                path[:] = path[: level - 1] + [entry.title]
            if index >= len(paragraphs):
                break
            segment = current_segment()
            sentences = split_sentences(paragraphs[index], count_tokens, max_tokens)
            for sentence_index, piece in enumerate(sentences):
                if not segment.units:
                    separator = ""
                elif sentence_index > 0:
                    separator = " "
                else:
                    separator = "\n\n"
                segment.units.append(
                    Unit(
                        text=piece, page=page.page, tokens=count_tokens(piece), separator=separator
                    )
                )
    return merge_heading_only_segments([s for s in segments if s.units], toc)


def merge_heading_only_segments(segments: list[Segment], toc: list[TocEntry]) -> list[Segment]:
    """Fold a segment that holds only its own heading into the next one of the same chapter."""
    titles = {normalize_heading(entry.title) for entry in toc}
    merged: list[Segment] = []
    carried: list[Unit] = []
    for index, segment in enumerate(segments):
        if carried:
            segment.units[0].separator = "\n\n"
            segment.units = carried + segment.units
            carried = []
        following = segments[index + 1] if index + 1 < len(segments) else None
        heading_only = all(normalize_heading(unit.text) in titles for unit in segment.units)
        if heading_only and following is not None and following.chapter == segment.chapter:
            carried = segment.units
            continue
        merged.append(segment)
    return merged


def pack(units: list[Unit], config: ChunkingConfig) -> list[list[Unit]]:
    groups: list[list[Unit]] = []
    current: list[Unit] = []
    current_tokens = 0
    for unit in units:
        if current and (
            current_tokens >= config.target_tokens
            or current_tokens + unit.tokens > config.max_tokens
        ):
            groups.append(current)
            overlap: list[Unit] = []
            overlap_tokens = 0
            for previous in reversed(current):
                if overlap_tokens + previous.tokens > config.overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_tokens += previous.tokens
            while overlap and overlap_tokens + unit.tokens > config.max_tokens:
                overlap_tokens -= overlap.pop(0).tokens
            current, current_tokens = overlap, overlap_tokens
        current.append(unit)
        current_tokens += unit.tokens
    if current:
        groups.append(current)
    return groups


def join_units(units: list[Unit]) -> str:
    return "".join(
        ("" if index == 0 else unit.separator) + unit.text for index, unit in enumerate(units)
    )


def chunk_book(
    pages: list[PageText],
    toc: list[TocEntry],
    config: ChunkingConfig,
    count_tokens: TokenCounter = approximate_tokens,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for segment in build_segments(pages, toc, count_tokens, config.max_tokens):
        for group in pack(segment.units, config):
            chunks.append(
                Chunk(
                    sequence=len(chunks),
                    text=join_units(group),
                    page_start=min(unit.page for unit in group),
                    page_end=max(unit.page for unit in group),
                    chapter=segment.chapter,
                    section=segment.section,
                    token_count=sum(unit.tokens for unit in group),
                )
            )
    return chunks
