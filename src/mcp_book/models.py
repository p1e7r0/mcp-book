from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TocEntry:
    level: int
    title: str
    page: int


@dataclass
class PageText:
    page: int
    text: str
    label: str | None = None
    ocr: bool = False


@dataclass
class Table:
    page: int
    index: int
    rows: list[list[str]]
    markdown: str


@dataclass
class BookMetadata:
    title: str
    author: str | None
    page_count: int


@dataclass
class ExtractedBook:
    metadata: BookMetadata
    pages: list[PageText]
    toc: list[TocEntry]
    tables: list[Table] = field(default_factory=list)


@dataclass
class Chunk:
    sequence: int
    text: str
    page_start: int
    page_end: int
    chapter: str | None
    section: str | None
    token_count: int
