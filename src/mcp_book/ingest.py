from __future__ import annotations

import hashlib
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from mcp_book.chunking import approximate_tokens, chunk_book
from mcp_book.config import Config
from mcp_book.embeddings import Embedder
from mcp_book.extract import extract_book
from mcp_book.models import Chunk
from mcp_book.store import BookRecord, Store

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def slugify(text: str, max_length: int = 60) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "book"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_book_id(store: Store, path: Path) -> str:
    base = slugify(path.stem)
    candidate, suffix = base, 2
    while store.get_book(candidate) is not None:
        candidate, suffix = f"{base}-{suffix}", suffix + 1
    return candidate


def find_pdfs(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"
    )


def embedding_input(title: str, chunk: Chunk) -> str:
    heading = " — ".join(filter(None, [title, chunk.chapter, chunk.section]))
    return f"{heading}\n\n{chunk.text}"


def ingest_folder(
    folder: Path,
    config: Config,
    store: Store,
    embedder: Embedder | None,
    *,
    rebuild: bool = False,
    prune: bool = False,
    progress: Callable[[str], None] = logger.info,
) -> IngestReport:
    if not folder.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {folder}")

    report = IngestReport()
    if rebuild:
        store.reset()

    use_vectors = embedder is not None and store.vectors_available
    if use_vectors:
        store.configure_vectors(embedder.name, embedder.dimension)
    elif embedder is not None:
        progress("sqlite-vec unavailable: indexing for keyword search only")
    count_tokens = embedder.count_tokens if embedder is not None else approximate_tokens

    for path in find_pdfs(folder):
        resolved = str(path.resolve())
        try:
            sha = file_sha256(path)
            existing = store.find_book_by_path(resolved)
            if existing is not None and existing.sha256 == sha:
                report.skipped.append(existing.id)
                continue

            started = time.monotonic()
            progress(f"Ingesting {path.name}")
            book = extract_book(path, config.ingest)
            chunks = chunk_book(book.pages, book.toc, config.chunking, count_tokens)
            embeddings = None
            if use_vectors and chunks:
                embeddings = embedder.embed_documents(
                    [embedding_input(book.metadata.title, chunk) for chunk in chunks]
                )

            book_id = existing.id if existing is not None else unique_book_id(store, path)
            record = BookRecord(
                id=book_id,
                title=book.metadata.title,
                author=book.metadata.author,
                file_path=resolved,
                sha256=sha,
                page_count=book.metadata.page_count,
            )
            store.replace_book(record, book, chunks, embeddings)
            (report.updated if existing is not None else report.added).append(book_id)

            ocr_pages = sum(page.ocr for page in book.pages)
            progress(
                f"  {book_id}: {book.metadata.page_count} pages ({ocr_pages} OCR), "
                f"{len(book.toc)} TOC entries, {len(book.tables)} tables, {len(chunks)} chunks "
                f"in {time.monotonic() - started:.1f}s"
            )
        except Exception as error:
            logger.exception("Failed to ingest %s", path)
            report.failed.append((str(path), str(error)))

    if prune:
        for book in store.list_books():
            if not Path(book.file_path).exists():
                store.delete_book(book.id)
                report.removed.append(book.id)
    return report
