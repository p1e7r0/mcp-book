from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any

from mcp_book.citation import format_citation
from mcp_book.config import Config
from mcp_book.embeddings import Embedder, create_embedder
from mcp_book.search import hybrid_search
from mcp_book.store import BookRecord, Store

MAX_PAGES_PER_READ = 10
MAX_TOP_K = 50
MAX_CONTEXT = 5

_FROM_CONFIG: Any = object()


class LibraryError(ValueError):
    """An error caused by the caller's arguments, safe to show to the assistant."""


class Library:
    """Read operations behind the MCP tools; every result carries a citation."""

    def __init__(
        self, config: Config, store: Store | None = None, embedder: Embedder | None = _FROM_CONFIG
    ) -> None:
        self.config = config
        self.store = store or Store(config.database_path)
        self.embedder = create_embedder(config.embedding) if embedder is _FROM_CONFIG else embedder

    def close(self) -> None:
        self.store.close()

    def _book(self, book_id: str) -> BookRecord:
        book = self.store.get_book(book_id)
        if book is None:
            raise LibraryError(f"Unknown book_id {book_id!r}. Call list_books for valid ids.")
        return book

    def _chapters(self, book_id: str) -> list[sqlite3.Row]:
        return [row for row in self.store.toc(book_id) if row["level"] == 1]

    @staticmethod
    def _chapter_at(chapters: list[sqlite3.Row], page: int) -> str | None:
        current = None
        for row in chapters:
            if row["page"] > page:
                break
            current = row["title"]
        return current

    def _page_refs(self, book_id: str, pages: Sequence[int]) -> dict[int, str]:
        labels = self.store.page_labels(book_id, list(dict.fromkeys(pages)))
        return {page: labels.get(page) or str(page) for page in pages}

    def _chunk_dict(self, book: BookRecord, row: sqlite3.Row) -> dict[str, Any]:
        refs = self._page_refs(book.id, [row["page_start"], row["page_end"]])
        return {
            "chunk_id": row["id"],
            "chapter": row["chapter"],
            "section": row["section"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "page_label": refs[row["page_start"]],
            "citation": format_citation(
                book.title, row["chapter"], refs[row["page_start"]], refs[row["page_end"]]
            ),
        }

    def list_books(self) -> dict[str, Any]:
        books = [
            {
                "book_id": book.id,
                "title": book.title,
                "author": book.author,
                "pages": book.page_count,
                "has_toc": bool(self.store.toc(book.id)),
                "citation": format_citation(book.title),
            }
            for book in self.store.list_books()
        ]
        result: dict[str, Any] = {"count": len(books), "books": books}
        if not books:
            result["note"] = "The library is empty: run `mcp-book ingest <folder>` first."
        return result

    def get_toc(self, book_id: str) -> dict[str, Any]:
        book = self._book(book_id)
        rows = self.store.toc(book_id)
        refs = self._page_refs(book_id, [row["page"] for row in rows])
        chapter = None
        entries = []
        for row in rows:
            if row["level"] == 1:
                chapter = row["title"]
            entries.append(
                {
                    "level": row["level"],
                    "title": row["title"],
                    "page": row["page"],
                    "page_label": refs[row["page"]],
                    "citation": format_citation(book.title, chapter, refs[row["page"]]),
                }
            )
        result: dict[str, Any] = {
            "book_id": book.id,
            "title": book.title,
            "author": book.author,
            "pages": book.page_count,
            "citation": format_citation(book.title),
            "entries": entries,
        }
        if not entries:
            result["note"] = "This PDF has no outline: use search or read_pages."
        return result

    def search(
        self, query: str, book_ids: Sequence[str] | None = None, top_k: int | None = None
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise LibraryError("The query is empty.")
        top_k = self.config.search.top_k if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise LibraryError(f"top_k must be between 1 and {MAX_TOP_K}.")
        books = {book_id: self._book(book_id) for book_id in dict.fromkeys(book_ids or [])}

        outcome = hybrid_search(
            self.store,
            self.embedder,
            query,
            book_ids=list(books) or None,
            top_k=top_k,
            config=self.config.search,
        )
        results = []
        for hit in outcome.hits:
            book_id = hit.chunk["book_id"]
            book = books.get(book_id) or self._book(book_id)
            books[book_id] = book
            results.append(
                {
                    **self._chunk_dict(book, hit.chunk),
                    "book_id": book.id,
                    "book_title": book.title,
                    "author": book.author,
                    "excerpt": hit.excerpt,
                    "score": hit.score,
                    "matched_by": hit.matched_by,
                }
            )
        result: dict[str, Any] = {"query": query, "mode": outcome.mode, "results": results}
        if outcome.note:
            result["note"] = outcome.note
        return result

    def read_pages(self, book_id: str, start_page: int, end_page: int) -> dict[str, Any]:
        book = self._book(book_id)
        if not 1 <= start_page <= book.page_count:
            raise LibraryError(f"start_page must be between 1 and {book.page_count}.")
        if end_page < start_page:
            raise LibraryError("end_page must be >= start_page.")
        requested_end = min(end_page, book.page_count)
        end = min(requested_end, start_page + MAX_PAGES_PER_READ - 1)

        chapters = self._chapters(book_id)
        pages = []
        for row in self.store.pages(book_id, start_page, end):
            page_ref = row["label"] or str(row["page"])
            chapter = self._chapter_at(chapters, row["page"])
            pages.append(
                {
                    "page": row["page"],
                    "page_label": page_ref,
                    "chapter": chapter,
                    "ocr": bool(row["ocr"]),
                    "text": row["text"],
                    "citation": format_citation(book.title, chapter, page_ref),
                }
            )
        refs = self._page_refs(book_id, [start_page, end])
        result: dict[str, Any] = {
            "book_id": book.id,
            "title": book.title,
            "start_page": start_page,
            "end_page": end,
            "total_pages": book.page_count,
            "truncated": end < requested_end,
            "citation": format_citation(
                book.title, self._chapter_at(chapters, start_page), refs[start_page], refs[end]
            ),
            "pages": pages,
        }
        if end < requested_end:
            result["next_start_page"] = end + 1
            result["note"] = f"At most {MAX_PAGES_PER_READ} pages per call."
        return result

    def read_chunk(self, chunk_id: int, context: int = 1) -> dict[str, Any]:
        if not 0 <= context <= MAX_CONTEXT:
            raise LibraryError(f"context must be between 0 and {MAX_CONTEXT}.")
        target = self.store.chunks_by_ids([chunk_id]).get(chunk_id)
        if target is None:
            raise LibraryError(
                f"Unknown chunk_id {chunk_id}. Chunk ids change when a book is re-ingested: "
                "search again."
            )
        book = self._book(target["book_id"])
        window = self.store.chunk_window(
            book.id, target["sequence"] - context, target["sequence"] + context
        )
        chunks = [
            {**self._chunk_dict(book, row), "is_target": row["id"] == chunk_id, "text": row["text"]}
            for row in window
        ]
        return {
            "chunk_id": chunk_id,
            "book_id": book.id,
            "title": book.title,
            "citation": self._chunk_dict(book, target)["citation"],
            "chunks": chunks,
        }

    def get_tables(self, book_id: str, page: int) -> dict[str, Any]:
        book = self._book(book_id)
        if not 1 <= page <= book.page_count:
            raise LibraryError(f"page must be between 1 and {book.page_count}.")
        page_ref = self._page_refs(book_id, [page])[page]
        citation = format_citation(
            book.title, self._chapter_at(self._chapters(book_id), page), page_ref
        )
        tables = [
            {"index": row["position"], "markdown": row["markdown"], "citation": citation}
            for row in self.store.tables(book_id, page)
        ]
        result: dict[str, Any] = {
            "book_id": book.id,
            "page": page,
            "page_label": page_ref,
            "citation": citation,
            "tables": tables,
        }
        if not tables:
            result["note"] = "No table detected on this page; read_pages returns its text."
        return result
