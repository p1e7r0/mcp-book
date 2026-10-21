from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mcp_book.models import Chunk, ExtractedBook

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    file_path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS toc (
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    level INTEGER NOT NULL,
    title TEXT NOT NULL,
    page INTEGER NOT NULL,
    PRIMARY KEY (book_id, position)
);
CREATE TABLE IF NOT EXISTS pages (
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    label TEXT,
    text TEXT NOT NULL,
    ocr INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (book_id, page)
);
CREATE TABLE IF NOT EXISTS book_tables (
    id INTEGER PRIMARY KEY,
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    position INTEGER NOT NULL,
    rows_json TEXT NOT NULL,
    markdown TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS book_tables_page ON book_tables (book_id, page);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    chapter TEXT,
    section TEXT,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    UNIQUE (book_id, sequence)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, heading, tokenize = 'unicode61 remove_diacritics 2'
);
"""


@dataclass
class BookRecord:
    id: str
    title: str
    author: str | None
    file_path: str
    sha256: str
    page_count: int


class Store:
    """Single SQLite file holding books, pages, tables, chunks, FTS5 and sqlite-vec indexes."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.vectors_available = self._load_sqlite_vec()
        self.connection.executescript(SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)

    def _load_sqlite_vec(self) -> bool:
        try:
            import sqlite_vec

            self.connection.enable_load_extension(True)
            sqlite_vec.load(self.connection)
            self.connection.enable_load_extension(False)
            return True
        except (AttributeError, ImportError, sqlite3.OperationalError) as error:
            logger.warning("sqlite-vec unavailable, semantic search disabled: %s", error)
            return False

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise
            self.connection.execute("COMMIT")

    def query(self, sql: str, parameters: Sequence[object] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.connection.execute(sql, parameters).fetchall()

    def get_meta(self, key: str) -> str | None:
        rows = self.query("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    @property
    def embedding_model(self) -> str | None:
        return self.get_meta("embedding_model")

    @property
    def has_vector_table(self) -> bool:
        rows = self.query("SELECT 1 FROM sqlite_master WHERE name = 'vec_chunks'")
        return bool(rows)

    def configure_vectors(self, model: str, dimension: int) -> None:
        """Create the vector table for a model, or check it matches the existing one."""
        current_model = self.embedding_model
        current_dimension = self.get_meta("embedding_dimension")
        if self.has_vector_table and current_model is not None:
            if current_model != model or int(current_dimension or 0) != dimension:
                raise ValueError(
                    f"The index was built with embedding model {current_model!r} "
                    f"({current_dimension} dims); the config asks for {model!r} ({dimension} "
                    "dims). Run `mcp-book ingest --rebuild` to re-embed every book."
                )
            return
        with self._lock:
            self.connection.execute(
                "CREATE VIRTUAL TABLE vec_chunks USING vec0("
                "chunk_id INTEGER PRIMARY KEY, book_id TEXT PARTITION KEY, "
                f"embedding FLOAT[{int(dimension)}] distance_metric=cosine)"
            )
        self.set_meta("embedding_model", model)
        self.set_meta("embedding_dimension", str(dimension))

    def reset(self) -> None:
        """Drop every book and the vector table (used by --rebuild)."""
        with self.transaction() as connection:
            connection.execute("DELETE FROM chunks_fts")
            connection.execute("DELETE FROM books")
            if self.has_vector_table:
                connection.execute("DROP TABLE vec_chunks")
            connection.execute(
                "DELETE FROM meta WHERE key IN ('embedding_model', 'embedding_dimension')"
            )

    # Books

    def list_books(self) -> list[BookRecord]:
        rows = self.query(
            "SELECT id, title, author, file_path, sha256, page_count FROM books ORDER BY title"
        )
        return [BookRecord(**dict(row)) for row in rows]

    def get_book(self, book_id: str) -> BookRecord | None:
        rows = self.query(
            "SELECT id, title, author, file_path, sha256, page_count FROM books WHERE id = ?",
            (book_id,),
        )
        return BookRecord(**dict(rows[0])) if rows else None

    def find_book_by_path(self, file_path: str) -> BookRecord | None:
        rows = self.query("SELECT id FROM books WHERE file_path = ?", (file_path,))
        return self.get_book(rows[0]["id"]) if rows else None

    def delete_book(self, book_id: str) -> None:
        with self.transaction() as connection:
            self._delete_book(connection, book_id)

    def _delete_book(self, connection: sqlite3.Connection, book_id: str) -> None:
        connection.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE book_id = ?)",
            (book_id,),
        )
        if self.has_vector_table:
            connection.execute("DELETE FROM vec_chunks WHERE book_id = ?", (book_id,))
        connection.execute("DELETE FROM books WHERE id = ?", (book_id,))

    def replace_book(
        self,
        record: BookRecord,
        book: ExtractedBook,
        chunks: list[Chunk],
        embeddings: np.ndarray | None,
    ) -> None:
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("One embedding per chunk is required")
        with self.transaction() as connection:
            self._delete_book(connection, record.id)
            connection.execute(
                "INSERT INTO books (id, title, author, file_path, sha256, page_count, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.title,
                    record.author,
                    record.file_path,
                    record.sha256,
                    record.page_count,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            connection.executemany(
                "INSERT INTO toc (book_id, position, level, title, page) VALUES (?, ?, ?, ?, ?)",
                [
                    (record.id, position, entry.level, entry.title, entry.page)
                    for position, entry in enumerate(book.toc)
                ],
            )
            connection.executemany(
                "INSERT INTO pages (book_id, page, label, text, ocr) VALUES (?, ?, ?, ?, ?)",
                [(record.id, p.page, p.label, p.text, int(p.ocr)) for p in book.pages],
            )
            connection.executemany(
                "INSERT INTO book_tables (book_id, page, position, rows_json, markdown) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (record.id, t.page, t.index, json.dumps(t.rows, ensure_ascii=False), t.markdown)
                    for t in book.tables
                ],
            )
            for index, chunk in enumerate(chunks):
                cursor = connection.execute(
                    "INSERT INTO chunks (book_id, sequence, chapter, section, page_start, "
                    "page_end, text, token_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.id,
                        chunk.sequence,
                        chunk.chapter,
                        chunk.section,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.text,
                        chunk.token_count,
                    ),
                )
                chunk_id = cursor.lastrowid
                heading = " ".join(filter(None, [record.title, chunk.chapter, chunk.section]))
                connection.execute(
                    "INSERT INTO chunks_fts (rowid, text, heading) VALUES (?, ?, ?)",
                    (chunk_id, chunk.text, heading),
                )
                if embeddings is not None:
                    connection.execute(
                        "INSERT INTO vec_chunks (chunk_id, book_id, embedding) VALUES (?, ?, ?)",
                        (chunk_id, record.id, np.asarray(embeddings[index], np.float32).tobytes()),
                    )

    # Reads

    def toc(self, book_id: str) -> list[sqlite3.Row]:
        return self.query(
            "SELECT level, title, page FROM toc WHERE book_id = ? ORDER BY position", (book_id,)
        )

    def pages(self, book_id: str, start: int, end: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT page, label, text, ocr FROM pages WHERE book_id = ? AND page BETWEEN ? AND ? "
            "ORDER BY page",
            (book_id, start, end),
        )

    def page_labels(self, book_id: str, pages: Sequence[int]) -> dict[int, str | None]:
        if not pages:
            return {}
        placeholders = ",".join("?" * len(pages))
        rows = self.query(
            f"SELECT page, label FROM pages WHERE book_id = ? AND page IN ({placeholders})",
            (book_id, *pages),
        )
        return {row["page"]: row["label"] for row in rows}

    def tables(self, book_id: str, page: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT position, rows_json, markdown FROM book_tables WHERE book_id = ? AND page = ? "
            "ORDER BY position",
            (book_id, page),
        )

    def chunks_by_ids(self, chunk_ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.query(f"SELECT * FROM chunks WHERE id IN ({placeholders})", tuple(chunk_ids))
        return {row["id"]: row for row in rows}

    def chunk_window(self, book_id: str, first: int, last: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM chunks WHERE book_id = ? AND sequence BETWEEN ? AND ? ORDER BY sequence",
            (book_id, first, last),
        )

    def fts_search(
        self, match: str, book_ids: Sequence[str] | None, limit: int
    ) -> list[tuple[int, float]]:
        sql = (
            "SELECT chunks_fts.rowid AS id, bm25(chunks_fts, 1.0, 0.5) AS score "
            "FROM chunks_fts JOIN chunks ON chunks.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ?"
        )
        parameters: list[object] = [match]
        if book_ids:
            sql += f" AND chunks.book_id IN ({','.join('?' * len(book_ids))})"
            parameters.extend(book_ids)
        sql += " ORDER BY score LIMIT ?"
        parameters.append(limit)
        return [(row["id"], row["score"]) for row in self.query(sql, parameters)]

    def vector_search(
        self, vector: np.ndarray, book_ids: Sequence[str] | None, limit: int
    ) -> list[tuple[int, float]]:
        if not self.has_vector_table:
            return []
        sql = "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?"
        parameters: list[object] = [np.asarray(vector, np.float32).tobytes(), limit]
        if book_ids:
            sql += f" AND book_id IN ({','.join('?' * len(book_ids))})"
            parameters.extend(book_ids)
        sql += " ORDER BY distance"
        return [(row["chunk_id"], row["distance"]) for row in self.query(sql, parameters)]
