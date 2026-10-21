from pathlib import Path

import numpy as np
import pytest

from mcp_book.embeddings import HashEmbedder, default_prefixes
from mcp_book.models import BookMetadata, Chunk, ExtractedBook, PageText, Table, TocEntry
from mcp_book.store import BookRecord, Store


def make_book(book_id: str, texts: list[str]) -> tuple[BookRecord, ExtractedBook, list[Chunk]]:
    record = BookRecord(book_id, f"Libro {book_id}", "Autore", f"/{book_id}.pdf", "sha", 2)
    book = ExtractedBook(
        metadata=BookMetadata(record.title, record.author, 2),
        pages=[PageText(1, texts[0], "i"), PageText(2, " ".join(texts[1:]), "1")],
        toc=[TocEntry(1, "Capitolo", 1)],
        tables=[Table(2, 0, [["a", "b"], ["1", "2"]], "| a | b |")],
    )
    chunks = [
        Chunk(i, text, 1 + (i > 0), 1 + (i > 0), "Capitolo", None, 5)
        for i, text in enumerate(texts)
    ]
    return record, book, chunks


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "index" / "library.sqlite")
    yield store
    store.close()


def test_replace_book_roundtrip(store: Store) -> None:
    embedder = HashEmbedder()
    store.configure_vectors(embedder.name, embedder.dimension)
    record, book, chunks = make_book("cucina", ["brodo di carne", "salsa di pomodoro", "pane"])

    store.replace_book(record, book, chunks, embedder.embed_documents([c.text for c in chunks]))

    assert [b.id for b in store.list_books()] == ["cucina"]
    assert [row["title"] for row in store.toc("cucina")] == ["Capitolo"]
    assert store.page_labels("cucina", [1, 2]) == {1: "i", 2: "1"}
    assert store.tables("cucina", 2)[0]["markdown"] == "| a | b |"
    [(chunk_id, _)] = store.fts_search('"pomodoro"', None, 5)
    assert store.chunks_by_ids([chunk_id])[chunk_id]["text"] == "salsa di pomodoro"
    nearest = store.vector_search(embedder.embed_query("pomodoro"), None, 3)
    assert nearest[0][0] == chunk_id


def test_reingest_replaces_rows_without_reusing_ids(store: Store) -> None:
    embedder = HashEmbedder()
    store.configure_vectors(embedder.name, embedder.dimension)
    record, book, chunks = make_book("cucina", ["uno", "due"])
    store.replace_book(record, book, chunks, embedder.embed_documents([c.text for c in chunks]))
    old_ids = set(store.chunks_by_ids(range(1, 10)))

    store.replace_book(record, book, chunks, embedder.embed_documents([c.text for c in chunks]))
    new_ids = {row["id"] for row in store.chunk_window("cucina", 0, 10)}

    assert len(new_ids) == 2
    assert not old_ids & new_ids
    assert len(store.query("SELECT * FROM chunks_fts")) == 2
    assert len(store.query("SELECT chunk_id FROM vec_chunks")) == 2


def test_search_filters_by_book(store: Store) -> None:
    embedder = HashEmbedder()
    store.configure_vectors(embedder.name, embedder.dimension)
    for book_id in ("primo", "secondo"):
        record, book, chunks = make_book(book_id, ["risotto allo zafferano", "altro"])
        store.replace_book(record, book, chunks, embedder.embed_documents([c.text for c in chunks]))

    fts = store.fts_search('"risotto"', ["secondo"], 5)
    vectors = store.vector_search(embedder.embed_query("risotto"), ["secondo"], 5)
    rows = store.chunks_by_ids([chunk_id for chunk_id, _ in fts + vectors])

    assert {row["book_id"] for row in rows.values()} == {"secondo"}


def test_delete_book_cleans_indexes(store: Store) -> None:
    embedder = HashEmbedder()
    store.configure_vectors(embedder.name, embedder.dimension)
    record, book, chunks = make_book("cucina", ["uno", "due"])
    store.replace_book(record, book, chunks, embedder.embed_documents([c.text for c in chunks]))

    store.delete_book("cucina")

    assert store.list_books() == []
    for table in ("chunks", "chunks_fts", "vec_chunks", "pages", "toc", "book_tables"):
        assert store.query(f"SELECT count(*) AS n FROM {table}")[0]["n"] == 0


def test_model_change_requires_rebuild(store: Store) -> None:
    store.configure_vectors("modello-a", 4)
    store.configure_vectors("modello-a", 4)
    with pytest.raises(ValueError, match="--rebuild"):
        store.configure_vectors("modello-b", 8)

    store.reset()
    store.configure_vectors("modello-b", 8)
    assert store.embedding_model == "modello-b"


def test_embedding_count_must_match_chunks(store: Store) -> None:
    record, book, chunks = make_book("cucina", ["uno", "due"])
    with pytest.raises(ValueError):
        store.replace_book(record, book, chunks, np.zeros((1, 4), np.float32))
    assert store.list_books() == []


def test_hash_embedder_is_normalized_and_deterministic() -> None:
    embedder = HashEmbedder(dimension=64)
    first, second = embedder.embed_documents(["salsa verde", "salsa verde"])
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_e5_models_get_prefixes() -> None:
    assert default_prefixes("intfloat/multilingual-e5-large") == ("query: ", "passage: ")
    assert default_prefixes("BAAI/bge-m3") == ("", "")
