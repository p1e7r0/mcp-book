from pathlib import Path

import pytest

from mcp_book.config import SearchConfig
from mcp_book.embeddings import HashEmbedder
from mcp_book.models import BookMetadata, Chunk, ExtractedBook, PageText
from mcp_book.search import build_fts_query, hybrid_search, make_excerpt
from mcp_book.store import BookRecord, Store

TEXTS = {
    "salse": [
        "La salsa di pomodoro cuoce a lungo con basilico.",
        "La maionese è un'emulsione stabile di olio e tuorlo.",
    ],
    "pane": [
        "Il pane richiede farina forte e una lunga lievitazione.",
        "I pomodori secchi arricchiscono la focaccia.",
    ],
}


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "library.sqlite")
    embedder = HashEmbedder()
    store.configure_vectors(embedder.name, embedder.dimension)
    for book_id, texts in TEXTS.items():
        record = BookRecord(book_id, book_id.title(), None, f"/{book_id}.pdf", "sha", 1)
        book = ExtractedBook(
            BookMetadata(record.title, None, 1), [PageText(1, " ".join(texts))], []
        )
        chunks = [Chunk(i, text, 1, 1, None, None, 10) for i, text in enumerate(texts)]
        store.replace_book(record, book, chunks, embedder.embed_documents(texts))
    yield store
    store.close()


def test_fts_query_uses_prefixes_and_drops_stopwords() -> None:
    assert build_fts_query("la salsa di pomodori") == '"sals"* OR "pomodor"*'
    assert build_fts_query("uova a 64 gradi") == '"uova" OR "64" OR "grad"*'
    assert build_fts_query("di la") == '"di" OR "la"'
    assert build_fts_query("?!") is None


def test_fts_query_is_safe_with_operators(store: Store) -> None:
    query = build_fts_query('NOT "olio" AND (burro* NEAR')
    assert query == '"not" OR "olio" OR "burr"* OR "near"'
    assert store.fts_search(query, None, 5)


def test_hybrid_search_ranks_and_reports_sources(store: Store) -> None:
    outcome = hybrid_search(
        store, HashEmbedder(), "pomodori", book_ids=None, top_k=5, config=SearchConfig()
    )

    assert outcome.mode == "hybrid"
    texts = [hit.chunk["text"] for hit in outcome.hits]
    assert set(texts[:2]) == {TEXTS["salse"][0], TEXTS["pane"][1]}
    assert "bm25" in outcome.hits[0].matched_by


def test_book_filter(store: Store) -> None:
    outcome = hybrid_search(
        store, HashEmbedder(), "pomodoro", book_ids=["pane"], top_k=5, config=SearchConfig()
    )
    assert {hit.chunk["book_id"] for hit in outcome.hits} == {"pane"}


def test_keyword_only_when_embeddings_disabled_or_model_differs(store: Store) -> None:
    disabled = hybrid_search(store, None, "maionese", book_ids=None, top_k=3, config=SearchConfig())
    other_model = hybrid_search(
        store, HashEmbedder(dimension=32), "maionese", book_ids=None, top_k=3, config=SearchConfig()
    )

    for outcome in (disabled, other_model):
        assert outcome.mode == "keyword"
        assert outcome.note
        assert outcome.hits[0].chunk["text"] == TEXTS["salse"][1]


def test_excerpt_centres_on_match() -> None:
    text = "parola " * 200 + "besciamella liscia " + "fine " * 200
    excerpt = make_excerpt(text, "besciamella")
    assert "besciamella" in excerpt
    assert excerpt.startswith("…") and excerpt.endswith("…")
    assert len(excerpt) <= 362
    assert make_excerpt("breve testo", "x") == "breve testo"
