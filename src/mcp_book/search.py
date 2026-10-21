from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field

from mcp_book.config import SearchConfig
from mcp_book.embeddings import Embedder
from mcp_book.store import Store

logger = logging.getLogger(__name__)

EXCERPT_CHARS = 360

STOPWORDS = frozenset(
    """
    a ad al alla alle allo agli ai anche che chi come con cosa da dal dalla dalle dei del della
    delle dello degli di e ed fra gli i il in la le lo ma ne nei nel nella nelle nello negli non o
    per piu più quale quali quando se si sono su sul sulla sulle tra un una uno è
    an and are as at be by for from how in is it of on or that the to what when which with
    """.split()
)


@dataclass
class Hit:
    chunk: sqlite3.Row
    score: float
    matched_by: list[str] = field(default_factory=list)
    excerpt: str = ""


@dataclass
class SearchOutcome:
    hits: list[Hit]
    mode: str
    note: str | None = None


def query_words(query: str) -> list[str]:
    words = re.findall(r"\w+", query.casefold())
    content = [word for word in words if word not in STOPWORDS]
    return content or words


def stem(word: str) -> str | None:
    """Prefix covering Italian plural/gender variants (pomodori → pomodor). Not a stemmer."""
    if len(word) >= 5 and not word.isdigit():
        return word[:-1] if word[-1] in "aeiouàèéìòù" else word
    return None


def build_fts_query(query: str) -> str | None:
    terms = []
    for word in dict.fromkeys(query_words(query)):
        prefix = stem(word)
        terms.append(f'"{prefix}"*' if prefix else f'"{word}"')
    return " OR ".join(terms) or None


def make_excerpt(text: str, query: str, size: int = EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= size:
        return text
    lowered = text.casefold()
    positions = [
        position
        for word in query_words(query)
        if (position := lowered.find(stem(word) or word)) >= 0
    ]
    center = min(positions, default=0)
    start = max(0, min(center - size // 3, len(text) - size))
    end = start + size
    if start > 0:
        start = text.find(" ", start) + 1 or start
    if end < len(text) and (cut := text.rfind(" ", start, end)) > start:
        end = cut
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def semantic_status(store: Store, embedder: Embedder | None) -> tuple[bool, str | None]:
    if embedder is None:
        return False, "Semantic search disabled in config (embedding.backend = none)."
    if not store.vectors_available:
        return False, "sqlite-vec could not be loaded: keyword search only."
    if not store.has_vector_table:
        return False, "The index has no embeddings: keyword search only."
    if store.embedding_model != embedder.name:
        return False, (
            f"The index was embedded with {store.embedding_model!r}, the config asks for "
            f"{embedder.name!r}: keyword search only until `mcp-book ingest --rebuild`."
        )
    return True, None


def hybrid_search(
    store: Store,
    embedder: Embedder | None,
    query: str,
    *,
    book_ids: Sequence[str] | None,
    top_k: int,
    config: SearchConfig,
) -> SearchOutcome:
    candidates = max(config.candidates, top_k)
    rankings: dict[str, list[int]] = {}

    fts_query = build_fts_query(query)
    if fts_query:
        rankings["bm25"] = [
            chunk_id for chunk_id, _ in store.fts_search(fts_query, book_ids, candidates)
        ]

    use_vectors, note = semantic_status(store, embedder)
    if use_vectors:
        try:
            vector = embedder.embed_query(query)
        except Exception as error:
            logger.exception("Embedding the query failed")
            use_vectors, note = (
                False,
                f"Embedding model unavailable ({error}): keyword search only.",
            )
        else:
            rankings["semantic"] = [
                chunk_id for chunk_id, _ in store.vector_search(vector, book_ids, candidates)
            ]

    scores: dict[int, float] = {}
    matched: dict[int, list[str]] = {}
    for source, ranked in rankings.items():
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (config.rrf_k + rank)
            matched.setdefault(chunk_id, []).append(source)

    best = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    rows = store.chunks_by_ids(best)
    hits = [
        Hit(
            chunk=rows[chunk_id],
            score=round(scores[chunk_id], 6),
            matched_by=matched[chunk_id],
            excerpt=make_excerpt(rows[chunk_id]["text"], query),
        )
        for chunk_id in best
        if chunk_id in rows
    ]
    mode = "hybrid" if use_vectors else "keyword"
    return SearchOutcome(hits=hits, mode=mode, note=note)
