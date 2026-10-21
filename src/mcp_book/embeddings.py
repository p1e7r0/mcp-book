from __future__ import annotations

import hashlib
import logging
import re
import threading
import unicodedata
from typing import Protocol

import numpy as np

from mcp_book.chunking import approximate_tokens
from mcp_book.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def count_tokens(self, text: str) -> int: ...


def default_prefixes(model: str) -> tuple[str, str]:
    if "e5" in model.lower():
        return "query: ", "passage: "
    return "", ""


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        query_prefix, passage_prefix = default_prefixes(config.model)
        self.query_prefix = config.query_prefix or query_prefix
        self.passage_prefix = config.passage_prefix or passage_prefix
        self._model = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.model

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", self.config.model)
                self._model = SentenceTransformer(self.config.model, device=self.config.device)
            return self._model

    @property
    def dimension(self) -> int:
        model = self.model
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())
        return int(model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])


class HashEmbedder:
    """Deterministic bag-of-words vectors. Only meant for tests and offline smoke runs."""

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"hash-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _features(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()
        words = re.findall(r"\w{3,}", text)
        return words + [word[:5] for word in words if len(word) > 5]

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest, "little") % self._dimension] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts]) if texts else np.zeros((0, 0))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        return approximate_tokens(text)


def create_embedder(config: EmbeddingConfig) -> Embedder | None:
    if config.backend == "none":
        return None
    if config.backend == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(config)
