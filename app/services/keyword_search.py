import re
from rank_bm25 import BM25Okapi

from app.core.logging_config import get_logger
from app.services.vector_store import VectorStore
from functools import lru_cache
from app.services.vector_store import VectorStore, get_vector_store  

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:

    def __init__(self, vector_store: VectorStore):
        self._store = vector_store
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._build()

    def _build(self) -> None:
        raw = self._store.collection.get(include=["documents", "metadatas"])
        self._ids = raw["ids"]
        self._documents = raw["documents"]
        self._metadatas = raw["metadatas"]

        if not self._documents:
            logger.warning("BM25Index built on an empty collection.")
            self._bm25 = None
            return

        tokenized_corpus = [_tokenize(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built over {len(self._documents)} chunks.")

    def refresh(self) -> None:
        """Call after adding new chunks to the vector store, to keep BM25 in sync."""
        self._build()

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float, str, dict]]:
        """Returns [(chunk_id, bm25_score, text, metadata), ...], best first."""
        if self._bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self._ids, scores, self._documents, self._metadatas),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(cid, score, doc, meta) for cid, score, doc, meta in ranked[:top_k] if score > 0]

@lru_cache
def get_bm25_index() -> "BM25Index":
    """Process-wide singleton - avoids re-tokenizing the whole corpus on every call."""
    return BM25Index(get_vector_store())