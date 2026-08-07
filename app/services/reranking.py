from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.services.vector_store import RetrievedChunk

settings = get_settings()
logger = get_logger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MAX_CHARS_PER_DOC = 2000  
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info(f"Loading reranker model: {_MODEL_NAME} (first call only)")
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
    if not chunks:
        return []

    top_k = top_k or settings.top_k_reranked
    model = _get_model()

    pairs = [[query, chunk.text[:_MAX_CHARS_PER_DOC]] for chunk in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    logger.info(f"Reranked {len(chunks)} candidates down to top {min(top_k, len(reranked))}")
    return reranked[:top_k]