from app.core.logging_config import get_logger
from app.core.config import get_settings
from app.services.query_transform import expand_query
from app.services.vector_store import VectorStore, RetrievedChunk

logger = get_logger(__name__)
settings = get_settings()


def _merge_results(result_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    best_by_id: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for chunk in results:
            existing = best_by_id.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best_by_id[chunk.chunk_id] = chunk

    merged = sorted(best_by_id.values(), key=lambda c: c.score, reverse=True)
    return merged


def retrieve(
    query: str,
    use_query_expansion: bool = True,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    store = VectorStore()
    top_k = top_k or settings.top_k_retrieval

    queries = expand_query(query) if use_query_expansion else [query]

    result_lists = [store.similarity_search(q, top_k=top_k) for q in queries]
    merged = _merge_results(result_lists)

    query_preview = query[:60] + "..." if len(query) > 60 else query
    logger.info(
        f"Retrieved {len(merged)} unique chunks from {len(queries)} query variant(s) "
        f"for: '{query_preview}'"
    )
    return merged[:top_k]