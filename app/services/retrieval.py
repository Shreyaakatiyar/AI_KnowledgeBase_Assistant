from collections import defaultdict

from app.core.logging_config import get_logger
from app.core.config import get_settings
from app.services.query_transform import expand_query
from app.services.vector_store import VectorStore, RetrievedChunk, get_vector_store
from app.services.keyword_search import BM25Index, get_bm25_index
from app.services.reranking import rerank

logger = get_logger(__name__)
settings = get_settings()

_RRF_K = 60 


def _reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in rank_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] += 1 / (k + rank)
    return scores


def _hybrid_search_single_query(
    query: str,
    store: VectorStore,
    bm25_index: BM25Index,
    top_k: int,
) -> list[RetrievedChunk]:
    candidate_k = max(top_k * 2, 10)  

    vector_results = store.similarity_search(query, top_k=candidate_k)
    keyword_results = bm25_index.search(query, top_k=candidate_k)

    vector_rank_list = [c.chunk_id for c in vector_results]
    keyword_rank_list = [cid for cid, _, _, _ in keyword_results]

    fused_scores = _reciprocal_rank_fusion([vector_rank_list, keyword_rank_list])

    chunk_lookup: dict[str, RetrievedChunk] = {c.chunk_id: c for c in vector_results}
    for cid, _score, doc, meta in keyword_results:
        if cid not in chunk_lookup:
            chunk_lookup[cid] = RetrievedChunk(
                text=doc, source=meta.get("source", "unknown"),
                page=meta.get("page", -1), score=0.0, chunk_id=cid,
            )

    fused = []
    for chunk_id, fused_score in fused_scores.items():
        chunk = chunk_lookup[chunk_id]
        chunk.score = fused_score
        fused.append(chunk)

    fused.sort(key=lambda c: c.score, reverse=True)
    return fused[:top_k]


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
    use_hybrid_search: bool = True,
    use_reranking: bool = True,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    store = get_vector_store()     
    retrieval_k = settings.top_k_retrieval
    queries = expand_query(query) if use_query_expansion else [query]

    if use_hybrid_search:
        bm25_index = get_bm25_index()
        result_lists = [
            _hybrid_search_single_query(q, store, bm25_index, retrieval_k) for q in queries
        ]
    else:
        result_lists = [store.similarity_search(q, top_k=retrieval_k) for q in queries]

    merged = _merge_results(result_lists)

    query_preview = query[:60] + "..." if len(query) > 60 else query
    logger.info(
        f"Retrieved {len(merged)} unique candidates from {len(queries)} query variant(s) "
        f"(hybrid={use_hybrid_search}) for: '{query_preview}'"
    )

    if use_reranking:
        final_k = top_k or settings.top_k_reranked
        return rerank(query, merged, top_k=final_k)

    final_k = top_k or retrieval_k
    return merged[:final_k]