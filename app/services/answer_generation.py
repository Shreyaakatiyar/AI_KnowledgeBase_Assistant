from app.core.logging_config import get_logger
from app.services.generation import generate_structured
from app.services.vector_store import RetrievedChunk
from app.models.schemas import AnswerResponse

logger = get_logger(__name__)

_ANSWER_SYSTEM_PROMPT = """You are an internal knowledge base assistant for Acme Corp.
Answer the user's question using ONLY the provided context below. Do not use
any outside knowledge, even if you happen to know the answer generally.

Rules:
- If the context does not contain enough information to answer confidently,
  set answer_found to false and explain what's missing in the answer field -
  do NOT guess or make up an answer.
- Cite every source chunk you actually used, with its exact source filename
  and page number as given in the context.
- Set confidence to "high" only if the answer is explicitly and completely
  supported by the context. Use "medium" if partially supported, "low" if
  significant inference was required."""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Chunk {i}] Source: {chunk.source} (page {chunk.page})\n{chunk.text}")
    return "\n\n".join(blocks)


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> AnswerResponse:
    if not chunks:
        logger.warning("generate_answer called with zero context chunks.")
        return AnswerResponse(
            answer="I don't have any relevant information in the knowledge base to answer this question.",
            answer_found=False,
            confidence="low",
            sources=[],
        )

    context = _build_context(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    answer = generate_structured(
        prompt=prompt,
        response_schema=AnswerResponse,
        system_instruction=_ANSWER_SYSTEM_PROMPT,
    )
    logger.info(
        f"Generated answer (found={answer.answer_found}, "
        f"confidence={answer.confidence}, sources={len(answer.sources)})"
    )
    return answer