from app.core.logging_config import get_logger
from app.core.exceptions import LLMGenerationError
from app.services.generation import generate

logger = get_logger(__name__)

_EXPANSION_SYSTEM_PROMPT = """You are a query expansion assistant for a company knowledge base search system.
Given a user's question, generate 3 alternative phrasings that capture the same
underlying information need, using different words, synonyms, or angles.

Rules:
- Output EXACTLY 3 lines, one query per line.
- Do NOT number them or add bullets, quotes, or explanations.
- Do NOT answer the question - only rephrase it.
- Keep each query concise (under 20 words)."""


def expand_query(original_query: str, num_variants: int = 3) -> list[str]:
    try:
        raw = generate(
            prompt=f"User question: {original_query}",
            system_instruction=_EXPANSION_SYSTEM_PROMPT,
        )
        variants = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        variants = variants[:num_variants]

        if not variants:
            logger.warning("Query expansion returned no usable variants; using original only.")
            return [original_query]

        logger.info(f"Expanded query into {len(variants)} variants: {variants}")
        return [original_query] + variants

    except LLMGenerationError as e:
        logger.warning(f"Query expansion failed, falling back to original query only: {e}")
        return [original_query]