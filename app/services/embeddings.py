import time
from google import genai
from google.genai import types

from app.core.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)

_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(
                model=settings.embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dimension,
                ),
            )
            return [embedding.values for embedding in result.embeddings]
        except Exception as e:
            last_error = e
            logger.warning(f"Embedding attempt {attempt}/{_MAX_RETRIES} failed: {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)

    raise LLMGenerationError(f"Embedding failed after {_MAX_RETRIES} attempts: {last_error}")


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]