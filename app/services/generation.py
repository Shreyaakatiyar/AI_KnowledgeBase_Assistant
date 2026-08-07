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


def generate(
    prompt: str,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
) -> str:
    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type=response_mime_type,
            )
            response = client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config=config,
            )
            if not response.text:
                raise LLMGenerationError("Model returned an empty response.")
            return response.text
        except Exception as e:
            last_error = e
            logger.warning(f"Generation attempt {attempt}/{_MAX_RETRIES} failed: {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)

    raise LLMGenerationError(f"Generation failed after {_MAX_RETRIES} attempts: {last_error}")