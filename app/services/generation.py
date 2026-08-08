import time
from typing import TypeVar
from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)

_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2

T = TypeVar("T", bound=BaseModel)


def _call_with_retries(
    prompt: str,
    system_instruction: str | None,
    response_mime_type: str | None,
    response_schema: type[BaseModel] | None,
):
    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
            )
            return client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            last_error = e
            logger.warning(f"Generation attempt {attempt}/{_MAX_RETRIES} failed: {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)

    raise LLMGenerationError(f"Generation failed after {_MAX_RETRIES} attempts: {last_error}")


def generate(prompt: str, system_instruction: str | None = None) -> str:
    """Generates free-form text. Used for query expansion and other internal steps."""
    response = _call_with_retries(prompt, system_instruction, None, None)
    if not response.text:
        raise LLMGenerationError("Model returned an empty response.")
    return response.text


def generate_structured(
    prompt: str,
    response_schema: type[T],
    system_instruction: str | None = None,
) -> T:
    response = _call_with_retries(
        prompt, system_instruction, "application/json", response_schema
    )
    if response.parsed is None:
        raise LLMGenerationError(
            f"Model response did not match schema {response_schema.__name__}: {response.text!r}"
        )
    return response.parsed