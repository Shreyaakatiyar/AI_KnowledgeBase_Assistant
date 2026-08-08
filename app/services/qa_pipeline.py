from app.core.logging_config import get_logger
from app.services.guardrails import check_input, check_output
from app.services.retrieval import retrieve
from app.services.answer_generation import generate_answer
from app.models.schemas import AnswerResponse

logger = get_logger(__name__)


def answer_question(query: str) -> AnswerResponse:
    input_check = check_input(query)
    if not input_check.passed:
        logger.info(f"Query rejected by input guardrail: {input_check.reason}")
        return AnswerResponse(
            answer=f"I can't help with that request. {input_check.reason}",
            answer_found=False,
            confidence="high",
            sources=[],
        )

    chunks = retrieve(query)
    answer = generate_answer(query, chunks)

    available_chunk_keys = {(c.source, c.page) for c in chunks}
    output_check = check_output(answer, available_chunk_keys)
    if not output_check.passed:
        logger.info(f"Answer flagged by output guardrail: {output_check.reason}")

    return answer