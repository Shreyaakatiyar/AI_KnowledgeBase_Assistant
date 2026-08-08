import time
from fastapi import APIRouter, HTTPException

from app.core.logging_config import get_logger
from app.core.exceptions import KnowledgeBaseError
from app.core.request_context import get_request_id
from app.services.qa_pipeline import answer_question
from app.services.vector_store import get_vector_store
from app.api.schemas import QuestionRequest, QuestionResponse, HealthResponse

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    store = get_vector_store()
    return HealthResponse(status="ok", documents_indexed=store.count())


@router.post("/ask", response_model=QuestionResponse, tags=["Q&A"])
def ask_question(request: QuestionRequest) -> QuestionResponse:
    start = time.perf_counter()
    try:
        result = answer_question(request.query)
    except KnowledgeBaseError as e:
        logger.error(f"Pipeline error answering question: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "message": "The assistant is temporarily unavailable. Please try again.",
                "request_id": get_request_id(),
            },
        )

    latency_ms = (time.perf_counter() - start) * 1000
    return QuestionResponse.from_answer(query=request.query, answer=result, latency_ms=latency_ms)