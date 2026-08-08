import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging_config import setup_logging, get_logger
from app.core.exceptions import KnowledgeBaseError, VectorStoreError
from app.core.request_context import set_request_id, get_request_id
from app.services.vector_store import get_vector_store
from app.services.keyword_search import get_bm25_index
from app.api.routes import router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting up: warming vector store and BM25 index...")
    try:
        store = get_vector_store()
        get_bm25_index()
        logger.info(f"Startup complete. {store.count()} chunks indexed and ready.")
    except VectorStoreError as e:
        logger.error(f"Startup failed to initialize the vector store: {e}")
        raise
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Knowledge Base Assistant",
    description="Enterprise knowledge base assistant with advanced RAG: "
                 "query transformation, hybrid search, reranking, and guardrails.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    start = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(f"{request.method} {request.url.path} raised an unhandled exception after {duration_ms:.1f}ms")
        raise


@app.exception_handler(KnowledgeBaseError)
async def knowledge_base_error_handler(request: Request, exc: KnowledgeBaseError) -> JSONResponse:
    logger.error(f"Unhandled pipeline error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred while processing your request.",
            "request_id": get_request_id(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("An unexpected error occurred.")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred.", "request_id": get_request_id()},
    )


app.include_router(router)


@app.get("/", tags=["System"])
def root():
    return {"service": "AI Knowledge Base Assistant", "docs": "/docs"}