from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging_config import setup_logging, get_logger
from app.core.exceptions import KnowledgeBaseError, VectorStoreError
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


@app.exception_handler(KnowledgeBaseError)
async def knowledge_base_error_handler(request: Request, exc: KnowledgeBaseError) -> JSONResponse:
    logger.error(f"Unhandled pipeline error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred while processing your request."},
    )


app.include_router(router)


@app.get("/", tags=["System"])
def root():
    return {"service": "AI Knowledge Base Assistant", "docs": "/docs"}