import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging_config import setup_logging, get_logger
from app.core.config import get_settings
from app.services.ingestion import ingest_directory
from app.services.vector_store import VectorStore

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


def main():
    docs_dir = Path("data/documents")
    logger.info(f"Building index from documents in {docs_dir}")

    chunks = ingest_directory(docs_dir)

    store = VectorStore()
    store.reset()  
    store.add_chunks(chunks)

    logger.info(f"Index built successfully. Total vectors stored: {store.count()}")


if __name__ == "__main__":
    main()