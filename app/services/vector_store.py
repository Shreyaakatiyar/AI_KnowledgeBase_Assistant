import chromadb
from chromadb.config import Settings as ChromaSettings
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging_config import get_logger
from app.services.ingestion import Chunk
from app.services.embeddings import embed_texts, embed_query

settings = get_settings()
logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float          
    chunk_id: str


class VectorStore:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(
                path=settings.vector_store_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize vector store: {e}") from e

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 50) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                embeddings = embed_texts(
                    [c.text for c in batch], task_type="RETRIEVAL_DOCUMENT"
                )
                self.collection.add(
                    ids=[c.chunk_id for c in batch],
                    embeddings=embeddings,
                    documents=[c.text for c in batch],
                    metadatas=[c.metadata for c in batch],
                )
                logger.info(f"Indexed batch {i // batch_size + 1}: {len(batch)} chunks")
            except Exception as e:
                raise VectorStoreError(f"Failed to add batch starting at index {i}: {e}") from e

    def similarity_search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k_retrieval
        available = self.count()
        if available == 0:
            logger.warning("similarity_search called on an empty vector store.")
            return []
        effective_k = min(top_k, available)

        try:
            query_embedding = embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_k,
            )
        except Exception as e:
            raise VectorStoreError(f"Similarity search failed: {e}") from e

        retrieved = []
        for doc, meta, dist, chunk_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            similarity_score = 1 - dist
            retrieved.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=meta.get("page", -1),
                score=similarity_score,
                chunk_id=chunk_id,
            ))
        return retrieved

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(settings.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store collection reset.")