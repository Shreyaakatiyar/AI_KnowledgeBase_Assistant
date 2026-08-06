from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    llm_model: str = "gemini-3.5-flash"
    embedding_model: str = "models/text-embedding-004"

    vector_store_path: str = "./data/vector_store"
    collection_name: str = "knowledge_base"

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieval: int = 10
    top_k_reranked: int = 4

    environment: str = "development"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()