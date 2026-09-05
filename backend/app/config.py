"""Application configuration and environment settings."""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for the AI Assistant."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # Project metadata
    PROJECT_NAME: str = "Enterprise AI Assistant"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # LLM Providers Configuration
    # Options: "gemini", "openai", "anthropic", "vllm", "mock"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    FALLBACK_PROVIDERS: List[str] = ["vllm", "mock"]

    # Provider API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY", "")

    # Local vLLM Configuration
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

    # Default LLM Hyperparameters
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.95
    DEFAULT_MAX_TOKENS: int = 1024

    # RAG & Vector Database Settings
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    RAG_DEFAULT_CHUNK_SIZE: int = 500
    RAG_DEFAULT_CHUNK_OVERLAP: int = 100
    RAG_DEFAULT_TOP_K: int = 3
    RAG_SCORE_THRESHOLD: float = 0.35

    # Reliability Settings
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_FACTOR: float = 1.5
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIME: int = 30  # seconds

    # Rate Limiting (Token Bucket / Requests per minute)
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10

    # Caching Configuration
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL", None)

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

settings = Settings()
