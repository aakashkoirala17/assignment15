"""RAG (Retrieval-Augmented Generation) schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Chunk of vectorized document content."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = Field(
        default=None, description="Similarity score or distance"
    )


class IngestResponse(BaseModel):
    """Response after document ingestion and vectorization."""

    status: str = "success"
    document_id: str
    filename: str
    chunks_created: int
    total_characters: int
    collection_name: str


class RAGQueryRequest(BaseModel):
    """Request for document retrieval or RAG answer generation."""

    query: str = Field(description="Search question or query")
    top_k: int = Field(default=3, ge=1, le=20)
    collection_name: Optional[str] = "default_knowledge_base"
    filter_metadata: Optional[Dict[str, Any]] = None
    generate_answer: bool = Field(
        default=True, description="Whether to generate synthesized answer using LLM"
    )
    provider: Optional[str] = None
    temperature: float = 0.3


class RAGQueryResponse(BaseModel):
    """Response with retrieved chunks and synthesized answer."""

    query: str
    answer: Optional[str] = None
    retrieved_chunks: List[DocumentChunk]
    sources: List[str]
    latency_ms: float
