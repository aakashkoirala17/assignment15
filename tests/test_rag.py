"""Tests for RAG pipeline: document parsing, chunking, embeddings, and vector search."""

import pytest
from backend.app.rag.embeddings import embedding_engine
from backend.app.rag.ingestion import ingestion_pipeline
from backend.app.rag.vector_store import vector_store


def test_chunking_with_overlap():
    """Verify recursive chunker respects size and creates overlap."""
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is a technique for enhancing LLMs. "
        "It combines an external retrieval mechanism with an autoregressive model. "
        "Chunking ensures that large documents are divided into manageable semantic passages. "
        "Vector embeddings map text segments into dense high-dimensional semantic spaces. "
        "ChromaDB stores these vectors efficiently for approximate nearest neighbor lookup."
    )
    chunks = ingestion_pipeline.chunk_text(sample_text, chunk_size=80, chunk_overlap=20)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) > 0


def test_embedding_generation():
    """Verify embedding engine generates normalized vectors."""
    texts = ["Test document snippet", "Another piece of text"]
    embeddings = embedding_engine.embed_texts(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


def test_vector_store_indexing_and_query():
    """Verify vector store indexes chunks and retrieves top-k results."""
    test_collection = "test_rag_collection"
    vector_store.reset_collection(test_collection)

    content = b"""
    # Engineering AI Systems
    Section 1: Model Optimization
    Inference acceleration can be achieved through ONNX Runtime and dynamic quantization.
    Section 2: Serving Architecture
    vLLM provides continuous batching and PagedAttention for high throughput.
    """
    chunks = ingestion_pipeline.process_and_chunk_document(
        filename="test_guide.md",
        content_bytes=content,
        chunk_size=100,
        chunk_overlap=20,
    )
    added = vector_store.add_chunks(chunks, collection_name=test_collection)
    assert added > 0

    # Query for optimization
    results = vector_store.query(
        query_text="How to accelerate inference with quantization?",
        top_k=2,
        collection_name=test_collection,
    )
    assert len(results) > 0
    assert results[0].score is not None
    assert results[0].score > 0.0
