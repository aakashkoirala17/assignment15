"""RAG ingestion and retrieval API endpoints."""

import logging
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.app.core.llm_provider import llm_service
from backend.app.core.rate_limiter import rate_limit_dependency
from backend.app.rag.ingestion import ingestion_pipeline
from backend.app.rag.vector_store import vector_store
from backend.app.schemas.chat import ChatMessage, ChatRequest
from backend.app.schemas.rag import IngestResponse, RAGQueryRequest, RAGQueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])


@router.post(
    "/upload",
    response_model=IngestResponse,
    summary="Upload, Parse, Chunk and Vectorize Document",
    dependencies=[Depends(rate_limit_dependency)],
)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="default_knowledge_base"),
    chunk_size: Optional[int] = Form(default=None),
    chunk_overlap: Optional[int] = Form(default=None),
) -> IngestResponse:
    """
    Ingests a document (PDF, TXT, MD, CSV, JSON), chunks it using recursive splitting,
    computes embeddings, and indexes them into ChromaDB.
    """
    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
            )

        chunks = ingestion_pipeline.process_and_chunk_document(
            filename=file.filename or "uploaded_doc.txt",
            content_bytes=content_bytes,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        vector_store.add_chunks(chunks, collection_name=collection_name)

        return IngestResponse(
            status="success",
            document_id=f"doc_{int(time.time())}",
            filename=file.filename or "unknown",
            chunks_created=len(chunks),
            total_characters=len(content_bytes),
            collection_name=collection_name,
        )
    except Exception as e:
        logger.error("RAG upload failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}",
        )


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Vector Similarity Search & RAG Answer Synthesis",
    dependencies=[Depends(rate_limit_dependency)],
)
async def query_rag(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    Performs vector similarity search against the vector database and synthesizes
    an augmented answer using retrieved context.
    """
    start_time = time.time()
    try:
        retrieved_chunks = vector_store.query(
            query_text=request.query,
            top_k=request.top_k,
            collection_name=request.collection_name or "default_knowledge_base",
            filter_metadata=request.filter_metadata,
        )

        sources = list(
            {
                chunk.metadata.get("filename", "unknown")
                for chunk in retrieved_chunks
                if "filename" in chunk.metadata
            }
        )

        answer = None
        if request.generate_answer:
            if not retrieved_chunks:
                context_str = "No relevant context found in knowledge base."
            else:
                context_str = "\n\n---\n\n".join(
                    [
                        f"[Source: {c.metadata.get('filename', 'doc')} | Score: {c.score}]\n{c.content}"
                        for c in retrieved_chunks
                    ]
                )

            system_prompt = (
                "You are an expert AI knowledge assistant with Retrieval-Augmented Generation capabilities. "
                "Answer the user's question accurately using ONLY the retrieved context below. "
                "Cite your sources where appropriate. If the context does not contain the answer, "
                "clearly state that the information is not present in the indexed documents.\n\n"
                f"Retrieved Context:\n{context_str}"
            )

            chat_req = ChatRequest(
                messages=[ChatMessage(role="user", content=request.query)],
                system_prompt=system_prompt,
                provider=request.provider,
                temperature=request.temperature,
                tools_enabled=False,
            )
            chat_resp = await llm_service.complete(chat_req)
            answer = chat_resp.content

        latency = (time.time() - start_time) * 1000
        return RAGQueryResponse(
            query=request.query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            sources=sources,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        logger.error("RAG query failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query execution failed: {str(e)}",
        )


@router.get(
    "/collections",
    summary="List Collections and Storage Statistics",
)
async def list_collections():
    """Returns all active collections and chunk counts."""
    collections = vector_store.list_collections()
    stats = [vector_store.get_stats(c) for c in collections]
    return {"collections": collections, "stats": stats}


@router.delete(
    "/collections/{collection_name}",
    summary="Reset or Delete a Collection",
)
async def reset_collection(collection_name: str):
    """Deletes and clears all vectors in the collection."""
    success = vector_store.reset_collection(collection_name)
    return {"status": "reset" if success else "failed", "collection": collection_name}
