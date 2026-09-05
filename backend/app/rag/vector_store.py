"""Vector Store Engine using ChromaDB for persistent and in-memory vector storage."""

import logging
import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.app.config import settings
from backend.app.rag.embeddings import embedding_engine
from backend.app.schemas.rag import DocumentChunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Enterprise vector store managing embeddings, collections, and hybrid retrieval."""

    def __init__(self, persist_directory: str = settings.CHROMA_PERSIST_DIR):
        self.persist_dir = persist_directory
        os.makedirs(self.persist_dir, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            logger.info("ChromaDB PersistentClient initialized at: %s", self.persist_dir)
        except Exception as e:
            logger.warning("Chroma PersistentClient error (%s); using EphemeralClient", e)
            self.client = chromadb.EphemeralClient()

    def get_or_create_collection(self, collection_name: str = "default_knowledge_base"):
        """Gets or initializes a ChromaDB collection with cosine distance metric."""
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        collection_name: str = "default_knowledge_base",
    ) -> int:
        """Embeds text and adds chunks to the vector database."""
        if not chunks:
            return 0

        col = self.get_or_create_collection(collection_name)
        texts = [chunk.content for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = embedding_engine.embed_texts(texts)

        # Chroma requires metadata values to be primitives (str, int, float, bool)
        clean_metadatas = []
        for chunk in chunks:
            clean_meta = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = i + batch_size
            col.upsert(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=clean_metadatas[i:end],
            )

        logger.info(
            "Successfully indexed %d chunks in collection '%s'",
            len(chunks),
            collection_name,
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = settings.RAG_DEFAULT_TOP_K,
        collection_name: str = "default_knowledge_base",
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Performs vector similarity search against the collection."""
        col = self.get_or_create_collection(collection_name)
        count = col.count()
        if count == 0:
            return []

        top_k = min(top_k, count)
        query_vector = embedding_engine.embed_query(query_text)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = col.query(**query_kwargs)

        chunks: List[DocumentChunk] = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                doc_text = results["documents"][0][i]
                doc_meta = results["metadatas"][0][i] if results["metadatas"] else {}
                # Cosine distance to similarity: 1.0 - distance
                dist = (
                    results["distances"][0][i]
                    if "distances" in results and results["distances"]
                    else 0.0
                )
                similarity = max(0.0, 1.0 - dist)

                chunks.append(
                    DocumentChunk(
                        chunk_id=doc_id,
                        content=doc_text,
                        metadata=doc_meta,
                        score=round(similarity, 4),
                    )
                )

        return chunks

    def get_stats(self, collection_name: str = "default_knowledge_base") -> Dict[str, Any]:
        """Returns statistics for a collection."""
        try:
            col = self.get_or_create_collection(collection_name)
            return {
                "collection_name": collection_name,
                "total_chunks": col.count(),
                "persist_directory": self.persist_dir,
            }
        except Exception as e:
            return {"error": str(e)}

    def list_collections(self) -> List[str]:
        """Lists all existing vector collections."""
        try:
            return [c.name for c in self.client.list_collections()]
        except Exception:
            return ["default_knowledge_base"]

    def reset_collection(self, collection_name: str = "default_knowledge_base") -> bool:
        """Deletes and recreates a collection."""
        try:
            self.client.delete_collection(name=collection_name)
            self.get_or_create_collection(collection_name)
            return True
        except Exception as e:
            logger.error("Error resetting collection %s: %s", collection_name, e)
            return False


vector_store = ChromaVectorStore()
