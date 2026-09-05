"""Embeddings Engine supporting PyTorch, ONNX Runtime, and fallback generators."""

import hashlib
import logging
from typing import List
import numpy as np
from backend.app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Unified embedding provider with local transformer and ONNX runtime support."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.embedding_dim = 384
        self._model = None
        self._model_loaded = False
        self._onnx_session = None
        self._onnx_tokenizer = None

    def _ensure_model_loaded(self):
        """Lazy load SentenceTransformer with local_files_only preference."""
        if self._model_loaded:
            return
        self._model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer

            # Try loading cached local files first without network delay
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
                logger.info("Loaded cached local SentenceTransformer: %s", self.model_name)
                return
            except Exception:
                pass

            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded SentenceTransformer model: %s", self.model_name)
        except Exception as e:
            logger.warning(
                "Could not load SentenceTransformer directly (%s). Using deterministic embedding engine.",
                e,
            )
            self._model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute normalized vector embeddings for a list of text strings."""
        if not texts:
            return []

        # 1. Use ONNX Runtime if initialized
        if self._onnx_session and self._onnx_tokenizer:
            try:
                return self._embed_onnx(texts)
            except Exception as e:
                logger.warning("ONNX embedding failed (%s), falling back", e)

        # 2. Use SentenceTransformer if loaded
        self._ensure_model_loaded()
        if self._model is not None:
            try:
                embeddings = self._model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                logger.warning("Transformer encode failed (%s), falling back", e)

        # 3. High-fidelity deterministic fallback
        return [self._deterministic_hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]

    def _deterministic_hash_vector(self, text: str) -> List[float]:
        """Generates a reproducible 384-dimensional unit vector using n-gram hashing."""
        dim = self.embedding_dim
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec.tolist()

        for word in words:
            # Word-level hash
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign

            # Character bigram hashes
            for i in range(len(word) - 1):
                bg = word[i : i + 2]
                h_bg = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16)
                idx_bg = h_bg % dim
                vec[idx_bg] += 0.5 * (1.0 if (h_bg >> 8) % 2 == 0 else -1.0)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def set_onnx_session(self, session, tokenizer):
        """Set an active ONNX Runtime session for accelerated inference."""
        self._onnx_session = session
        self._onnx_tokenizer = tokenizer
        logger.info("Activated ONNX Runtime embedding engine")

    def _embed_onnx(self, texts: List[str]) -> List[List[float]]:
        """Inference using ONNX Runtime with mean pooling and L2 normalization."""
        encoded = self._onnx_tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="np"
        )
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in encoded:
            inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

        outputs = self._onnx_session.run(None, inputs)
        token_embeddings = outputs[0]  # shape: (batch_size, seq_len, hidden_dim)

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = np.expand_dims(attention_mask, -1).repeat(
            token_embeddings.shape[-1], -1
        )
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        # Normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = pooled / norms
        return normalized.tolist()


embedding_engine = EmbeddingEngine()
