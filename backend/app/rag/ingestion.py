"""Document parsing, ingestion, and recursive chunking pipeline."""

import io
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional
from pypdf import PdfReader
from backend.app.config import settings
from backend.app.schemas.rag import DocumentChunk

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Handles parsing and chunking of heterogeneous documents."""

    def __init__(
        self,
        default_chunk_size: int = settings.RAG_DEFAULT_CHUNK_SIZE,
        default_chunk_overlap: int = settings.RAG_DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = default_chunk_size
        self.chunk_overlap = default_chunk_overlap

    def parse_file(
        self, filename: str, content_bytes: bytes
    ) -> List[Dict[str, Any]]:
        """Parses file contents into text segments with metadata."""
        ext = filename.lower().split(".")[-1]
        pages_or_sections: List[Dict[str, Any]] = []

        if ext == "pdf":
            reader = PdfReader(io.BytesIO(content_bytes))
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_or_sections.append(
                        {
                            "text": text,
                            "metadata": {
                                "filename": filename,
                                "page_number": page_idx + 1,
                                "source": filename,
                            },
                        }
                    )
        elif ext in ["json"]:
            try:
                data = json.loads(content_bytes.decode("utf-8", errors="ignore"))
                pretty_text = json.dumps(data, indent=2)
                pages_or_sections.append(
                    {
                        "text": pretty_text,
                        "metadata": {
                            "filename": filename,
                            "format": "json",
                            "source": filename,
                        },
                    }
                )
            except Exception:
                text = content_bytes.decode("utf-8", errors="ignore")
                pages_or_sections.append(
                    {"text": text, "metadata": {"filename": filename, "source": filename}}
                )
        elif ext in ["txt", "md", "csv", "html", "py", "sh"]:
            text = content_bytes.decode("utf-8", errors="ignore")
            pages_or_sections.append(
                {"text": text, "metadata": {"filename": filename, "source": filename}}
            )
        else:
            text = content_bytes.decode("utf-8", errors="ignore")
            pages_or_sections.append(
                {"text": text, "metadata": {"filename": filename, "source": filename}}
            )

        return pages_or_sections

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Recursively splits text into chunks of target size with specified overlap.
        Splits first on paragraphs, then lines, then sentences, then words.
        """
        c_size = chunk_size or self.chunk_size
        c_overlap = chunk_overlap or self.chunk_overlap
        if c_overlap >= c_size:
            c_overlap = max(0, c_size // 4)

        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

        def _split(text_segment: str, seps: List[str]) -> List[str]:
            if len(text_segment) <= c_size or not seps:
                return [text_segment] if text_segment.strip() else []

            sep = seps[0]
            parts = text_segment.split(sep)
            chunks: List[str] = []
            current = ""

            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) <= c_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    if len(part) > c_size:
                        # Sub-split using next separator
                        sub_chunks = _split(part, seps[1:])
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        current = part

            if current.strip():
                chunks.append(current.strip())

            return chunks

        raw_chunks = _split(text, separators)

        # Apply overlap blending across consecutive chunks
        if c_overlap == 0 or len(raw_chunks) <= 1:
            return raw_chunks

        blended_chunks = []
        for i, chunk in enumerate(raw_chunks):
            if i == 0:
                blended_chunks.append(chunk)
            else:
                prev = raw_chunks[i - 1]
                overlap_prefix = prev[-c_overlap:] if len(prev) > c_overlap else prev
                blended_chunks.append(f"...{overlap_prefix} {chunk}")

        return blended_chunks

    def process_and_chunk_document(
        self,
        filename: str,
        content_bytes: bytes,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """Parses document bytes and generates vectorized DocumentChunk objects."""
        sections = self.parse_file(filename, content_bytes)
        all_chunks: List[DocumentChunk] = []

        chunk_counter = 0
        for section in sections:
            text = section["text"]
            base_meta = section["metadata"]
            chunks = self.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            for c in chunks:
                if not c.strip():
                    continue
                meta = dict(base_meta)
                meta["chunk_index"] = chunk_counter
                meta["char_length"] = len(c)
                chunk_id = f"{filename}_{chunk_counter}_{uuid.uuid4().hex[:6]}"

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        content=c,
                        metadata=meta,
                    )
                )
                chunk_counter += 1

        return all_chunks


ingestion_pipeline = DocumentIngestionPipeline()
