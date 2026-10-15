"""Local PDF + retrieval layer for VBCUA Phase 1.

The RAG layer is deliberately local and dependency-light:
PDF -> text chunks -> SentenceTransformer embeddings -> SQLite -> cosine retrieval.
Gemini is not required for ingestion or retrieval.
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from database import Database
from semantics import load_embedding_model

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5


@dataclass
class RetrievedChunk:
    document_id: int
    document_name: str
    chunk_id: int
    page: int
    text: str
    score: float


def extract_pdf_pages(pdf_bytes: bytes) -> List[tuple[int, str]]:
    """Extract text page-by-page, preserving page numbers for citations."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("pypdf is required for PDF ingestion. Run pip install -r requirements.txt.") from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[tuple[int, str]] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_no, _clean_text(text)))
    return pages


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(
    pages: List[tuple[int, str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[tuple[int, str]]:
    """Create overlapping word-window chunks while retaining source pages."""
    chunk_size = max(200, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    chunks: List[tuple[int, str]] = []

    for page_no, text in pages:
        words = text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append((page_no, chunk))
            if end >= len(words):
                break
            start = end - overlap
    return chunks


def ingest_pdf(
    database: Database,
    pdf_bytes: bytes,
    filename: str,
    embedding_model: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """Index a PDF, replacing an existing document with the same content hash."""
    if not pdf_bytes:
        raise ValueError("The PDF is empty.")
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files can be added to the VBCUA knowledge base.")

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    pages = extract_pdf_pages(pdf_bytes)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No extractable text was found in this PDF. Scanned/image-only PDFs need OCR first.")

    model = load_embedding_model(embedding_model)
    if model is None:
        raise RuntimeError(
            f"The embedding model '{embedding_model}' could not be loaded. "
            "PDF retrieval needs the local embedding model."
        )

    texts = [text for _, text in chunks]
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)

    document_id = database.upsert_document(
        name=filename,
        file_hash=file_hash,
        page_count=len(pages),
        chunk_count=len(chunks),
        embedding_model=embedding_model,
    )
    database.replace_document_chunks(
        document_id,
        [(page, text, vector.tolist()) for (page, text), vector in zip(chunks, vectors)],
    )

    return {
        "document_id": document_id,
        "name": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "hash": file_hash,
    }


def retrieve(
    database: Database,
    query: str,
    embedding_model: str,
    top_k: int = DEFAULT_TOP_K,
    document_id: Optional[int] = None,
) -> List[RetrievedChunk]:
    """Retrieve the most relevant indexed chunks for a query."""
    query = (query or "").strip()
    if not query:
        return []

    model = load_embedding_model(embedding_model)
    if model is None:
        return []

    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    rows = database.fetch_document_chunks(document_id=document_id)
    scored: List[RetrievedChunk] = []
    for row in rows:
        vector = np.asarray(row["embedding"], dtype=np.float32)
        score = float(np.dot(query_vector, vector))
        scored.append(
            RetrievedChunk(
                document_id=int(row["document_id"]),
                document_name=str(row["document_name"]),
                chunk_id=int(row["id"]),
                page=int(row["page"]),
                text=str(row["text"]),
                score=score,
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[: max(1, int(top_k))]


def build_context(results: List[RetrievedChunk]) -> str:
    """Format retrieved evidence for later grounded evaluation prompts."""
    blocks = []
    for item in results:
        blocks.append(
            f"[Source: {item.document_name}, page {item.page}, relevance {item.score:.3f}]\n{item.text}"
        )
    return "\n\n".join(blocks)


__all__ = [
    "RetrievedChunk",
    "extract_pdf_pages",
    "chunk_pages",
    "ingest_pdf",
    "retrieve",
    "build_context",
]
