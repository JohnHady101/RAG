"""Ingestion: PDF -> chunks -> embeddings -> pgvector."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))


from chunking import create_knowledge_base
from config import CHUNK_OVERLAP, CHUNK_SIZE
from db import init_db
from vector_store import add_documents


def ingest_pdf(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    limit: int | None = None,
) -> list[int]:
    """Chunk a PDF and store every chunk in the vector store.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        limit: Ingest at most this many chunks (None = all).

    Returns:
        The inserted row ids.
    """
    init_db()
    chunks = create_knowledge_base(pdf_path, chunk_size, chunk_overlap)
    if limit is not None:
        chunks = chunks[:limit]
    return add_documents(chunks)
