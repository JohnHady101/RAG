"""Unit tests for chunking (no database or API access needed)."""

import os
import tempfile

import fitz

from src.chunking import create_knowledge_base


def _make_pdf(text: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        path = tmp.name
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    doc.save(path)
    doc.close()
    return path


def test_create_knowledge_base_chunks_and_keeps_metadata():
    path = _make_pdf("Hello chunking! " * 100)
    try:
        chunks = create_knowledge_base(path, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 100 for c in chunks)
        assert all(c.page_content.strip() for c in chunks)
        assert all("page" in c.metadata for c in chunks)
    finally:
        os.remove(path)


def test_create_knowledge_base_short_doc_stays_whole():
    path = _make_pdf("Hello CI/CD Pipeline!")
    try:
        chunks = create_knowledge_base(path)
        assert len(chunks) == 1
        assert "Hello CI/CD" in chunks[0].page_content
    finally:
        os.remove(path)
