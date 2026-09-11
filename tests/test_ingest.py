import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

import fitz
from _pytest.capture import CaptureFixture

from src import ingest
<<<<<<< HEAD
=======

>>>>>>> ruff


def test_ingest_pdf(capsys: CaptureFixture, monkeypatch):
    """Test ingesting a dummy PDF without needing a live DB or API.

    Mocks init_db and add_documents so CI stays a pure unit test.
    """
    # 1. Create a dummy PDF (close temp file first for Windows compat)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        dummy_pdf_path = tmp.name
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello CI/CD Pipeline!")
    doc.save(dummy_pdf_path)
    doc.close()

    try:
        # 2. Stub out external side effects (DB + embeddings).
        monkeypatch.setattr(ingest, "init_db", lambda: None)

        def fake_add_documents(chunks):
            ids = list(range(1, len(chunks) + 1))
            for doc_id in ids:
                print(f"Inserted document id={doc_id}")
            return ids

        monkeypatch.setattr(ingest, "add_documents", fake_add_documents)

        # 3. Run ingestion
        ids = ingest.ingest_pdf(dummy_pdf_path)

        # 4. Assertions
        captured = capsys.readouterr()
        assert len(ids) >= 1
        assert "Inserted document id=" in captured.out
    finally:
        # 5. Cleanup
        os.remove(dummy_pdf_path)
