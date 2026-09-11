import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import fitz
from src import ingest
from _pytest.capture import CaptureFixture


def test_ingest_pdf(capsys: CaptureFixture):
    """Test ingesting a dummy PDF into the vector store."""
    # 1. Create a dummy PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Hello CI/CD Pipeline!")
        doc.save(tmp.name)
        doc.close()
        dummy_pdf_path = tmp.name

    try:
        # 2. Run ingestion
        ids = ingest.ingest_pdf(dummy_pdf_path)

        # 3. Assertions
        captured = capsys.readouterr()
        assert len(ids) >= 1
        assert "Inserted document id=" in captured.out
    finally:
        # 4. Cleanup
        os.remove(dummy_pdf_path)

