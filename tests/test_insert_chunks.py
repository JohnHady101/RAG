import os
import tempfile

import fitz

from src.insertchunks import insert_documents_from_pdf


def test_insert_documents_from_pdf(capsys):
    """Test the insert_documents_from_pdf function."""
    # 1. Create a dummy PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Hello CI/CD Pipeline!")
        doc.save(tmp.name)
        doc.close()
        dummy_pdf_path = tmp.name

    try:
        # 2. Run your function
        insert_documents_from_pdf(dummy_pdf_path)
        
        # 3. Assertions
        captured = capsys.readouterr()
        assert "✅ Inserted document id=" in captured.out
    finally:
        # 4. Cleanup
        os.remove(dummy_pdf_path)