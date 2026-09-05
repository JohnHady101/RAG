import os
import tempfile
import fitz  # PyMuPDF
from src.doc_loaders import load_text_file, pdf_loader

def test_load_text_file(capsys):
    """Test that text loader runs without errors and prints output."""
    load_text_file()
    captured = capsys.readouterr()
    assert "Loaded document: This is a sample text file." in captured.out

def test_pdf_loader(capsys):
    """Test PDF loader by generating a dummy PDF on the fly."""
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
        documents = pdf_loader(dummy_pdf_path)
        
        # 3. Assertions
        captured = capsys.readouterr()
        assert "loaded 1 document(s) from PDF" in captured.out
        assert len(documents) == 1
        assert "Hello CI/CD" in documents[0].page_content
    finally:
        # 4. Cleanup
        os.remove(dummy_pdf_path)