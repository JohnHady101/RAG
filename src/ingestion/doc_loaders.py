import os
import tempfile

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
)


def load_text_file() -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file.")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            # Fixed: print the specific doc, not the whole list
            print(f"Loaded document: {doc.page_content}") 
    finally:
        os.remove(temp_file_path)

def pdf_loader(pdf_path: str) -> list:
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    print(f"loaded {len(documents)} document(s) from PDF")

    return documents

if __name__ == "__main__":
    # the path of a file outside the parent dir of current python file
    pdf_loader("src/annualreport-2025.pdf")