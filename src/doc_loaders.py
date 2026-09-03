import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyMuPDFLoader
)
from dotenv import load_dotenv

load_dotenv()

print("Loading environment variables from .env file...")

def load_text_file():
    # create a temporary text file with some content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file.")
        temp_file_path = temp_file.name

    # load the text file using the TextLoader
    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print(documents)
            print(f"Loaded document: {doc.page_content}")
    finally:
        # clean up the temporary file
        os.remove(temp_file_path)

def pdf_loader(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    print(f"loaded {len(documents)} document(s) from PDF")

    for i, doc in enumerate(documents):
        print(f"document {i+1} content preview: {doc.page_content[:10]}")
        print(f"metadata: {doc.metadata}")

if __name__ == "__main__":
    print(os.getcwd())
    pdf_loader("annualreport-2025.pdf")