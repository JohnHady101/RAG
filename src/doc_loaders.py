import os
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
)

load_dotenv()

print("Loading environment variables from .env file...")

def load_text_file():
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

def pdf_loader(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    print(f"loaded {len(documents)} document(s) from PDF")

    print(documents[0])

    # for i, doc in enumerate(documents):
        
    #     if (i == 1):
    #         print(dir(doc))
    #     # print(f"document {i+1} content preview: {doc.page_content[:10]}")
    #     # print(f"metadata: {doc.metadata}")
        
    return documents

if __name__ == "__main__":
    # the path of a file outside the parent dir of current python file
    pdf_loader("src/annualreport-2025.pdf")