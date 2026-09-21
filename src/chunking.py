"""Split loaded documents into overlapping chunks for retrieval."""

import os
import sys

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))



from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from doc_loaders import pdf_loader


def create_knowledge_base(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Load a PDF and split it into overlapping text chunks.

    Args:
        pdf_path: Path to the PDF file t o chunk.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Overlapping characters between consecutive
            chunks (preserves context across boundaries).

    Returns:
        One Document per chunk. Each chunk keeps its source page's
        metadata (e.g. `source`, `page`).
    """
    documents = pdf_loader(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    print(f"split {len(documents)} page(s) into {len(chunks)} chunk(s)")
    return chunks

if __name__ ==  "__main__":
    # the path of a file outside the parent dir of current python file
    create_knowledge_base("data/annualreport-2025.pdf")