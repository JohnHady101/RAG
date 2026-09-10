"""RAG pipeline: build a chunked knowledge base from a PDF file."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    # Works when imported as a package (e.g. `from src.rag_pipeline import ...`)
    from src.doc_loaders import pdf_loader
except ImportError:
    # Works when src/ itself is on sys.path (e.g. `python src/rag_pipeline.py`)
    from doc_loaders import pdf_loader


def create_knowledge_base(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Load a PDF and split it into overlapping text chunks.

    Args:
        pdf_path: Path to the PDF file to chunk.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between
            consecutive chunks (preserves context across boundaries).

    Returns:
        A list of Documents, one per chunk. Each chunk keeps the
        metadata of its source page (e.g. `source`, `page`).
    """
    documents = pdf_loader(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    print(f"split {len(documents)} page(s) into {len(chunks)} chunk(s)")
    return chunks


if __name__ == "__main__":
    import os

    PDF_PATH = os.path.join(os.path.dirname(__file__), "annualreport-2025.pdf")
    chunks = create_knowledge_base(PDF_PATH)

    print(f"total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- chunk {i} ({len(chunk.page_content)} chars, metadata={chunk.metadata}) ---")
        print(chunk.page_content[:300])
