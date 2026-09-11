"""Ask: retrieve relevant chunks and synthesize a grounded answer."""

from config import GENERATION_MODEL
from db import init_db
from embeddings import get_client
from vector_store import search


def retrieve(query: str, top_k: int = 5):
    """Return the top_k most similar stored chunks for a query."""
    init_db()
    return search(query, top_k=top_k)


def answer_question(query: str, top_k: int = 5) -> str:
    """Answer a question using retrieved chunks as context.

    Returns the model's answer text. Retrieved context is quoted
    verbatim so the answer stays grounded in the ingested documents.
    """
    rows = retrieve(query, top_k=top_k)
    context = "\n\n---\n\n".join(
        f"[chunk {doc_id} | similarity {similarity:.2f}]\n{content}"
        for doc_id, content, _metadata, similarity in rows
    )
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    response = get_client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text
