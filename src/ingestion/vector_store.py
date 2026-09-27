"""Persistence and similarity search over the pgvector store."""

import json

import numpy as np

from db import get_cursor
from src.models.embeddings import get_embedding


def add_document(text: str, metadata: dict | None = None) -> int:
    """Embed one text, store it, and return its row id."""
    embedding = get_embedding(text)
    cur = get_cursor()
    cur.execute(
        """
        INSERT INTO documents (content, metadata, embedding)
        VALUES (%s, %s, %s) 
        RETURNING id;
        """,
        (text, json.dumps(metadata or {}), embedding),
    )
    doc_id = cur.fetchone()[0]
    cur.connection.commit()
    print(f"Inserted document id={doc_id}")
    return doc_id


def add_documents(chunks) -> list[int]:
    """Store many langchain Documents (page_content + metadata)."""
    return [
        add_document(doc.page_content, metadata=doc.metadata) for doc in chunks
    ]


def search(query: str, top_k: int = 3):
    """Return the top_k rows most similar to the query.

    Each row is (id, content, metadata, similarity) with cosine
    similarity in [0, 1].
    """
    q_emb = get_embedding(query)

    if hasattr(q_emb, "embedding"):  # e.g. OpenAI-style Embedding object
        q_emb = q_emb.embedding
    elif isinstance(q_emb, np.ndarray):
        q_emb = q_emb.tolist()

    cur = get_cursor()
    cur.execute(
        """
        SELECT id, content, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (q_emb, q_emb, top_k),
    )
    return cur.fetchall()

def bm25engine(query: str, top_k: int = 3):
    """Return the top_k rows most similar to the query using BM25.

    Each row is (id, content, metadata, similarity) with BM25
    similarity score.
    """
    cur = get_cursor()
    cur.execute(
        """
        SELECT id, content, metadata,
               ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) AS similarity
        FROM documents
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        ORDER BY similarity DESC
        LIMIT %s;
        """,
        (query, query, top_k),
    )
    return cur.fetchall()