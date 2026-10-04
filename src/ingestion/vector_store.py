"""Persistence and similarity search over the pgvector store."""

import json

import numpy as np

from db import _load_sql, get_cursor
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

    # cur = get_cursor()
    # cur.execute(
    #     """
    #     SELECT id, content, metadata,
    #            1 - (embedding <=> %s::vector) AS similarity
    #     FROM documents
    #     ORDER BY embedding <=> %s::vector
    #     LIMIT %s;
    #     """,
    #     (q_emb, q_emb, top_k),
    # )
    # return cur.fetchall()
    return bm25_search(query, top_k=top_k)


def bm25_search(query: str, top_k: int = 5):
    """Return the top_k rows matching query via full-text search.

    Runs src/sql/bm25_search.sql (ts_rank over content_tsv with
    websearch_to_tsquery). Postgres approximates BM25-style ranking
    here via ts_rank; for true Okapi BM25 use the pg_bm25 extension.

    Each row is (id, content, metadata, score) with score from
    ts_rank (higher = more relevant). Returns [] when nothing matches.
    """
    if not query or not query.strip():
        return []
    cur = get_cursor()
    cur.execute(_load_sql("bm25_search.sql"), (query, top_k))
    return cur.fetchall()

