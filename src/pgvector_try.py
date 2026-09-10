import json
import os

import numpy as np
import psycopg2
from google import genai
from pgvector.psycopg2 import register_vector

# ── Config ──────────────────────────────────────────────
DB_CONFIG = {
    "host": "pgvector",
    "port": 5432,
    "dbname": "vectordb",
    "user": "myuser",
    "password": "mypassword",
}
EMBEDDING_MODEL = "gemini-embedding-2"  # 3072 dimensions
EMBEDDING_DIM = 3072
API = "API_KEY"

# ── Step 1: Connect & enable pgvector ──────────────────
conn = psycopg2.connect(**DB_CONFIG)
register_vector(conn)  # Register the vector type with psycopg2
cur = conn.cursor()

# 1. Create the extension FIRST
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# 2. THEN register the vector type with psycopg2
register_vector(conn)       

# ── Step 2: Create the table (run once) ────────────────
with open(os.path.dirname(__file__) + "/sql/create_table.sql", "r") as f:
    create_table_sql = f.read()

cur.execute(create_table_sql)
conn.commit()

# ── Step 3: Generate an embedding ──────────────────────
client = genai.Client(api_key= API)

def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents="What is the meaning of life?"
)

    return result.embeddings[0].values

# ── Step 4: Insert the document ────────────────────────
def add_document(text: str, metadata: dict | None = None):
    embedding = get_embedding(text)
    cur.execute(
        """
        INSERT INTO documents (content, metadata, embedding)
        VALUES (%s, %s, %s)
        RETURNING id;
        """,
        (text, json.dumps(metadata or {}), embedding),
    )
    doc_id = cur.fetchone()[0]
    conn.commit()
    print(f"✅ Inserted document id={doc_id}")
    return doc_id

# ── Usage ──────────────────────────────────────────────
# add_document(
#     text="pgvector is a PostgreSQL extension for vector similarity search.",
#     metadata={"source": "docs", "author": "alice"},
# )

# ── Bonus: Similarity search ───────────────────────────
def search(query: str, top_k: int = 3):
    q_emb = get_embedding(query)

     # Ensure q_emb is a list of floats or a string representation
    if hasattr(q_emb, 'embedding'):  # e.g., OpenAI Embedding object
        q_emb = q_emb.embedding
    elif isinstance(q_emb, np.ndarray):  # if using numpy
        q_emb = q_emb.tolist()

    q_emb_str = "[" + ",".join(str(float(x)) for x in q_emb) + "]"
    

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

# results = search("What is pgvector?")
# for row in results:
#     print(f"[{row[3]:.4f}] {row[1][:80]}...")


# ── Cleanup ────────────────────────────────────────────
# cur.close()
# conn.close()

if __name__ == "__main__":
    print(type(get_embedding("What is the meaning of life?")[0]))
    