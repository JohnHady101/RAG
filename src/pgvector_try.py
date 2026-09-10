import psycopg2
import pgvector
from pgvector.psycopg2 import register_vector
import os
from google import genai

# ── Config ──────────────────────────────────────────────
DB_CONFIG = {
    "host": "pgvector",
    "port": 5432,
    "dbname": "vectordb",
    "user": "myuser",
    "password": "mypassword",
}
EMBEDDING_MODEL = "text-embedding-004"  # 1536 dimensions
EMBEDDING_DIM = 1536
API = "API_KEY"

# ── Step 1: Connect & enable pgvector ──────────────────
conn = psycopg2.connect(**DB_CONFIG)


cur = conn.cursor()

# 1. Create the extension FIRST
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# 2. THEN register the vector type with psycopg2
register_vector(conn)       

# ── Step 2: Create the table (run once) ────────────────
with open(os.path.dirname(__file__) + "/create_table.sql", "r") as f:
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

    return result.embeddings

# ── Step 4: Insert the document ────────────────────────
def add_document(text: str, metadata: dict | None = None):
    embedding = get_embedding(text)
    cur.execute(
        """
        INSERT INTO documents (content, metadata, embedding)
        VALUES (%s, %s, %s)
        RETURNING id;
        """,
        (text, metadata or {}, embedding),
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
# def search(query: str, top_k: int = 3):
#     q_emb = get_embedding(query)
#     cur.execute(
#         """
#         SELECT id, content, metadata,
#                1 - (embedding <=> %s) AS similarity
#         FROM documents
#         ORDER BY embedding <=> %s
#         LIMIT %s;
#         """,
#         (q_emb, q_emb, top_k),
#     )
#     return cur.fetchall()

# results = search("What is pgvector?")
# for row in results:
#     print(f"[{row[3]:.4f}] {row[1][:80]}...")

print(get_embedding("hi there"))

# ── Cleanup ────────────────────────────────────────────
cur.close()
conn.close()