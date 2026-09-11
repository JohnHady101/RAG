"""Central configuration for the RAG pipeline.

Every value can be overridden with an environment variable.
Hardcoded values below are local-dev defaults (docker-compose service names).
"""

import os

# ── Postgres / pgvector ──────────────────────────────────
DB_CONFIG = {
    "host": os.environ.get("PGVECTOR_HOST", "localhost"),
    "port": int(os.environ.get("PGVECTOR_PORT", "5432")),
    "dbname": os.environ.get("PGVECTOR_DB", "vectordb"),
    "user": os.environ.get("PGVECTOR_USER", "myuser"),
    "password": os.environ.get("PGVECTOR_PASSWORD", "mypassword"),
}

# ── Gemini ───────────────────────────────────────────────
# Prefer GEMINI_API_KEY from the environment; falls back to the
# previously hardcoded dev key so the existing container keeps working.
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "API_KEY",
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "3072"))
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-3.6-flash")

# ── Chunking ─────────────────────────────────────────────
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# ── Data ─────────────────────────────────────────────────
PDF_PATH = os.environ.get(
    "RAG_PDF_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "annualreport-2025.pdf"),
)
