"""Central configuration for the RAG pipeline.

Every value is loaded from the environment, with local-dev defaults in
`.env` (see `.env.example`). python-dotenv loads that file automatically,
so `export`ing variables manually is not required for local runs.
"""

import os

from dotenv import load_dotenv

# Load `<repo>/.env` (no-op if the file is missing, e.g. in CI).
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── Postgres / pgvector ──────────────────────────────────
DB_CONFIG = {
    "host": os.environ.get("PGVECTOR_HOST", "pgvector"),
    "port": int(os.environ.get("PGVECTOR_PORT", "5432")),
    "dbname": os.environ.get("PGVECTOR_DB", "vectordb"),
    "user": os.environ.get("PGVECTOR_USER", "myuser"),
    "password": os.environ.get("PGVECTOR_PASSWORD", "mypassword"),
}

# ── Gemini ───────────────────────────────────────────────
# No hardcoded fallback: set GEMINI_API_KEY in `.env` or the environment.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "3072"))
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-3.6-flash")

# ── Chunking ─────────────────────────────────────────────
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# ── Data ─────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_pdf_path = os.environ.get("RAG_PDF_PATH", os.path.join(_REPO_ROOT, "data", "annualreport-2025.pdf"))
# Resolve repo-relative paths (e.g. `data/annualreport-2025.pdf` from `.env`)
# against the repo root so the CLI works from any cwd.
PDF_PATH = _pdf_path if os.path.isabs(_pdf_path) else os.path.join(_REPO_ROOT, _pdf_path)
