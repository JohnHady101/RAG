"""Central configuration for the RAG pipeline.

Single source of truth: the environment, loaded from the repo-root
`.env` (see `.env.example`). This module holds NO default values —
a missing variable fails fast with a clear error instead of silently
using a stale hardcoded fallback.
"""

import os

from dotenv import load_dotenv

# Load `<repo>/.env` (no-op if the file is missing, e.g. in CI where
# variables come from the real environment instead).
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(
            f"Missing required config {name!r}: set it in `.env` "
            f"(see `.env.example`) or export it."
        )
    return value


def _required_int(name: str) -> int:
    raw = _required(name)
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(
            f"Invalid config {name!r}={raw!r}: expected an integer."
        ) from None


# ── Postgres / pgvector ──────────────────────────────────
DB_CONFIG = {
    "host": _required("PGVECTOR_HOST"),
    "port": _required_int("PGVECTOR_PORT"),
    "dbname": _required("PGVECTOR_DB"),
    "user": _required("PGVECTOR_USER"),
    "password": _required("PGVECTOR_PASSWORD"),
}

# ── Gemini ───────────────────────────────────────────────
GEMINI_API_KEY = _required("GEMINI_API_KEY")
EMBEDDING_MODEL = _required("EMBEDDING_MODEL")
EMBEDDING_DIM = _required_int("EMBEDDING_DIM")
GENERATION_MODEL = _required("GENERATION_MODEL")

# ── Chunking ─────────────────────────────────────────────
CHUNK_SIZE = _required_int("CHUNK_SIZE")
CHUNK_OVERLAP = _required_int("CHUNK_OVERLAP")

# ── Data ─────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_pdf_path = _required("RAG_PDF_PATH")
# Resolve repo-relative paths (e.g. `data/annualreport-2025.pdf` from `.env`)
# against the repo root so the CLI works from any cwd.
PDF_PATH = _pdf_path if os.path.isabs(_pdf_path) else os.path.join(_REPO_ROOT, _pdf_path)
