"""Postgres/pgvector connection management.

Connections are created lazily (nothing happens at import time), so
importing this module never requires a live database. Call init_db()
once before first use to create the extension and tables.
"""

import os

import psycopg2
from pgvector.psycopg2 import register_vector

from config import DB_CONFIG

_conn = None

_SQL_DIR = os.path.join(os.path.dirname(__file__), "sql")


def _load_sql(filename: str) -> str:
    """Load a SQL file from src/sql/."""
    with open(os.path.join(_SQL_DIR, filename)) as f:
        return f.read()


def get_conn():
    """Return a shared connection, creating it on first use."""
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(**DB_CONFIG)
        register_vector(_conn)
    return _conn


def get_cursor():
    """Return a cursor on the shared connection."""
    return get_conn().cursor()


def init_db() -> None:
    """Create the pgvector extension, tables, and BM25 index (idempotent)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    cur.execute(_load_sql("001_create_documents.sql"))
    conn.commit()
    cur.execute(_load_sql("add_vector_col.sql"))
    conn.commit()
    cur.close()


def close() -> None:
    """Close the shared connection (mainly useful for tests)."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
