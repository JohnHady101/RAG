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
    """Create the pgvector extension and tables (idempotent)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    sql_path = os.path.join(os.path.dirname(__file__), "sql", "001_create_documents.sql")
    with open(sql_path) as f:
        cur.execute(f.read())
    conn.commit()
    cur.close()


def close() -> None:
    """Close the shared connection (mainly useful for tests)."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
