# RAG — PDF Retrieval-Augmented Generation with pgvector + Gemini

A minimal, Docker-ready RAG pipeline over a PDF knowledge base:

**PDF → chunks → Gemini embeddings → pgvector → cosine search → grounded Gemini answer.**

Default knowledge base: `data/annualreport-2025.pdf`.

## Architecture

```
data/*.pdf
   │
   ▼
doc_loaders.pdf_loader (LangChain PyMuPDFLoader)
   │
   ▼
chunking.create_knowledge_base (RecursiveCharacterTextSplitter)
   │  CHUNK_SIZE=1000, CHUNK_OVERLAP=200
   ▼
embeddings.get_embedding (gemini-embedding-2, 3072-dim)
   │
   ▼
vector_store.add_documents → Postgres pgvector (documents table)
   │
   ▼
query → embed → pgvector cosine search (<=>) → top-k chunks
   │
   ▼
qa.answer_question → grounded prompt → gemini-3.6-flash answer
```

## Project Structure

```
.
├── data/
│   └── annualreport-2025.pdf      # default knowledge base
├── src/
│   ├── main.py                    # CLI entry point: ingest / query / ask / demo
│   ├── config.py                  # central config (env-overridable)
│   ├── doc_loaders.py             # pdf_loader, load_text_file
│   ├── chunking.py                # create_knowledge_base
│   ├── embeddings.py              # Gemini client + get_embedding
│   ├── db.py                      # psycopg2 connection, init_db
│   ├── vector_store.py            # add_document(s), search
│   ├── ingest.py                  # ingest_pdf orchestration
│   ├── qa.py                      # retrieve, answer_question
│   └── sql/
│       └── 001_create_documents.sql
├── tests/
│   ├── test_chunking.py
│   ├── test_doc_loaders.py
│   └── test_ingest.py
├── dockerfile
├── docker-compose.yml             # app + pgvector/pgvector:pg16
├── requirements.txt
├── pytest.ini                     # pythonpath = src
└── .github/workflows/ci-cd.yml    # ruff + pytest + package on main
```

### Module responsibilities

| File | Purpose |
|------|---------|
| `src/main.py` | CLI: `ingest`, `query`, `ask`, `demo` |
| `src/config.py` | `DB_CONFIG`, `GEMINI_API_KEY`, models, chunking, `PDF_PATH` |
| `src/doc_loaders.py` | `pdf_loader()` via `PyMuPDFLoader` |
| `src/chunking.py` | `RecursiveCharacterTextSplitter` → `list[Document]` |
| `src/embeddings.py` | lazy `genai.Client`, `get_embedding(text)` |
| `src/db.py` | lazy shared connection, `init_db()`, `get_cursor()`, `close()` |
| `src/vector_store.py` | `INSERT`, cosine search `1 - (embedding <=> query)` |
| `src/ingest.py` | `ingest_pdf(pdf_path, chunk_size, chunk_overlap, limit)` |
| `src/qa.py` | `retrieve()` + `answer_question()` grounded prompt |

### Database schema (`src/sql/001_create_documents.sql`)

```sql
CREATE TABLE IF NOT EXISTS documents (
    id        SERIAL PRIMARY KEY,
    content   TEXT NOT NULL,
    metadata  JSONB DEFAULT '{}',
    embedding vector(3072)
);
```

## Prerequisites

- Python 3.10+ (Docker image uses 3.13, CI tests 3.10 / 3.11)
- Docker + Docker Compose (for pgvector), or a local Postgres with `pgvector`
- Gemini API key (`GEMINI_API_KEY`)

## Configuration

All in `src/config.py`, loaded from the environment with `python-dotenv`
from the repo-root `.env` (see `.env.example`). Single source of truth:
`.env` / real environment. `config.py` holds **no default values** — a
missing variable raises `RuntimeError` naming it instead of silently
falling back. Precedence: real environment variables override `.env`.
`.env` is gitignored — never commit secrets.

```bash
cp .env.example .env   # then fill in GEMINI_API_KEY
```

| Variable | Set in | Description |
|----------|--------|-------------|
| `PGVECTOR_HOST` | `.env.example` | use `localhost` outside Docker, `pgvector` inside Compose |
| `PGVECTOR_PORT` | `.env.example` | |
| `PGVECTOR_DB` | `.env.example` | |
| `PGVECTOR_USER` | `.env.example` | |
| `PGVECTOR_PASSWORD` | `.env.example` | |
| `GEMINI_API_KEY` | you (no default) | **set in `.env`, do not commit a key** |
| `EMBEDDING_MODEL` | `.env.example` | |
| `EMBEDDING_DIM` | `.env.example` | must match `vector(3072)` |
| `GENERATION_MODEL` | `.env.example` | |
| `CHUNK_SIZE` | `.env.example` | |
| `CHUNK_OVERLAP` | `.env.example` | |
| `RAG_PDF_PATH` | `.env.example` | default ingest target (repo-relative paths resolve against repo root) |

`docker-compose.yml` passes `.env` to the `app` service via `env_file`
(and forces `PGVECTOR_HOST=pgvector` there); the `pgvector` service reads
`POSTGRES_*` from the same file with `myuser/mypassword/vectordb` fallbacks.

## Quickstart

### 1. Start pgvector + app shell

```powershell
docker compose up -d
# connection from host: postgresql://myuser:mypassword@localhost:5432/vectordb
# connection from app container: postgresql://myuser:mypassword@pgvector:5432/vectordb
```

### 2. Install deps (local, without Docker)

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set environment (`.env`)

```powershell
Copy-Item .env.example .env
# edit .env: set GEMINI_API_KEY, and for local runs set PGVECTOR_HOST=localhost
```

Linux/macOS:

```bash
cp .env.example .env   # then fill in GEMINI_API_KEY
# PGVECTOR_HOST=localhost for local runs (Compose overrides it to pgvector)
```

### 4. Ingest the PDF

```bash
python src/main.py ingest
python src/main.py ingest data/annualreport-2025.pdf --limit 10   # smoke test, first 10 chunks
```

Output: `ingested N chunk(s)` + `Inserted document id=N` per chunk.

### 5. Query / Ask

```bash
# retrieval only — prints similarity % + snippet
python src/main.py query "What was revenue in 2025?" --top-k 5

# RAG answer — grounded in retrieved chunks
python src/main.py ask "What was revenue in 2025?" --top-k 5

# fully logged end-to-end sample (init db → embed → search → prompt → generate)
python src/main.py demo "What was revenue in 2025?" --top-k 5 --show-prompt
```

`demo` stages (`src/main.py:36`):

1. `init_db()` — ensure `vector` extension + `documents` table
2. Embed query with `EMBEDDING_MODEL`
3. pgvector search: `ORDER BY embedding <=> query::vector LIMIT top_k`
4. Build prompt: `Answer ... using only the context below...`
5. Generate with `GENERATION_MODEL`, print answer + source ids

## Testing & Lint

```bash
pytest tests/ -v
ruff check .
```

- `pytest.ini`: `pythonpath = src`
- `tests/test_chunking.py`, `test_doc_loaders.py`: create dummy PDFs with PyMuPDF, no DB/API needed
- `tests/test_ingest.py`: monkeypatches `init_db` / `add_documents`
- CI (`.github/workflows/ci-cd.yml`): spins up `pgvector/pgvector:pg16` service, copies `.env.example` → `.env` for non-secret defaults, injects only `GEMINI_API_KEY` (from GitHub Secrets) + `PGVECTOR_HOST=localhost`, then runs `ruff check` + `pytest`. Zips `src/ + requirements.txt` to `deployment_package.zip` on `main` push.

## Dependencies (`requirements.txt`)

- `langchain_community==0.4.2`, `langchain-text-splitters`, `pandas`, `pymupdf`
- `pgvector`, `psycopg2`, `google-genai`, `python-dotenv`
- `pytest`, `ruff`

## Notes / Limitations

- `add_documents()` embeds + inserts one row per chunk sequentially — fine for a single report, slow for large corpora (no batching).
- No hybrid search, reranking, or eval harness — pure dense cosine (`<=>`).
- No hardcoded secrets: `GEMINI_API_KEY` must come from `.env` or the environment.
- pgvector data is ephemeral in Compose (volume commented out in `docker-compose.yml`).
