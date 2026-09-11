"""Single entry point for the RAG pipeline.

Run with src/ on the path, e.g. from the repo root:

    python src/main.py ingest [PDF_PATH] [--limit N]
    python src/main.py query "your question" [--top-k N]
    python src/main.py ask "your question" [--top-k N]
    python src/main.py demo "your question" [--top-k N] [--show-prompt]
"""

import argparse

from config import EMBEDDING_MODEL, GENERATION_MODEL, PDF_PATH
from db import get_cursor, init_db
from embeddings import get_client, get_embedding
from ingest import ingest_pdf
from qa import answer_question, retrieve


def _cmd_ingest(args) -> None:
    ids = ingest_pdf(args.pdf, limit=args.limit)
    print(f"ingested {len(ids)} chunk(s)")


def _cmd_query(args) -> None:
    for doc_id, content, _metadata, similarity in retrieve(
        args.question, top_k=args.top_k
    ):
        print(f"{similarity * 100:05.2f}% | id={doc_id} | {content[:120]}...")


def _cmd_ask(args) -> None:
    print(answer_question(args.question, top_k=args.top_k))


def run_demo(question: str, top_k: int = 5, show_prompt: bool = False) -> str:
    """Run a fully logged sample of the RAG pipeline end to end.

    Pipeline stages:
        1. LLM query embedding  (Gemini ``EMBEDDING_MODEL``)
        2. Vector-database search (pgvector cosine similarity ``<=>``)
        3. Prompt construction   (retrieved chunks -> grounded context)
        4. LLM answer generation (Gemini ``GENERATION_MODEL``)

    Args:
        question: The natural-language question to answer.
        top_k: How many chunks to retrieve from the vector store.
        show_prompt: Print the full prompt sent to the LLM.

    Returns:
        The LLM's answer text.
    """
    # ── Step 0: make sure the pgvector schema exists ─────────────
    print("== Step 0: init vector database ==")
    init_db()
    print("   pgvector extension + documents table ready.\n")

    # ── Step 1: embed the LLM query ──────────────────────────────
    print(f"== Step 1: embed query with {EMBEDDING_MODEL} ==")
    print(f'   question: "{question}"')
    query_embedding = get_embedding(question)
    print(f"   embedding dim: {len(query_embedding)}")
    print(f"   embedding preview: {query_embedding[:5]}...\n")

    # ── Step 2: vector-database research (similarity search) ──────
    print(f"== Step 2: search pgvector (top_k={top_k}) ==")
    print("   SQL: SELECT id, content, metadata,")
    print("               1 - (embedding <=> query::vector) AS similarity")
    print("        FROM documents ORDER BY embedding <=> query::vector LIMIT top_k")
    cur = get_cursor()
    cur.execute(
        """
        SELECT id, content, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_embedding, query_embedding, top_k),
    )
    rows = cur.fetchall()
    if not rows:
        print("   no chunks found — run `python src/main.py ingest` first.\n")
    for rank, (doc_id, content, metadata, similarity) in enumerate(rows, start=1):
        snippet = content[:150].replace("\n", " ")
        print(f"   #{rank} similarity={similarity:.4f} id={doc_id} metadata={metadata}")
        print(f"       {snippet}...")
    print()

    # ── Step 3: build the grounded prompt ────────────────────────
    print("== Step 3: build grounded prompt ==")
    context = "\n\n---\n\n".join(
        f"[chunk {doc_id} | similarity {similarity:.2f}]\n{content}"
        for doc_id, content, _metadata, similarity in rows
    )
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    print(f"   context chunks: {len(rows)}, prompt chars: {len(prompt)}")
    if show_prompt:
        print("   --- prompt start ---")
        print(prompt)
        print("   --- prompt end ---")
    print()

    # ── Step 4: LLM answer generation ────────────────────────────
    print(f"== Step 4: generate answer with {GENERATION_MODEL} ==")
    response = get_client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    answer = response.text
    print("   --- answer ---")
    print(answer)
    print("   --- sources ---")
    for doc_id, _content, _metadata, similarity in rows:
        print(f"   id={doc_id} similarity={similarity:.4f}")
    return answer


def _cmd_demo(args) -> None:
    run_demo(args.question, top_k=args.top_k, show_prompt=args.show_prompt)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="RAG pipeline over the PDF knowledge base"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="chunk a PDF and store it")
    p_ingest.add_argument("pdf", nargs="?", default=PDF_PATH)
    p_ingest.add_argument("--limit", type=int, default=None)
    p_ingest.set_defaults(func=_cmd_ingest)

    p_query = sub.add_parser("query", help="print top-k similar chunks")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.set_defaults(func=_cmd_query)

    p_ask = sub.add_parser("ask", help="answer using retrieved chunks")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.set_defaults(func=_cmd_ask)

    p_demo = sub.add_parser(
        "demo",
        help="full sample: embed query -> pgvector search -> grounded LLM answer",
    )
    p_demo.add_argument("question")
    p_demo.add_argument("--top-k", type=int, default=5)
    p_demo.add_argument("--show-prompt", action="store_true")
    p_demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
