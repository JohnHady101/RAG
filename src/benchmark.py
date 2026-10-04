"""BEIR benchmark for the RAG retrieval stack.

Evaluates the same retrieval logic the app serves (``qa.retrieve`` /
``ingestion.vector_store``) on a standard BEIR collection and reports
the usual IR metrics: NDCG@k, MAP@k, Recall@k and Precision@k.

BEIR format refresher (also used by this module)::

    corpus: dict[doc_id, {"title": ..., "text": ...}]
    queries: dict[query_id, query_text]
    qrels:  dict[query_id, dict[doc_id, relevance_grade]]  (grade > 0 = relevant)

Usage (from the repo root)::

    # Fast, no model download — mirrors production qa.retrieve (BM25):
    python src/benchmark.py --dataset scifact --mode bm25

    # Dense cosine with the same embedding model as the vector store:
    python src/benchmark.py --dataset scifact --mode dense --max-queries 50

    # Reciprocal-rank-fusion hybrid of the two:
    python src/benchmark.py --dataset scifact --mode hybrid --max-queries 50

    # Live end-to-end path: ingest the BEIR corpus into Postgres/pgvector
    # and query it through qa.retrieve (needs a running DB + .env):
    python src/benchmark.py --dataset scifact --mode pg --max-corpus 500 --max-queries 20

    # Fully offline demo (no download needed):
    python src/benchmark.py --dataset synthetic --mode bm25

Dataset loading priority per ``--dataset`` (except ``synthetic``):

    1. ``beir`` package (``GenericDataLoader``) when installed
       (``pip install beir``).
    2. Official BEIR zip download cached under ``--data-dir``
       (default ``data/beir``).
    3. Tiny built-in synthetic collection (with a warning).

Only numpy + stdlib are required at import time. Heavy deps
(torch/transformers/langchain/DB client) are imported lazily inside the
retriever that needs them, so ``ruff``/``pytest`` stay green without a
GPU or database.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import urllib.request
import zipfile

# Make `import config / db / qa / ...` work both as
# `python src/benchmark.py` (script dir on sys.path) and as
# `pytest` / `python -m benchmark` with `pythonpath = src`.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy ships with the project
    np = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEIR_DOWNLOAD_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
)

# Small default: ~5k docs, fast to embed and quick to download.
SUPPORTED_DATASETS = (
    "trec-covid",
    "scifact",
    "fiqa",
    "nfcorpus",
    "arguana",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "hotpotqa",
    "msmarco",
    "nq",
    "synthetic",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (shared by the in-memory BM25)."""
    return _TOKEN_RE.findall(text.lower())


def corpus_text(title: str, text: str) -> str:
    """BEIR convention: the retrievable text is ``title + " " + text``."""
    title, text = (title or "").strip(), (text or "").strip()
    if title and text:
        return f"{title} {text}"
    return title or text


# ---------------------------------------------------------------------------
# Pure-Python Okapi BM25 (mirrors production Postgres ts_rank ranking)
# ---------------------------------------------------------------------------

class InMemoryBM25:
    """Minimal Okapi BM25 over a pre-tokenized corpus.

    Mirrors the *role* of production ``bm25_search`` (lexical ranking over
    chunk text) without needing a live Postgres, so the benchmark runs in
    CI. Defaults are standard Okapi values (k1=1.2, b=0.75).
    """

    def __init__(
        self,
        tokenized_corpus: list[list[str]],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.docs = tokenized_corpus
        self.n_docs = len(tokenized_corpus)
        self.doc_len = [len(d) for d in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / max(self.n_docs, 1)
        self.doc_freq: dict[str, int] = {}
        for doc in tokenized_corpus:
            for term in set(doc):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def score(self, query_tokens: list[str], index: int) -> float:
        doc = self.docs[index]
        if not doc:
            return 0.0
        tf: dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        dl = self.doc_len[index]
        norm = 1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9)
        total = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            total += idf * freq * (self.k1 + 1.0) / (freq + self.k1 * norm)
        return total

    def search(
        self, query_tokens: list[str], top_k: int
    ) -> tuple[list[int], list[float]]:
        scores = [(self.score(query_tokens, i), i) for i in range(self.n_docs)]
        scores.sort(key=lambda pair: pair[0], reverse=True)
        top = [(s, i) for s, i in scores[:top_k] if s > 0]
        return [i for _, i in top], [s for s, _ in top]


# ---------------------------------------------------------------------------
# Metrics (BEIR-style, graded qrels)
# ---------------------------------------------------------------------------

def _binary_relevant(grade: float) -> bool:
    return grade > 0


def _dcg(gains: list[float]) -> float:
    return sum(
        (2.0**g - 1.0) / math.log2(i + 2) for i, g in enumerate(gains)
    )


def metrics_for_query(
    retrieved: list[str],
    relevant: dict[str, int | float],
    k_values: list[int],
) -> dict[int, dict[str, float]]:
    """Compute NDCG/MAP/Recall/Precision at each cutoff for one query.

    * Relevance is binary (grade > 0) except NDCG, which uses graded gains.
    * MAP@k is the TREC variant: ``sum(P@i * rel_i) / #relevant`` truncated
      at ``k`` (0.0 when a query has no judged relevant docs).
    """
    n_relevant = sum(1 for g in relevant.values() if _binary_relevant(g))
    ideal_gains = sorted(
        (float(g) for g in relevant.values() if _binary_relevant(g)),
        reverse=True,
    )
    out: dict[int, dict[str, float]] = {}
    for k in k_values:
        top = retrieved[:k]
        gains = [float(relevant.get(doc_id, 0)) for doc_id in top]
        hits = sum(1 for doc_id in top if _binary_relevant(relevant.get(doc_id, 0)))
        idcg = _dcg(ideal_gains[:k])
        ndcg = (_dcg(gains) / idcg) if idcg > 0 else 0.0
        # Average precision truncated at k.
        ap_sum = 0.0
        hits_so_far = 0
        for i, doc_id in enumerate(top, start=1):
            if _binary_relevant(relevant.get(doc_id, 0)):
                hits_so_far += 1
                ap_sum += hits_so_far / i
        ap = (ap_sum / n_relevant) if n_relevant else 0.0
        out[k] = {
            "ndcg": ndcg,
            "map": ap,
            "recall": (hits / n_relevant) if n_relevant else 0.0,
            "precision": (hits / k) if k else 0.0,
        }
    return out


def evaluate_results(
    qrels: dict[str, dict[str, int | float]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[int, dict[str, float]]:
    """Average per-query metrics over all judged queries.

    ``results`` maps ``query_id -> {doc_id: score}`` (higher = better).
    Queries with no judged relevant docs still count (as zeros), matching
    ``beir.retrieval.evaluation.EvaluateRetrieval`` behaviour.
    """
    k_values = sorted(set(k_values))
    accum: dict[int, dict[str, float]] = {
        k: {"ndcg": 0.0, "map": 0.0, "recall": 0.0, "precision": 0.0}
        for k in k_values
    }
    n_queries = 0
    for qid, judged in qrels.items():
        ranked = sorted(
            results.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True
        )
        retrieved = [doc_id for doc_id, _ in ranked]
        per_k = metrics_for_query(retrieved, judged, k_values)
        n_queries += 1
        for k in k_values:
            for metric, value in per_k[k].items():
                accum[k][metric] += value
    if n_queries:
        for k in k_values:
            for metric in accum[k]:
                accum[k][metric] /= n_queries
    out = {**accum, "_num_queries": n_queries}  # type: ignore[dict-item]
    return out


# Backwards-friendly alias used by some BEIR-style harnesses.
compute_metrics = evaluate_results


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def synthetic_dataset() -> tuple[dict, dict, dict]:
    """Tiny offline collection so the harness always runs without network."""
    corpus = {
        f"d{i}": {
            "title": f"Report part {i}",
            "text": text,
        }
        for i, text in enumerate(
            [
                "Annual revenue grew to 4.2 billion dollars in 2025.",
                "Net income fell due to restructuring costs in 2025.",
                "The company launched a new cloud product line.",
                "Revenue guidance for next year is conservative.",
                "Operating margin improved on lower cloud costs.",
                "The board approved a share buyback program.",
                "Research spending focused on machine learning chips.",
                "Cash flow from operations reached a record high.",
                "The annual report discusses risk factors at length.",
                "Dividends were raised for the tenth straight year.",
                "A new factory opened to expand hardware capacity.",
                "Revenue in Europe declined on currency headwinds.",
            ]
        )
    }
    queries = {
        "q0": "What was revenue in 2025?",
        "q1": "Did the company raise dividends?",
        "q2": "What drove the improvement in operating margin?",
    }
    qrels = {
        "q0": {"d0": 2, "d3": 1},
        "q1": {"d9": 2},
        "q2": {"d4": 2},
    }
    return corpus, queries, qrels


def _load_with_beir(
    dataset: str, split: str, data_dir: str
) -> tuple[dict, dict, dict] | None:
    """Try the ``beir`` package's GenericDataLoader; None if unavailable."""
    try:
        from beir.datasets.data_loader import GenericDataLoader  # type: ignore
    except ImportError:
        return None
    # Use the standard BEIR on-disk layout: download + GenericDataLoader.
    local = os.path.join(data_dir, dataset)
    if not os.path.isdir(local):
        _download_beir_zip(dataset, data_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(
        split=split
    )
    return corpus, queries, qrels


def _download_beir_zip(dataset: str, data_dir: str) -> str:
    """Download + extract the official BEIR zip into ``data_dir``."""
    os.makedirs(data_dir, exist_ok=True)
    dest_dir = os.path.join(data_dir, dataset)
    if os.path.isdir(dest_dir):
        return dest_dir
    url = BEIR_DOWNLOAD_URL.format(dataset)
    zip_path = os.path.join(data_dir, f"{dataset}.zip")
    print(f"downloading BEIR dataset {dataset!r} from {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)
    return dest_dir


def _load_beir_from_disk(
    dataset: str, split: str, data_dir: str
) -> tuple[dict, dict, dict]:
    """Parse an extracted BEIR folder (corpus/queries/qrels) without ``beir``."""
    base = os.path.join(data_dir, dataset)
    if not os.path.isdir(base):
        _download_beir_zip(dataset, data_dir)
    corpus_path = os.path.join(base, "corpus.jsonl")
    queries_path = os.path.join(base, "queries.jsonl")
    # BEIR ships qrels as `{split}.tsv` (older) or `qrels/{split}.tsv` (newer).
    candidates = [
        os.path.join(base, "qrels", f"{split}.tsv"),
        os.path.join(base, f"{split}.tsv"),
        os.path.join(base, "qrels.tsv"),
    ]
    qrels_path = next((p for p in candidates if os.path.isfile(p)), candidates[0])

    corpus: dict[str, dict[str, str]] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("_id", row.get("id", row.get("doc_id", ""))))
            corpus[doc_id] = {
                "title": row.get("title", "") or "",
                "text": row.get("text", "") or "",
            }
    queries: dict[str, str] = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("_id", row.get("id", "")))
            queries[qid] = row.get("text", "") or ""
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            qid, doc_id, score = str(row[0]), str(row[1]), row[2]
            if qid.lower() in ("query-id", "qid", "query_id"):
                continue  # header row
            try:
                grade = int(float(score))
            except ValueError:
                continue
            qrels.setdefault(qid, {})[doc_id] = grade
    return corpus, queries, qrels


def load_beir_dataset(
    dataset: str, split: str = "test", data_dir: str = "data/beir"
) -> tuple[dict, dict, dict]:
    """Load ``(corpus, queries, qrels)`` for a BEIR dataset.

    Falls back to :func:`synthetic_dataset` for ``dataset="synthetic"``
    or when the download fails (offline environments).
    """
    if dataset == "synthetic":
        return synthetic_dataset()
    # 1. Prefer the `beir` package when installed (canonical loader).
    try:
        loaded = _load_with_beir(dataset, split, data_dir)
        if loaded is not None:
            return loaded
    except Exception as exc:  # noqa: BLE001 - fall through to manual parser
        print(f"beir loader failed ({exc}); trying manual download ...")
    # 2. Manual parse of the official zip (no extra dependency).
    try:
        return _load_beir_from_disk(dataset, split, data_dir)
    except Exception as exc:  # noqa: BLE001 - offline fallback to synthetic
        print(
            f"warning: could not load BEIR dataset {dataset!r} ({exc}); "
            "using the built-in synthetic collection."
        )
        return synthetic_dataset()


def subsample(
    corpus: dict,
    queries: dict,
    qrels: dict,
    max_corpus: int | None,
    max_queries: int | None,
) -> tuple[dict, dict, dict]:
    """Deterministically cap corpus/queries (sorted ids) for quick runs."""
    if max_corpus is not None:
        keep = set(sorted(corpus)[:max_corpus])
        corpus = {d: corpus[d] for d in sorted(keep)}
        qrels = {
            q: {d: s for d, s in docs.items() if d in keep}
            for q, docs in qrels.items()
        }
        qrels = {q: docs for q, docs in qrels.items() if docs}
    if max_queries is not None:
        keep_q = set(sorted(queries)[:max_queries])
        queries = {q: queries[q] for q in sorted(keep_q) if q in queries}
        qrels = {q: docs for q, docs in qrels.items() if q in queries}
    # Drop queries with no judged relevant docs (nothing to score).
    queries = {q: t for q, t in queries.items() if q in qrels}
    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Chunking (same splitter as production ingestion, optional for long docs)
# ---------------------------------------------------------------------------

def maybe_chunk_corpus(
    corpus: dict,
    chunk_size: int = 0,
    chunk_overlap: int = 200,
) -> tuple[list[str], list[str]]:
    """Flatten the corpus to parallel ``(chunk_texts, chunk_doc_ids)`` lists.

    ``chunk_size <= 0`` (default) keeps one text per BEIR doc, which is
    right for BEIR passages. Pass ``--chunk-size 1000`` to exercise the
    production ``RecursiveCharacterTextSplitter`` path used by
    ``ingestion.chunking.create_knowledge_base``.
    """
    doc_ids = sorted(corpus)
    if chunk_size <= 0:
        texts = [
            corpus_text(corpus[d].get("title", ""), corpus[d].get("text", ""))
            for d in doc_ids
        ]
        return texts, doc_ids
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    if RecursiveCharacterTextSplitter is None:
        # Naive sliding-window fallback (no extra dependency).
        texts: list[str] = []
        owners: list[str] = []
        step = max(chunk_size - chunk_overlap, 1)
        for d in doc_ids:
            full = corpus_text(
                corpus[d].get("title", ""), corpus[d].get("text", "")
            )
            if len(full) <= chunk_size:
                texts.append(full)
                owners.append(d)
            else:
                for start in range(0, len(full), step):
                    texts.append(full[start : start + chunk_size])
                    owners.append(d)
                    if start + chunk_size >= len(full):
                        break
        return texts, owners
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts, owners = [], []
    for d in doc_ids:
        full = corpus_text(corpus[d].get("title", ""), corpus[d].get("text", ""))
        for chunk in splitter.split_text(full):
            texts.append(chunk)
            owners.append(d)
    return texts, owners


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

def _get_embedding_batch_fn():
    """Return a ``list[str] -> np.ndarray`` embedder using production code."""
    try:
        from models.huggingface import vectorize_texts  # batched, preferred
    except ImportError:
        try:
            from src.models.huggingface import vectorize_texts
        except ImportError:
            vectorize_texts = None  # type: ignore[assignment]
    if vectorize_texts is not None:
        def embed(texts: list[str]):
            vecs = vectorize_texts(texts)
            if hasattr(vecs, "detach"):
                vecs = vecs.detach()
            try:
                import torch

                if isinstance(vecs, torch.Tensor):
                    vecs = vecs.cpu().numpy()
            except ImportError:
                pass
            arr = np.asarray(vecs, dtype=np.float64)
            return arr

        return embed
    # Fall back to one-by-one production get_embedding.
    try:
        from models.embeddings import get_embedding
    except ImportError:
        from src.models.embeddings import get_embedding  # type: ignore[no-redef]

    def embed(texts: list[str]):
        return np.asarray([get_embedding(t) for t in texts], dtype=np.float64)

    return embed


def dense_retrieve(
    queries: dict[str, str],
    chunk_texts: list[str],
    chunk_doc_ids: list[str],
    top_k: int,
    batch_size: int = 32,
) -> dict[str, dict[str, float]]:
    """In-memory dense cosine retrieval (same ordering as pgvector ``<=>``).

    Chunk scores are aggregated to doc level with max-pooling (``maxP``):
    each BEIR doc scores as its best-matching chunk.
    """
    if np is None:
        raise RuntimeError("numpy is required for dense retrieval.")
    embed = _get_embedding_batch_fn()
    doc_ids = sorted(set(chunk_doc_ids))
    doc_index = {d: i for i, d in enumerate(doc_ids)}

    # Embed corpus in batches.
    chunk_vecs = []
    for start in range(0, len(chunk_texts), batch_size):
        chunk_vecs.append(embed(chunk_texts[start : start + batch_size]))
    corpus_mat = np.vstack(chunk_vecs).astype(np.float64)
    norms = np.linalg.norm(corpus_mat, axis=1, keepdims=True)
    corpus_mat = corpus_mat / np.maximum(norms, 1e-12)

    qids = sorted(queries)
    query_mat = embed([queries[q] for q in qids]).astype(np.float64)
    query_mat = query_mat / np.maximum(
        np.linalg.norm(query_mat, axis=1, keepdims=True), 1e-12
    )

    # Cosine similarity; argpartition for top chunks then maxP to doc level.
    sims = query_mat @ corpus_mat.T
    results: dict[str, dict[str, float]] = {}
    take = min(top_k * max(1, len(chunk_texts) // max(len(doc_ids), 1) + 1), len(chunk_texts))
    take = max(take, top_k)
    for qi, qid in enumerate(qids):
        row = sims[qi]
        idx = np.argpartition(-row, take - 1)[:take]
        idx = idx[np.argsort(-row[idx])]
        best: dict[str, float] = {}
        for ci in idx:
            d = chunk_doc_ids[int(ci)]
            s = float(row[int(ci)])
            if d not in best or s > best[d]:
                best[d] = s
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        # Map to positional scores so downstream code sees doc ids.
        _ = doc_index
        results[qid] = {d: s for d, s in ranked}
    return results


def bm25_retrieve(
    queries: dict[str, str],
    chunk_texts: list[str],
    chunk_doc_ids: list[str],
    top_k: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, dict[str, float]]:
    """In-memory Okapi BM25 retrieval with maxP chunk->doc aggregation."""
    tokenized = [tokenize(t) for t in chunk_texts]
    model = InMemoryBM25(tokenized, k1=k1, b=b)
    results: dict[str, dict[str, float]] = {}
    for qid in sorted(queries):
        qtok = tokenize(queries[qid])
        idx, scores = model.search(qtok, top_k=top_k * 4)
        best: dict[str, float] = {}
        for ci, s in zip(idx, scores):
            d = chunk_doc_ids[ci]
            if d not in best or s > best[d]:
                best[d] = s
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results[qid] = {d: s for d, s in ranked}
    return results


def rrf_fuse(
    runs: list[dict[str, dict[str, float]]],
    top_k: int,
    rrf_k: int = 60,
) -> dict[str, dict[str, float]]:
    """Reciprocal Rank Fusion over per-query ranked lists (higher = better)."""
    fused: dict[str, dict[str, float]] = {}
    qids = set()
    for run in runs:
        qids.update(run)
    for qid in sorted(qids):
        scores: dict[str, float] = {}
        for run in runs:
            ranked = sorted(
                run.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True
            )
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        fused[qid] = dict(
            sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        )
    return fused


def pg_retrieve(
    queries: dict[str, str],
    corpus: dict,
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Live end-to-end retrieval through ``qa.retrieve`` (needs Postgres).

    Ingests ``corpus`` into the ``documents`` table with
    ``metadata={"beir_doc_id": <id>}`` and maps retrieved rows back to
    BEIR doc ids via that metadata. Requires ``.env`` + a running
    pgvector (see README quickstart).
    """
    try:
        from db import get_cursor, init_db
    except ImportError:
        from src.db import get_cursor, init_db  # type: ignore[no-redef]
    try:
        from qa import retrieve
    except ImportError:
        from src.qa import retrieve  # type: ignore[no-redef]
    try:
        from ingestion.vector_store import add_documents
    except ImportError:
        from src.ingestion.vector_store import add_documents  # type: ignore[no-redef]
    try:
        from langchain_core.documents import Document
    except ImportError:
        Document = None  # type: ignore[assignment]

    init_db()
    cur = get_cursor()
    cur.execute("DELETE FROM documents;")
    cur.connection.commit()

    doc_ids = sorted(corpus)
    texts = [
        corpus_text(corpus[d].get("title", ""), corpus[d].get("text", ""))
        for d in doc_ids
    ]
    if Document is not None:
        chunks = [
            Document(page_content=t, metadata={"beir_doc_id": d})
            for t, d in zip(texts, doc_ids)
        ]
    else:  # minimal shim with the attrs add_documents needs
        class _Doc:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

        chunks = [_Doc(t, {"beir_doc_id": d}) for t, d in zip(texts, doc_ids)]
    add_documents(chunks)

    results: dict[str, dict[str, float]] = {}
    for qid in sorted(queries):
        rows = retrieve(queries[qid], top_k=top_k)
        scored: dict[str, float] = {}
        for _row_id, content, metadata, score in rows:
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except ValueError:
                    metadata = {}
            beir_id = (metadata or {}).get("beir_doc_id")
            if beir_id is None:
                # Fall back to exact content match (chunking disabled here).
                for d, t in zip(doc_ids, texts):
                    if t == content:
                        beir_id = d
                        break
            if beir_id is not None and beir_id not in scored:
                scored[beir_id] = float(score)
        results[qid] = scored
    return results


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset: str = "scifact",
    split: str = "test",
    data_dir: str = "data/beir",
    mode: str = "bm25",
    top_k: int = 10,
    k_values: tuple[int, ...] | list[int] = (1, 3, 5, 10),
    max_corpus: int | None = None,
    max_queries: int | None = None,
    batch_size: int = 32,
    chunk_size: int = 0,
    chunk_overlap: int = 200,
    k1: float = 1.2,
    b: float = 0.75,
    rrf_k: int = 60,
) -> dict:
    """Load a BEIR collection, retrieve, and score. Returns a report dict."""
    k_values = sorted({k for k in k_values if k > 0} or {top_k})
    top_k = max(top_k, max(k_values))

    corpus, queries, qrels = load_beir_dataset(dataset, split, data_dir)
    corpus, queries, qrels = subsample(corpus, queries, qrels, max_corpus, max_queries)
    if not queries:
        raise RuntimeError(f"No judged queries left for {dataset}/{split}.")

    if mode == "pg":
        results = pg_retrieve(queries, corpus, top_k=top_k)
        chunk_texts: list[str] = []
    else:
        chunk_texts, chunk_doc_ids = maybe_chunk_corpus(
            corpus, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if mode == "bm25":
            results = bm25_retrieve(
                queries, chunk_texts, chunk_doc_ids, top_k=top_k, k1=k1, b=b
            )
        elif mode == "dense":
            results = dense_retrieve(
                queries,
                chunk_texts,
                chunk_doc_ids,
                top_k=top_k,
                batch_size=batch_size,
            )
        elif mode == "hybrid":
            dense = dense_retrieve(
                queries, chunk_texts, chunk_doc_ids,
                top_k=top_k, batch_size=batch_size,
            )
            lexical = bm25_retrieve(
                queries, chunk_texts, chunk_doc_ids,
                top_k=top_k, k1=k1, b=b,
            )
            results = rrf_fuse([dense, lexical], top_k=top_k, rrf_k=rrf_k)
        else:
            raise ValueError(
                f"Unknown mode {mode!r}: choose bm25|dense|hybrid|pg."
            )

    metrics = evaluate_results(qrels, results, k_values)
    return {
        "dataset": dataset,
        "split": split,
        "mode": mode,
        "top_k": top_k,
        "k_values": k_values,
        "metrics": {k: v for k, v in metrics.items() if k != "_num_queries"},
        "num_queries": metrics.get("_num_queries", len(queries)),
        "num_corpus": len(corpus),
        "num_chunks": len(chunk_texts) if mode != "pg" else len(corpus),
    }


def format_report(report: dict) -> str:
    """Render a compact human-readable metrics table."""
    lines = [
        (
            f"BEIR benchmark: dataset={report['dataset']} "
            f"split={report['split']} mode={report['mode']}"
        ),
        (
            f"corpus docs={report['num_corpus']} chunks={report['num_chunks']} "
            f"queries={report['num_queries']} top_k={report['top_k']}"
        ),
        "",
        f"{'k':>6} {'NDCG':>8} {'MAP':>8} {'Recall':>8} {'P@k':>8}",
    ]
    for k in sorted(report["k_values"]):
        m = report["metrics"][k]
        lines.append(
            f"{k:>6} {m['ndcg']:>8.4f} {m['map']:>8.4f} "
            f"{m['recall']:>8.4f} {m['precision']:>8.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the RAG retriever on a BEIR collection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="scifact",
                        help=f"BEIR dataset ({'|'.join(SUPPORTED_DATASETS)})")
    parser.add_argument("--split", default="test", help="BEIR split (test|dev)")
    parser.add_argument("--data-dir", default="data/beir",
                        help="Cache dir for BEIR downloads")
    parser.add_argument("--mode", default="bm25",
                        choices=["bm25", "dense", "hybrid", "pg"],
                        help="bm25: lexical (default, mirrors qa.retrieve). "
                             "dense: HF embeddings + cosine (mirrors pgvector). "
                             "hybrid: RRF fusion. pg: live Postgres via qa.retrieve.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Documents retrieved per query")
    parser.add_argument("--metrics-k", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Cutoffs to report (e.g. --metrics-k 5 10)")
    parser.add_argument("--max-corpus", type=int, default=None,
                        help="Cap corpus docs (deterministic, for quick runs)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Cap queries (deterministic, for quick runs)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Embedding batch size (dense/hybrid)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Chunk long docs with the production splitter "
                             "(0 = one text per BEIR doc)")
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant")
    parser.add_argument("--output", default=None,
                        help="Write JSON report to this path")
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        dataset=args.dataset,
        split=args.split,
        data_dir=args.data_dir,
        mode=args.mode,
        top_k=args.top_k,
        k_values=args.metrics_k,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k1=args.k1,
        b=args.b,
        rrf_k=args.rrf_k,
    )
    print(format_report(report))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.output}")
    return report


if __name__ == "__main__":
    main()
