-- BM25-style full-text search over documents.content_tsv.
-- Params (psycopg2, in order): query text, LIMIT.
SELECT
    id,
    content,
    metadata,
    ts_rank(content_tsv, query) AS score
FROM documents,
     websearch_to_tsquery('english', %s) AS query
WHERE content_tsv @@ query
ORDER BY score DESC
LIMIT %s;