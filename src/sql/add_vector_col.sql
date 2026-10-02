ALTER TABLE documents
ADD COLUMN IF NOT EXISTS content_tsv tsvector
GENERATED ALWAYS AS (
    to_tsvector('english', content)
) STORED;

CREATE INDEX IF NOT EXISTS documents_content_tsv_idx
ON documents
USING GIN (content_tsv);