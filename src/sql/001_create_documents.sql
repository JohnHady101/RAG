CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(1024)
);

SELECT * FROM documents;

-- explain analyze SELECT * FROM documents where id = 120;
-- DROP TABLE IF EXISTS documents;