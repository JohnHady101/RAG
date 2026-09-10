CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(3072)
);

