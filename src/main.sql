-- 1. Create a table with a vector column (e.g., 3 dimensions)

-- CREATE TABLE items (
--     id bigserial PRIMARY KEY,
--     content text
-- );

-- 2. Insert some sample data
INSERT INTO items (content) VALUES
    ('apple'),
    ('banana'),
    ('cherry');

SELECT * from items;
-- -- 3. Find the 2 closest items to a specific vector using Cosine Distance (<=>)
-- SELECT content, embedding <=> '[3, 1, 2]' AS distance
-- FROM items
-- ORDER BY distance
-- LIMIT 2;