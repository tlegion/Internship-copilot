CREATE VIRTUAL TABLE chunk_fts USING fts5(
    chunk_id UNINDEXED,
    text
);

