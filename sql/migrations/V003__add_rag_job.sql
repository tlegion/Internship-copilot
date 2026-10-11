CREATE TABLE chunks_jobs (
    chunk_id TEXT PRIMARY KEY NOT NULL,
    job_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,

    FOREIGN KEY (job_id)
        REFERENCES jobs(id)
        ON DELETE CASCADE,
    
    UNIQUE(job_id, chunk_index)
);

CREATE TABLE chunk_embeddings_jobs (
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding TEXT NOT NULL,
    PRIMARY KEY (
        content_hash,
        embedding_model
    )
);

CREATE VIRTUAL TABLE chunk_fts_jobs USING fts5(
    chunk_id UNINDEXED,
    text
);