CREATE TABLE evidence (
    evidence_id TEXT PRIMARY KEY NOT NULL,
    candidate_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    category TEXT NOT NULL,
    date TEXT,
    text TEXT NOT NULL
);

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY NOT NULL,
    evidence_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,

    FOREIGN KEY (evidence_id)
        REFERENCES evidence(evidence_id)
        ON DELETE CASCADE,

    UNIQUE (evidence_id, chunk_index)
);

CREATE TABLE chunk_embeddings (
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding TEXT NOT NULL,

    PRIMARY KEY (
        content_hash,
        embedding_model
    )
);