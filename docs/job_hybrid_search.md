# Job hybrid search (Stage 3)

`list_jobs` browses stored records with exact filters and pagination;
`lookup_job` fetches a known ID; `search_jobs_hybrid` ranks distinct jobs by
natural-language role preferences. Its rank expresses retrieval relevance,
not candidate fit. Search is read-only and requires no tool approval.

## Manual smoke test

Run these commands from the repository root, using the existing Python environment:

```powershell
.\.venv\Scripts\python.exe scripts/migrate_db.py
.\.venv\Scripts\python.exe scripts/sync_job_index.py
.\.venv\Scripts\python.exe loop.py
```

Migration registers the existing V003 as database version 4. Historical SQL is
unchanged. If V003 was applied manually without updating `user_version`, verify
that version bookkeeping before running the migration command.

Synchronization reports jobs, chunks, reused cached embeddings and new embeddings.
It writes only the Job index tables. Missing embeddings require the configured
`OPENAI_API_KEY` and incur embedding API usage. These commands have not been run
against the actual dataset during implementation.

Ask the Agent:

> Find up to 10 internship roles involving Python, LLM APIs, RAG, and backend engineering.

Then try the same request restricted to an existing exact company and location.
Inspect that each returned job appears once, includes its full description and
has a unique rank starting at 1. Running synchronization again without changes
should report zero new embeddings. After creating/editing/deleting a job, search
should report a stale index until synchronization is rerun.

## Index and freshness policy

Documents include quoted title, company and location plus the original description;
Job IDs appear only in chunk identity. Chunks are bounded to 800 tokens, with
approximately 50 tokens of description overlap and paragraph/sentence boundaries
when possible. Long descriptions repeat metadata; unusually large metadata is
split with the whole document instead. Exact chunk texts have SHA-256 hashes.

Chunking follows the Evidence chunker's paragraph/sentence grouping and token
splitting. Job text keeps its original whitespace so freshness checks remain
sensitive to description edits. Token slices back up only when they end inside
a Unicode character. Overlap is simply the previous chunk's last approximately
50 tokens; no binary boundary search is used.

Synchronization generates one vector per uncached unique hash using
`text-embedding-3-small`. It preserves the separate Job embedding cache. Chunk,
FTS and cache writes commit together. The transaction rechecks its source job
snapshot, declining the rebuild if jobs changed while embeddings were generated.

Every search recomputes expected chunks for all current jobs and compares full
chunk identities, indices, texts and hashes, FTS rows, and embedding availability.
An inconsistent index returns `ToolError` instructing the developer to run
`python scripts/sync_job_index.py`; missing tables return `StorageError` with
migration and synchronization instructions. This is a global check, even if the
requested filters would match no jobs. An empty, consistent database returns
`{"jobs": []}`. Invalid cached vectors cause `StorageError`; manually corrupted
cache entries need repair/removal before sync can regenerate them.

Freshness verification and both SQL retrieval branches share a read transaction,
so results represent one consistent database snapshot. Concurrent changes after
that snapshot are visible to the next search. `create_job` performs no indexing
and retains its original successful insert and two-layer approval contracts.

## Ranking and limits

Parameterized exact company/location filters restrict both branches. Dense
retrieval generates one query vector and scans eligible chunk vectors with the
existing cosine helper. FTS5 uses quoted keyword phrases joined with OR and orders
by ascending raw `bm25()` scores; empty keywords skip that branch.

RRF uses the Evidence weights (dense 0.8, BM25 0.2) and its initial rank constant
of zero, by calling the existing `reciprocal_rank_fusion()` directly. Internal
rows retain branch ranks/scores and fusion scores. All eligible
chunks are retained before fusion and deduplication. The first fused appearance
of each job determines its rank; `top_k` counts distinct jobs (default 10, maximum
20). Joined original job metadata supplies full descriptions without per-job SQL
queries. The model receives only job fields and rank.

This MVP performs global chunk verification and exact vector scanning on each
search; CPU/memory cost grows with the dataset. Synchronization sends missing
texts through the existing embedding helper in one batch, so very large indexes
may need future provider-aware batching. No relevance cutoff or LLM reranker is
implemented. No new indexes or migration tables are necessary for this scale.
Cache lookup uses the same parameterized `IN` query as Evidence indexing, in
500-hash batches to stay below SQLite's bound-parameter limit.

## Automated checks

```powershell
$env:TIKTOKEN_CACHE_DIR = "$PWD\.cache\tiktoken" # if using the existing local tokenizer cache
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary databases and mocked embeddings; no paid API calls or real
dataset mutations occur. Tokenizer initialization needs its ordinary cached
encoding data (or an initial public tokenizer download).
