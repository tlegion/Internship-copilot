# test_1 — Synthetic E2E Evaluation Fixture

This fixture is deliberately fictional. `test_1` replaces the earlier proposed `candidate_002` name because it is a testing identity, not a real candidate. It can support evidence retrieval, claim verification, profile building, skill gating, job browsing/search, and isolated job-write/approval tests.

## Contents
- `evidence.json`: 12 original Evidence records, using the same field names and list shape as `rag_lab/data/evidence.json`. Every `candidate_id` is `test_1`.
- `jobs.json`: 8 synthetic Jobs with `id`, `title`, `company`, `description`, `location` fields, matching the SQLite jobs table.
- Job 001 and Job 008 are near-duplicates that differ **only by location**, deliberately exercising `check_similar()`. Job 003 has a SQL NULL location, not "Remote".

## Evidence scenarios
- Clear academic FastAPI project; staging-only release RC-17
- Historical SQL beginner state and later demonstrated SQL work
- Increasing retry delays with **unspecified** algorithm (not proven exponential)
- 95% confidence interval versus 72% classifier accuracy
- Provisional RUN-42 count versus authoritative final correction
- Equally credible contradictory snapshots from the same moment
- Mock AWS exercise versus real production work
- Prompt-injection text quoted as **untrusted source data**

## Intended loading and safety contract (not implemented in this commit)
1. These files are **static fixtures only**. Do not run `scripts/seed_rag.py` or `scripts/init_db.py` against the user's existing `data/internship.db` to load them.
2. An eventual eval setup should create a fresh isolated SQLite database **for each scenario or clean run**, apply existing migrations, load these fixtures, build both evidence and job retrieval indexes, and verify the resulting state before invoking the main Agent or any live API evaluation.
3. The existing app has a hard-coded `storages.database.DB_PATH`; the eval setup must safely redirect or inject the test DB path **before** any tool can run. Prefer a fail-closed guard so tests cannot silently fall back to the real DB.
4. Compare DB rows to fixture values (not just row counts), check unique IDs and candidate ownership, and validate both index content and embedding presence. Do not equate "expected table state" with "expected row count" alone.
5. Isolate and deterministically mock human approvals in write scenarios. If a test creates a Job, discard its scenario DB afterward or intentionally rebuild/sync its job index before testing search freshness.
6. Treat regeneration of embeddings and index sync as explicit **eval preparation**, not a hidden action in normal `loop.py` conversation initialization. Real embedding calls may incur API cost.
7. Keep scenario questions, gold labels and grader prompts **separate** from the seed; the Agent must not see expected answers.
8. Permanent live-eval traces and run metadata (including Git HEAD and dirty status) belong in a separate append-only log location, not this fixture directory.

No business database rows, embeddings, index tables, or runtime code are changed by adding these fixture files.
