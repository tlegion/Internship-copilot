# Synthetic live evaluations

Architecture: scripted user turns → the production Main Agent → the real Tool
Registry/Skills/handlers → hard Python checks → one optional independent semantic
grading pass → durable local reports. The runner never selects a Tool on the
Agent's behalf. Expected results and grading instructions are not included in the
Agent conversation. No new agent framework or dependencies are required.

## Run

Use the repository virtual environment (or an environment with requirements.txt):

```bash
.venv/bin/python -m evals.run --suite smoke --dry-run
.venv/bin/python -m evals.run --suite full --dry-run
.venv/bin/python -m evals.run --suite smoke --live --max-api-requests 150
.venv/bin/python -m evals.run --suite full --live --max-api-requests 600
```

The default is dry-run. Imports, dry-run and deterministic unittest discovery do
not make paid API calls. Live mode requires an OpenAI API key through the existing
environment/.env mechanism. **Implementation validation does not run a live suite.**
The first real run requires the user's explicit authorization or their invocation
of `--live`.

Smoke has seven cases covering all nine executable entry points. Full has 31 cases,
including smoke, semantic distinctions, multi-turn context, injection, approvals,
identity isolation, errors and a deliberately stale Job index. Select fewer cases:

```bash
.venv/bin/python -m evals.run --suite full --case staging_negative --live --max-cases 1 --max-api-requests 30
```

`--max-cases` rejects oversized selections; it never silently skips cases.
`--max-api-requests` counts **actual HTTP attempts**, including setup embeddings,
Main Agent, Extraction, Checker retries, both rerankers, Grader and SDK retries.
Budget exhaustion aborts the run; it cannot produce a successful evaluation.
Each case also declares bounded turns, tool batches (`max_step`) and total Tool
calls. Timeouts/retries remain the production settings; this is a request limit,
not a token/dollar limit. Responses with usage expose token counts in traces;
missing measurements are left unavailable. Model prices and real costs are not
estimated. `--grader-model` defaults to `gpt-6-luna`; Agent/Checker settings remain
unchanged. `--no-grader` marks results as Python-only, with semantic accuracy
ungraded. A Python-only PASS does not establish semantic correctness.

## Database safety and preparation

The committed `fixtures/test_1/{evidence,jobs}.json` records are loaded verbatim into
a fresh private SQLite database created from the existing four migrations. A small
separate `fixtures/isolation/evidence.json` adds only the explicitly synthetic
`synthetic_other` identity. It does not change the test_1 source data.

The runner never calls fixed-path migration/seed entrypoints and never opens
`data/internship.db`. During business execution, a scoped DB_PATH override and
SQLite connection guard reject paths outside the private sandbox and reject loss
of the override. SQLite URI paths and symlinks are resolved before checking. There
is no fallback. This synchronous, process-local instrumentation must run in its
own process, without a concurrent CLI conversation or parallel Eval in that process.

Explicit live preparation uses the existing Evidence and Job index synchronizers.
Record equality (including IDs, ownership, nulls and every field), schema version,
integrity, chunk text/hashes, FTS content/integrity and embedding presence, numeric
validity and dimensions are checked. Missing, incomplete, corrupt and stale indexes
fail before any evaluated scenario. The stale-index case explicitly injects and
logs a fault **after** successful validation to test error reporting. Index setup
is never added to ordinary conversation startup. Dry-run checks fixture/case
contracts, paths, migrations and record equality; new embeddings/indexes are
explicitly deferred, not falsely declared validated.

Every case receives a separate verified copy of the clean prepared snapshot,
including read-only cases. Writes cannot contaminate subsequent cases. Whole
database table state is compared before/after: only exact expected Job insertions
are allowed; existing Jobs and every other table must remain unchanged. Both
initial and similar-Job approval boundaries use declared boolean queues. Unexpected
approvals and any terminal input fail closed; all arguments/decisions are traced.

A completed preparation saves its synthetic snapshot alongside its report. Reuse
avoids paying again for identical fixture embeddings:

```bash
.venv/bin/python -m evals.run --suite smoke --live --reuse-snapshot evals/runs/<run>/prepared.sqlite
```

Only snapshots under `evals/runs/` are accepted. Their SHA-256, fixture/migration/
chunker/validator fingerprints, complete records and indexes are revalidated before
reuse. Never substitute the real internship database. Snapshots and raw reports
are ignored by Git. Temporary per-case databases are cleaned up independently of
permanent reports.

## Hard checks and semantic grading

Python checks actual tool trajectories, Skill activation in an earlier batch,
arguments, result/error contracts, verdicts, citation ownership/exact text/metadata,
expected API paths, approval decisions and database deltas. It does not pretend
that an exact substring proves the quotation's meaning. The Grader independently
checks final answers, scope, chronology, project classification, semantic citation
roles, hidden qualifications and honest handling of errors. Its structured output
is a transport contract, not proof of a reliable evaluation.

A hard check failure always makes the case FAIL, regardless of a positive grade.
PASS means the stated checks and enabled grading passed. FAIL means a material
violation. REVIEW means genuine ambiguity in the references/rubric requiring human
adjudication; it is not an escape from difficult reasoning. A correct factual
`insufficient_evidence` normally merits PASS. Grader API errors, refusals,
incomplete or invalid responses are GRADER_ERROR, never REVIEW. Grading uses one
independent structured pass with no tools or database access. Grader payloads
include originals, ground truth, scripted turns, actual answers/tool events,
approvals and hard checks; redundant raw API bodies/vectors stay in the durable
trace rather than inflating the grading context.

## Reports and failures

Each live run creates `evals/runs/<UTC>_<git-sha>_<run-id>/` before preparation:

- `manifest.json`: UTC times, run ID, branch/full SHA, dirty status/content
  fingerprint, fixture/case/rubric hashes, model settings, selection and limits.
- `trace.jsonl`: append-only, flushed/fsynced structured events, real request and
  response bodies, Tool calls/results/errors, approvals, validation failures,
  API latency and usage when supplied by the SDK. Headers and keys are excluded
  or redacted. Database-state events use `format_version: 2`: each table has
  before/after row counts, presence flags and SHA-256 hashes of its complete
  contents. Identifiable changed rows include IDs and relevant field changes;
  embeddings, blobs and long values use hashes/lengths instead of full repeated
  values (long text also has a short preview). Full states stay in memory for
  exact assertions. Only synthetic fixture data belongs in live evaluations.
- `summary.json`: full scenario results, including every assertion and grade,
  atomically replaced after each case and finalized on failure or interruption.
- `prepared.sqlite` and `prepared.json`: reusable, synthetic prepared snapshot and
  fingerprints, when preparation completed.

New completed preparations produce these five files. Separate `assertions.json`
and `grades.json` are no longer generated. Historical run directories are left
untouched: their existing summaries and optional duplicate report files remain
readable, and older `database_state` events retain full `before`/`after` payloads.
Table hashes use UTF-8 JSON with sorted keys, compact separators and rows sorted
by that same canonical representation; duplicate rows still contribute to the hash.
Logging changes do not alter snapshot fingerprints or validation, so previously
valid prepared snapshots remain reusable.

Inspect the first failed assertion, its Tool output and matching request number,
then the independent grade. `api` distinguishes Main Agent, embeddings, reranking,
Extraction, Checker and Grader. Checker attempts are individual requests and Python
validation failures have explicit events. A hard process kill may leave finish time
unset and summary RUNNING/empty; the fsynced trace still identifies the last step.
KeyboardInterrupt and budget exhaustion are recorded as ABORTED. Non-PASS runs exit
nonzero. A dirty fingerprint identifies uncommitted contents but does not archive
those sources: retain your worktree/patch when reproducing such a run.

## Extend and validate

Add strictly typed synthetic records under fixtures, extend the explicit fixture
loader/ownership checks when adding a dataset, and author cases in
`cases/test_1.json`. Keep prompts separate from expected outputs and ground truth;
reference original IDs and use deterministic expectations only for unambiguous
facts. `equals`/`contains` use dotted output paths (`*` maps a list). Approval queues
must match every boundary. New cases must not weaken existing unittests.

```bash
.venv/bin/python -m unittest tests.test_evals
.venv/bin/python -m unittest discover -s tests
git diff --check
```

On macOS, an existing Skill test compares a resolved `/private/var/...` path with an
unresolved `/var/...` temporary path. For a canonical-path test run without changing
the tokenizer cache location, import the existing tokenizer first, then set
`tempfile.tempdir = str(Path(tempfile.gettempdir()).resolve())` before discovery.

Known limits: live models, retrieval and semantic grading are nondeterministic;
reports make runs inspectable rather than promising identical completions. The
fixture is small and cannot establish broad product quality. Exact Tool counts are
intentional trajectory expectations, not a guarantee of all reasonable strategies.
The harness validates cached embeddings structurally and by snapshot fingerprint;
it cannot prove their semantic quality without retrieval/real-model evaluation.
No real API behavior, model availability, costs or end-to-end quality is verified
until an explicitly authorized live run completes.
