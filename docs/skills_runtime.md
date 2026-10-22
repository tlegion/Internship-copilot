# Skills Runtime (Stage 3)

The runtime follows the [Agent Skills format](https://agentskills.io/specification):
one directory per Skill, YAML frontmatter in `SKILL.md`, and optional supporting
resources. `name` and `description` are required; names must match their directory.
Optional frontmatter is preserved without granting tool authorization.

## Call flow

1. `conversation_loop()` discovers `skills/*/SKILL.md` metadata once and creates
   a local `loaded_skills` set. Discovery reads only the frontmatter, not resources
   or the Markdown instructions.
2. `provider.py` asks `tool_call.get_tool_schemas()` for each model request. The
   `load_skill` schema contains the catalog names as an enum and their descriptions.
   The System Prompt does not duplicate the catalog or Skill instructions.
3. `load_skill(name)` returns the complete UTF-8 `SKILL.md` through the existing
   Tool Result and history path. It does not change state itself.
4. `tool_loop()` evaluates every call in a response using the same pre-execution
   loaded-state snapshot. Only successful loads update the session after the batch.
5. The next model request sees newly available tools. `read_skill_resource` is
   visible only after loading a Skill, with an enum of loaded names. It reads text
   relative to that directory and refuses resolved paths outside it.

Base tools remain available with their original schemas, handlers and approval
settings. `SKILL_TOOL_LIST` is an explicit developer-maintained mapping in
`tool_call.py`; schemas are generated from its trusted handlers. The
`candidate-recognition` Skill gates the read-only `build_candidate_profile` and
`verify_candidate_claim` tools. The `job-comparison` Skill uses the existing job
tools. Generic `search_evidence_hybrid` remains a base tool.
Schema visibility and execution both use `get_available_tools()`, which copies
the base registry instead of mutating it. No loaded-state set is module-global.

## Supported and intentionally unsupported features

- Safe YAML discovery, complete instruction loading, and supporting UTF-8 text.
- Standard `name`/`description` validation; optional fields are retained as metadata.
- `allowed-tools` is informational, not executable authorization or approval bypass.
- No Skill script execution, automatic Python imports, shell tool, or script runner.
- No remote Skill fetching, nested Skill discovery, hot reload, binary asset rendering
  or automatic resolution of reference chains. Resources are read on explicit calls.
- Invalid discovered frontmatter produces `ToolError` and stops startup for repair.
  An empty/missing Skills directory leaves only the existing base tools available.

## Manual smoke test

From the repository root, install the requirements in your existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe loop.py
```

Ask: "Load job-comparison, read its comparison guide, then compare job_001 and
job_002 using their original records. Mark missing requirements as unknown."

Inspect the CLI Tool Calls and Outputs: the Skill instructions should arrive via
`load_skill`, the reference via a subsequent `read_skill_resource`, and the jobs
via existing tools. The runtime does not execute any Skill scripts. Use existing
valid Job IDs if those examples are absent. Enter `exit`, restart, and verify that
the Skill must be loaded again before its resources can be read.

The existing per-turn limit is still three tool batches, so an Agent may load the
Skill, read the reference and retrieve jobs before giving its final answer. A
same-response load and resource request is refused until the following request.
The live smoke test requires the existing model/API setup and Job database; it was
documented but not run during implementation.

## Automated tests

```powershell
$env:TIKTOKEN_CACHE_DIR = "$PWD\.cache\tiktoken" # when using the local tokenizer cache
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Skills tests use temporary directories and mocked models/tools. Existing database
tests use temporary SQLite fixtures. No paid API requests or real dataset writes
are needed. PyYAML is the only new dependency.
