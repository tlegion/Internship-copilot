# Function-derived tool schemas

`tool_call.py` uses the official SDK's
[`function_schema()` helper](https://openai.github.io/openai-agents-python/ref/function_schema/).
Only schema generation is reused; execution still uses the existing Responses
client, `run_tool()`, and original business handlers.

## Changes

- `tools/jobs.py`, `tools/evidence.py`: existing descriptions moved to Google-style
  docstrings. Chinese descriptions are translated; existing English wording is
  preserved. Nullable limit annotations now reflect existing behavior, and their
  existing numeric bounds are also expressed as Pydantic constraints.
- `tool_call.py`: removed all five handwritten base parameter schemas and the
  handwritten `load_skill` / `read_skill_resource` parameter schemas.
  `SKILL_TOOL_SCHEMAS` is removed; trusted Skill handlers generate their schemas
  directly. `tools` and `BASE_TOOL_SCHEMAS` remain generated compatibility views;
  the provider uses fresh `get_tool_schemas()` results.
- `provider.py`, `models.py`: malformed JSON and non-object payloads can reach
  dispatcher validation and return a Tool Error instead of crashing decoding.
- `requirements.txt`: explicitly pins `openai-agents==0.24.0` and
  `pydantic==2.13.5`, matching the installed, tested environment. Other dependency
  declarations are unchanged. This is a direct dependency manifest, not a full
  transitive lockfile.
- Tests: existing schema assertions and test handlers follow generated contracts;
  `test_function_schemas.py` adds validation and signature-change coverage.

## Contract and execution

The Python signature, type hints, defaults, and docstring generate a Pydantic
model and strict JSON Schema. No second parameter definition is maintained.
Models reject unknown arguments and use strict validation to avoid coercing
strings, booleans, or floats into integers. `to_call_args()` preserves Python
values, including nested models and `None`.

Dispatch order is availability, validation, approval, then the original handler.
Validation failures return `success: false`, `error_type: ToolError`, and an
`Invalid arguments` message. Existing business exceptions retain their error
type and arguments. Business checks and both creation approval steps remain.

Python defaults still apply when a defaulted argument is omitted in a direct
dispatcher call. The model-facing strict schema requires every property;
nullable properties accept explicit null. `create_job.location` is nullable but
has no default, so omitting it is rejected before approval.

`BASE_TOOL_LIST` and `SKILL_TOOL_LIST` still control trusted execution and approval.
Session closures expose only Skill names and resource paths, without exposing
catalog or activation state as model arguments. Their Literal annotations derive
the current enum constraints. Single-value Literals are displayed as enums to
preserve the previous schema format. Skill descriptions still come from catalog
metadata. Loading a Skill enables its trusted registrations only in the next
batch, with the existing loop unchanged. Skill files cannot register handlers.

## Validation

82 unit tests passed with socket connections blocked and the existing tokenizer
cache configured. No model API calls were made. `pip check` passed. All base
schemas passed JSON Schema meta-validation. Existing English descriptions were
compared against HEAD and matched after whitespace normalization.

PowerShell regression command (uses the local tokenizer cache):

```powershell
$env:TIKTOKEN_CACHE_DIR = (Resolve-Path .cache/tiktoken).Path
.venv\Scripts\python.exe -m unittest discover -s tests -q
.venv\Scripts\python.exe -m pip check
```

No paid Responses API acceptance test was run. Generated nullable fields use
Pydantic's `anyOf` form rather than the former type arrays. Future dependency
upgrades should rerun the suite because the SDK helper and Pydantic determine
the generated format. Tests that replace trusted handlers must use real typed
functions or preserve the wrapped function's signature and annotations.

## Generated lookup_job schema

```json
{
  "type": "function",
  "name": "lookup_job",
  "description": "Retrieve job details by job ID.",
  "parameters": {
    "properties": {
      "job_id": {
        "description": "Job ID, for example job_001",
        "title": "Job Id",
        "type": "string"
      }
    },
    "required": ["job_id"],
    "title": "lookup_job_args",
    "type": "object",
    "additionalProperties": false
  },
  "strict": true
}
```
