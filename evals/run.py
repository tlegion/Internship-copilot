"""Explicit CLI: dry-run is the safe default, --live opts into paid requests."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
from unittest.mock import patch

from evals.fixtures import ROOT, digest, load_fixtures
from evals.scenarios import CASES_PATH, load_cases


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    result.add_argument(
        "--case", action="append", dest="case_ids", help="Select an exact case ID (repeatable)"
    )
    result.add_argument("--max-cases", type=int, default=40)
    result.add_argument("--max-api-requests", type=int, default=150)
    result.add_argument("--grader-model", default="gpt-6-luna")
    result.add_argument(
        "--no-grader", action="store_true",
        help="Python-only evaluation; semantic accuracy ungraded",
    )
    result.add_argument(
        "--reuse-snapshot", type=Path,
        help="Reuse a verified prepared.sqlite from a previous evals/runs directory",
    )
    return result


def configuration(args):
    if args.max_cases < 1 or args.max_api_requests < 1:
        raise ValueError("Case and API limits must be positive")
    fixtures = load_fixtures()
    cases = load_cases(args.suite, fixtures)
    if args.case_ids:
        unknown = set(args.case_ids) - {case.id for case in cases}
        if unknown:
            raise ValueError(f"Unknown case IDs for suite: {sorted(unknown)}")
        cases = [case for case in cases if case.id in args.case_ids]
    if len(cases) > args.max_cases:
        raise ValueError(
            f"Selected {len(cases)} cases exceeds --max-cases {args.max_cases}; "
            "select fewer explicitly"
        )
    if args.reuse_snapshot:
        path = args.reuse_snapshot.resolve(strict=True)
        runs_directory = (ROOT / "evals/runs").resolve()
        if not path.is_relative_to(runs_directory) or path.name != "prepared.sqlite":
            raise ValueError("Reusable snapshots must be evals/runs/<run>/prepared.sqlite")
        args.reuse_snapshot = path
    return fixtures, cases


def snapshot_metadata(fixtures):
    from scripts.migrate_db import MIGRATIONS

    paths = [
        *MIGRATIONS, ROOT / "rag_lab/chunking.py", ROOT / "rag_lab/job_chunking.py",
        ROOT / "evals/isolation.py",
    ]
    return {
        "fixtures": fixtures["hashes"],
        "implementation": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        "embedding_model": "text-embedding-3-small",
    }


def reuse_snapshot(sandbox, source, fixtures):
    from evals.isolation import SafetyError

    expected = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    same_configuration = expected.get("configuration") == snapshot_metadata(fixtures)
    if not same_configuration or expected.get("sha256") != digest(source):
        raise SafetyError("Prepared snapshot fingerprint/configuration mismatch")
    shutil.copyfile(source, sandbox.snapshot)
    sandbox.validate_records(sandbox.snapshot, fixtures)
    sandbox.validate_indexes(sandbox.snapshot, fixtures)


def save_snapshot(sandbox, log, fixtures):
    target = log.directory / "prepared.sqlite"
    shutil.copyfile(sandbox.snapshot, target)
    log.write("prepared.json", {
        "configuration": snapshot_metadata(fixtures), "sha256": digest(target),
    })


def scripted_execution(case, log):
    import loop
    from models import UserInput, ModelResponse, ToolOutput
    from evals.isolation import SafetyError

    prompts = iter(case.prompts)

    def reader(round):
        prompt = next(prompts, None)
        return UserInput(prompt, None, round) if prompt is not None else None

    def sink(item):
        if isinstance(item, UserInput):
            log.event("user", text=item.text, round=item.round)
        elif isinstance(item, ModelResponse):
            calls = []
            for call in item.tool_calls:
                arguments = call.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        arguments = {"invalid_json": arguments}
                calls.append({
                    "tool": call.tool_name, "arguments": arguments, "call_id": call.call_id,
                })
            log.event(
                "assistant", text=item.text, round=item.round,
                status=item.status, tool_calls=calls,
            )
        elif isinstance(item, ToolOutput):
            log.event(
                "tool_output", tool=item.tool_name, call_id=item.call_id,
                round=item.round, step=item.step, output=item.output,
            )
            if not item.output.get("success"):
                log.event(
                    "tool_failure", tool=item.tool_name, call_id=item.call_id, output=item.output,
                )
        else:
            log.event("runtime_error", error=item)

    original = loop.tool_loop
    used = 0

    def bounded_tools(calls, *args, **kwargs):
        nonlocal used
        if used + len(calls) > case.max_tool_calls:
            raise SafetyError("Scenario total Tool call limit exceeded")
        used += len(calls)
        return original(calls, *args, **kwargs)

    with patch.object(loop, "tool_loop", bounded_tools):
        return loop.conversation_loop(
            max_round=len(case.prompts), max_step=case.max_step,
            max_tool_call=case.max_tool_calls, input_reader=reader, event_sink=sink,
        )


def _grader_payload(case, fixtures, events, checks):
    # No oracle/expected data reaches the production conversation.
    evidence = [
        row for row in fixtures["evidence"]
        if row["evidence_id"] in case.evidence_refs or row["candidate_id"] == case.candidate_id
    ]
    event_kinds = {
        "user", "assistant", "tool_output", "tool_failure", "approval", "approval_unexpected",
        "checker_validation_failure", "runtime_error", "scenario_error", "fault_injected",
    }
    api_fields = ("kind", "api", "request_number", "status_code", "usage", "latency_seconds")
    return {
        "scenario": case.model_dump(),
        "originals": {
            "evidence": evidence,
            "jobs": [row for row in fixtures["jobs"] if row["id"] in case.job_refs],
        },
        "scripted_conversation": case.prompts,
        "events": [event for event in events if event["kind"] in event_kinds],
        "api_observations": [
            {key: event.get(key) for key in api_fields}
            for event in events if event["kind"].startswith("api_")
        ],
        "assertions": checks,
    }


def run_case(case, sandbox, fixtures, log, args):
    from evals.assertions import assertions
    from evals.grader import grade, overall
    from evals.instrumentation import Approvals, observe_checker
    from evals.recording import BudgetExceeded, summarize_database_changes

    log.scenario = case.id
    log.event("scenario_started", description=case.description)
    start = len(log.events)
    path = sandbox.reset(case.id, fixtures)
    if case.fault == "stale_job_index":
        with sandbox.bind(path), closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("DELETE FROM chunks_jobs")
        log.event("fault_injected", fault=case.fault)
    before = sandbox.state(path)
    approvals = Approvals(case.approvals, log)
    interruption = None
    try:
        with sandbox.bind(path), approvals.install(), observe_checker(log):
            scripted_execution(case, log)
    except BaseException as error:
        log.event("scenario_error", error_type=type(error).__name__, error=str(error))
        if isinstance(error, (KeyboardInterrupt, SystemExit, BudgetExceeded)):
            interruption = error
    after = sandbox.state(path)
    log.event("database_state", format_version=2, tables=summarize_database_changes(before, after))
    events = log.events[start:]
    checks = assertions(case, events, before, after, fixtures, approvals)
    log.event("assertion_results", results=checks)
    grading = None
    if interruption is None and not args.no_grader:
        payload = _grader_payload(case, fixtures, events, checks)
        log.event("grader_started", model=args.grader_model)
        try:
            grading = grade(payload, args.grader_model)
            log.event("grader_result", result=grading)
        except (BudgetExceeded, KeyboardInterrupt, SystemExit) as error:
            grading = {
                "status": "GRADER_ERROR", "error_type": type(error).__name__, "error": str(error),
            }
            log.event("grader_result", result=grading)
            interruption = error
    status = "ABORTED" if interruption else overall(checks, grading)
    result = {
        "id": case.id, "status": status, "assertions": checks, "grading": grading,
        "semantic_grading": not args.no_grader,
    }
    log.event("scenario_finished", status=status)
    log.scenario = None
    return result, interruption


def live(args, fixtures, cases):
    from evals.grader import RUBRIC_HASH
    from evals.instrumentation import observe_apis
    from evals.isolation import Sandbox
    from evals.recording import RunLog, RequestBudget, BudgetExceeded

    # Create persistent reports before any setup/provider work can fail.
    metadata = {
        "suite": args.suite,
        "cases": [case.id for case in cases],
        "fixture_hashes": fixtures["hashes"],
        "case_hash": digest(CASES_PATH),
        "rubric_hash": RUBRIC_HASH,
        "models": {
            "main_agent": "gpt-6-luna", "checker": "gpt-6-luna", "extraction": "gpt-6-luna",
            "reranker": "gpt-6-luna", "embedding": "text-embedding-3-small",
            "grader": None if args.no_grader else args.grader_model, "reasoning_effort": "none",
            "main_max_output_tokens": 1024, "checker_max_output_tokens": 4096,
            "checker_max_attempts": 3, "store": False,
        },
        "max_cases": args.max_cases,
        "max_api_requests": args.max_api_requests,
        "reused_snapshot": str(args.reuse_snapshot) if args.reuse_snapshot else None,
    }
    log = RunLog(metadata)
    budget = RequestBudget(args.max_api_requests)
    results = []
    status, error = "ERROR", None
    sandbox = None
    try:
        sandbox = Sandbox()
        with observe_apis(log, budget):
            log.event("preparation_started")
            if args.reuse_snapshot:
                reuse_snapshot(sandbox, args.reuse_snapshot, fixtures)
            else:
                sandbox.create(fixtures)
                sandbox.prepare_indexes(fixtures)
            save_snapshot(sandbox, log, fixtures)
            log.event("preparation_validated", snapshot_sha256=digest(sandbox.snapshot))
            for case in cases:
                result, interruption = run_case(case, sandbox, fixtures, log, args)
                results.append(result)
                log.write("summary.json", {
                    "status": "RUNNING", "results": results, "api_requests": budget.used,
                })
                if interruption:
                    raise interruption
            statuses = {item["status"] for item in results}
            status = next(
                (name for name in ("FAIL", "GRADER_ERROR", "REVIEW") if name in statuses), "PASS"
            )
    except BaseException as caught:
        interrupted = isinstance(caught, (KeyboardInterrupt, SystemExit, BudgetExceeded))
        status = "ABORTED" if interrupted else "ERROR"
        error = {"type": type(caught).__name__, "message": str(caught)}
        log.event("run_error", **error)
    finally:
        if sandbox:
            sandbox.close()
        summary = {
            "status": status, "results": results, "api_requests": budget.used,
            "error": error, "semantic_grading": not args.no_grader,
        }
        log.finish(summary)
    print(json.dumps({
        "status": status, "report": str(log.directory), "api_requests": budget.used,
    }))
    return 0 if status == "PASS" else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        fixtures, cases = configuration(args)
        if args.live:
            return live(args, fixtures, cases)
        from evals.isolation import Sandbox

        sandbox = Sandbox()
        try:
            if args.reuse_snapshot:
                reuse_snapshot(sandbox, args.reuse_snapshot, fixtures)
            else:
                sandbox.create(fixtures)
            indexes = (
                "verified cached snapshot" if args.reuse_snapshot
                else "deferred to explicit live preparation"
            )
            print(json.dumps({
                "status": "DRY_RUN", "cases": [case.id for case in cases],
                "fixtures": fixtures["hashes"], "database": "isolated migrations/records verified",
                "indexes": indexes, "paid_api_requests": 0,
            }))
        finally:
            sandbox.close()
        return 0
    except Exception as error:
        print(json.dumps({
            "status": "CONFIG_ERROR", "type": type(error).__name__, "error": str(error),
        }))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
