from contextlib import redirect_stdout, closing
from copy import deepcopy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx2 as httpx

from evals import run
from evals.assertions import assertions
from evals.fixtures import Job, load_fixtures, ROOT
from evals.grader import EvalGrade, grade, overall
from evals.instrumentation import Approvals, observe_apis, observe_checker
from evals.isolation import Sandbox, SafetyError
from evals.recording import (
    BudgetExceeded, RequestBudget, RunLog, git_metadata, summarize_database_changes,
)
from evals.scenarios import load_cases, Scenario, TOOLS
from models import ModelResponse, ModelToolCall
from storages import database


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = load_fixtures()
        self.sandbox = Sandbox()
        self.addCleanup(self.sandbox.close)
        self.sandbox.create(self.fixtures)
        self.directory = tempfile.TemporaryDirectory(prefix="eval-report-test-")
        self.addCleanup(self.directory.cleanup)
        self.log = RunLog({"suite": "offline-test"}, Path(self.directory.name) / "run")
        self.addCleanup(self.log._stream.close)
        self.cases = {case.id: case for case in load_cases("full", self.fixtures)}
        self.args = run.parser().parse_args(["--no-grader"])

    def prepare(self):
        def vectors(texts):
            return [[1.0] + [0.0] * 1535 for _ in texts]

        with (
            patch("rag_lab.indexing.embedding_texts", side_effect=vectors),
            patch("rag_lab.job_indexing.embedding_texts", side_effect=vectors),
        ):
            self.sandbox.prepare_indexes(self.fixtures)

    def model(self, name=None, arguments=None, text=None):
        calls = [ModelToolCall(name, arguments or {}, "call-" + name)] if name else []
        return ModelResponse(text, calls, "completed", SimpleNamespace(output=[]), 1)

    def test_fixture_contracts_and_source_ownership(self):
        self.assertEqual(len(self.fixtures["jobs"]), 8)
        self.assertEqual(len(self.fixtures["evidence"]), 13)
        candidates = {row["candidate_id"] for row in self.fixtures["evidence"]}
        self.assertEqual(candidates, {"test_1", "synthetic_other"})
        self.assertIsNone(self.fixtures["jobs"][2]["location"])
        for invalid in (
            {**self.fixtures["jobs"][0], "extra": "bad"},
            {**self.fixtures["jobs"][0], "location": 3},
        ):
            with self.assertRaises(ValueError):
                Job.model_validate(invalid)

    def test_all_cases_validate_and_smoke_covers_nine_tools(self):
        smoke = load_cases("smoke", self.fixtures)
        self.assertEqual(len(smoke), 7)
        self.assertEqual(len(self.cases), 31)
        self.assertEqual({item.tool for case in smoke for item in case.expected}, TOOLS)
        self.assertTrue(all(case.ground_truth and case.prompts for case in self.cases.values()))
        with self.assertRaises(ValueError):
            Scenario.model_validate({**smoke[0].model_dump(), "id": "../escape"})

    def test_migrations_records_and_no_implicit_creation(self):
        self.sandbox.validate_records(self.sandbox.snapshot, self.fixtures)
        with (
            self.sandbox.bind(self.sandbox.snapshot),
            closing(sqlite3.connect(self.sandbox.snapshot)) as connection,
            connection,
        ):
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 4)
            connection.execute(
                "UPDATE evidence SET candidate_id='wrong' WHERE evidence_id=?",
                (self.fixtures["evidence"][0]["evidence_id"],),
            )
        with self.assertRaisesRegex(SafetyError, "differs"):
            self.sandbox.validate_records(self.sandbox.snapshot, self.fixtures)
        with self.assertRaises(SafetyError):
            with self.sandbox.bind(self.sandbox.root / "missing.sqlite"):
                pass
        with self.assertRaises(FileExistsError):
            self.sandbox.create(self.fixtures)

    def test_sqlite_guard_fails_closed_for_paths_uris_symlinks_and_lost_override(self):
        previous = database.DB_PATH
        outside = Path(self.directory.name) / "outside.sqlite"
        with self.sandbox.bind(self.sandbox.snapshot):
            for target in (
                outside, f"file:{outside}?mode=rw", ":memory:", ROOT / "data/internship.db",
            ):
                with self.subTest(target=target), self.assertRaises(SafetyError):
                    sqlite3.connect(target, uri=str(target).startswith("file:"))
            link = self.sandbox.root / "escape.sqlite"
            link.symlink_to(outside)
            with self.assertRaises(SafetyError):
                sqlite3.connect(link)
            with patch.object(database, "DB_PATH", outside):
                with self.assertRaises(SafetyError):
                    database.db_connection_ro()
        self.assertEqual(database.DB_PATH, previous)
        self.assertFalse(outside.exists())

    def test_index_preparation_and_scenario_reset_are_isolated(self):
        self.prepare()
        first = self.sandbox.reset("first", self.fixtures)
        with self.sandbox.bind(first):
            database.insert_job("job_009", "A", "B", "C", None)
        second = self.sandbox.reset("second", self.fixtures)
        self.sandbox.validate_records(second, self.fixtures)
        with self.sandbox.bind(second):
            self.assertIsNone(database.get_job("job_009"))
        self.assertEqual(len(self.sandbox.state(first)["jobs"]), 9)
        self.assertEqual(len(self.sandbox.state(second)["jobs"]), 8)

    def test_missing_incomplete_stale_fts_and_corrupt_embeddings_rejected(self):
        self.prepare()
        faults = [
            "DELETE FROM chunks", "DELETE FROM chunk_fts", "DELETE FROM chunk_embeddings",
            "UPDATE chunks SET text='stale'", "DELETE FROM chunks_jobs",
            "DELETE FROM chunk_fts_jobs", "DELETE FROM chunk_embeddings_jobs",
            "UPDATE chunk_embeddings SET embedding='[0]'",
            "UPDATE chunk_embeddings_jobs SET embedding='invalid'",
            "UPDATE chunk_embeddings SET dimensions=4",
        ]
        for index, statement in enumerate(faults):
            with self.subTest(statement=statement):
                path = self.sandbox.reset(f"fault{index}", self.fixtures)
                with (
                    self.sandbox.bind(path),
                    closing(sqlite3.connect(path)) as connection, connection,
                ):
                    connection.execute(statement)
                with self.assertRaises((SafetyError, sqlite3.DatabaseError)):
                    self.sandbox.validate_indexes(path, self.fixtures)

    def test_snapshot_reuse_validates_hash_and_every_record(self):
        self.prepare()
        run.save_snapshot(self.sandbox, self.log, self.fixtures)
        other = Sandbox()
        self.addCleanup(other.close)
        source = self.log.directory / "prepared.sqlite"
        run.reuse_snapshot(other, source, self.fixtures)
        other.validate_records(other.snapshot, self.fixtures)
        source.write_bytes(source.read_bytes() + b"changed")
        another = Sandbox()
        self.addCleanup(another.close)
        with self.assertRaisesRegex(SafetyError, "fingerprint"):
            run.reuse_snapshot(another, source, self.fixtures)

    def test_approval_boundaries_real_create_handler_and_rejection_deltas(self):
        import tool_call

        self.prepare()
        for name in (
            "write_accepted", "write_rejected", "duplicate_job",
            "similar_accepted", "similar_rejected",
        ):
            with self.subTest(case=name):
                case = self.cases[name]
                path = self.sandbox.reset(name, self.fixtures)
                before = self.sandbox.state(path)
                approvals = Approvals(case.approvals, self.log)
                with self.sandbox.bind(path), approvals.install():
                    output = tool_call.run_tool("create_job", case.expected[0].arguments, "write")
                self.assertFalse(any(approvals.pending.values()))
                after = self.sandbox.state(path)
                self.assertEqual(len(after["jobs"]) - len(before["jobs"]), len(case.new_jobs))
                self.assertEqual(output["success"], case.expected[0].success)
                boundaries = [event["boundary"] for event in approvals.events]
                self.assertEqual(boundaries, list(case.approvals))
                self.assertTrue(all(
                    event["arguments"] == case.expected[0].arguments for event in approvals.events
                ))

    def test_unscripted_approval_and_human_input_fail_closed(self):
        approvals = Approvals({}, self.log)
        with approvals.install():
            import approval

            with self.assertRaises(SafetyError):
                approval.request_tool_approval("create_job", {})
            with self.assertRaises(SafetyError):
                input("must not prompt")

    def test_scripted_production_loop_registry_and_trajectory_assertions(self):
        import tool_call

        self.prepare()
        case = self.cases["browse_jobs"]
        outputs = iter([
            self.model("list_jobs", {"limit": 2}),
            self.model(text="Here are the first two jobs."),
        ])

        def agent(*args, **kwargs):
            self.log.event("api_request", api="main_agent")
            return next(outputs)

        with (
            patch("loop.call_model", side_effect=agent),
            patch("loop.run_tool", wraps=tool_call.run_tool) as registry,
        ):
            result, interruption = run.run_case(
                case, self.sandbox, self.fixtures, self.log, self.args,
            )
        self.assertIsNone(interruption)
        self.assertEqual(result["status"], "PASS", result["assertions"])
        registry.assert_called_once()
        self.assertTrue(any(event["kind"] == "user" for event in self.log.events))
        self.assertTrue(any(event["kind"] == "tool_output" for event in self.log.events))

    def test_grader_payload_excludes_database_states_vectors_and_raw_api_bodies(self):
        self.prepare()
        self.args.no_grader = False
        outputs = iter([
            self.model("list_jobs", {"limit": 2}), self.model(text="First two jobs."),
        ])

        def agent(*args, **kwargs):
            self.log.event(
                "api_request", api="main_agent", body={"vector_blob": "MUST_STAY_LOCAL"},
            )
            self.log.event(
                "api_response", api="main_agent", body={"embedding": [1.0] * 1536},
                usage={"input_tokens": 9},
            )
            return next(outputs)

        with (
            patch("loop.call_model", side_effect=agent),
            patch("evals.grader.grade", return_value={
                "status": "GRADER_ERROR", "error": "synthetic refusal",
            }) as grader,
        ):
            result, interruption = run.run_case(
                self.cases["browse_jobs"], self.sandbox, self.fixtures, self.log, self.args,
            )
        self.assertIsNone(interruption)
        self.assertEqual(result["status"], "GRADER_ERROR")
        grader.assert_called_once()
        payload = grader.call_args.args[0]
        self.assertFalse(any(
            event["kind"] in {"database_state", "api_request", "api_response"}
            for event in payload["events"]
        ))
        self.assertNotIn("MUST_STAY_LOCAL", json.dumps(payload))
        self.assertNotIn("sqlite_blob_hex", json.dumps(payload))
        self.assertTrue(any(event["kind"] == "database_state" for event in self.log.events))
        self.assertTrue(any(
            event["kind"] == "grader_result" and event["result"]["status"] == "GRADER_ERROR"
            for event in self.log.events
        ))
        self.assertIn("MUST_STAY_LOCAL", (self.log.directory / "trace.jsonl").read_text())
        self.assertTrue(any(
            event.get("usage") == {"input_tokens": 9} for event in payload["api_observations"]
        ))

    def test_missing_tool_or_wrong_trajectory_is_hard_failure(self):
        self.prepare()
        with patch("loop.call_model", return_value=self.model(text="I guessed some jobs.")):
            result, _ = run.run_case(
                self.cases["browse_jobs"], self.sandbox, self.fixtures, self.log, self.args,
            )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any(
            not item["passed"] and "execution_count" in item["name"]
            for item in result["assertions"]
        ))

    def test_skill_activation_same_batch_is_detected(self):
        self.prepare()
        case = self.cases["verify_academic"]
        state = self.sandbox.state(self.sandbox.snapshot)
        events = [
            dict(kind="assistant", round=1, text=None, tool_calls=[
                dict(tool="load_skill", call_id="load",
                     arguments={"name": "candidate-recognition"}),
                dict(tool="verify_candidate_claim", call_id="verify",
                     arguments={"candidate_id": "test_1"}),
            ]),
            dict(kind="tool_output", tool="load_skill", call_id="load", round=1, step=1,
                 output={"success": True, "result": "skill"}),
            dict(kind="tool_output", tool="verify_candidate_claim", call_id="verify",
                 round=1, step=1,
                 output={"success": False, "error_type": "ToolError"}),
        ]
        checks = assertions(case, events, state, state, self.fixtures, Approvals({}, self.log))
        self.assertTrue(any(
            not item["passed"] and "skill_loaded_before_batch" in item["name"] for item in checks
        ))

    def test_cross_candidate_and_inexact_citations_fail_python_checks(self):
        self.prepare()
        case = self.cases["verify_academic"]
        state = self.sandbox.state(self.sandbox.snapshot)
        result = dict(
            candidate_id="test_1", claim="academic", verdict="supported", reason="A reason",
            selected_evidence=[dict(
                evidence_id="ev_isolation_foreign_001", role="supports", quote="invented",
            )],
        )
        events = [
            dict(kind="assistant", round=1, text="verified", tool_calls=[
                dict(tool="verify_candidate_claim", call_id="v",
                     arguments={"candidate_id": "test_1"}),
            ]),
            dict(kind="tool_output", tool="verify_candidate_claim", call_id="v", round=1, step=2,
                 output={"success": True, "result": result}),
        ]
        checks = assertions(case, events, state, state, self.fixtures, Approvals({}, self.log))
        self.assertTrue(any(
            not item["passed"] and "citation_source" in item["name"] for item in checks
        ))

    def test_checker_validation_failures_are_observed_without_changing_behavior(self):
        from tools import candidate

        invalid = candidate.ClaimAssessment(
            is_atomic=True, verdict="supported", reason=" ", selected_evidence=[],
        )
        with observe_checker(self.log), self.assertRaises(Exception):
            candidate._validate_check_assessment(invalid, {})
        self.assertEqual(self.log.events[-1]["kind"], "checker_validation_failure")
        self.assertIn("must not be empty", self.log.events[-1]["error"])

    def test_grader_structured_output_pass_fail_review(self):
        with patch("openai.OpenAI") as factory:
            for verdict in ("PASS", "FAIL", "REVIEW"):
                expected = EvalGrade(
                    verdict=verdict, reason="Grounded judgment",
                    failed_criteria=[], evidence_refs=["job_001"],
                )
                factory.return_value.responses.parse.return_value = SimpleNamespace(
                    status="completed", output_parsed=expected,
                )
                result = grade({"synthetic": True}, "test-grader")
                self.assertEqual(result["status"], verdict)
                request = factory.return_value.responses.parse.call_args.kwargs
                self.assertEqual(request["model"], "test-grader")
                self.assertIs(request["text_format"], EvalGrade)
                self.assertIs(request["store"], False)
                self.assertNotIn("tools", request)

    def test_grader_api_refusal_incomplete_and_schema_errors_are_not_review(self):
        with patch("openai.OpenAI") as factory:
            parse = factory.return_value.responses.parse
            for value in (
                SimpleNamespace(status="incomplete", output_parsed=None),
                SimpleNamespace(status="completed", output_parsed=None),
                SimpleNamespace(status="completed", output_parsed={"verdict": "PASS"}),
            ):
                parse.return_value = value
                self.assertEqual(grade({})["status"], "GRADER_ERROR")
            parse.side_effect = ValueError("provider schema error")
            self.assertEqual(grade({})["status"], "GRADER_ERROR")
            parse.side_effect = BudgetExceeded("budget")
            with self.assertRaises(BudgetExceeded):
                grade({})

    def test_positive_grade_cannot_override_hard_failure(self):
        self.assertEqual(overall([{"passed": False}], {"status": "PASS"}), "FAIL")
        for verdict in ("PASS", "FAIL", "REVIEW", "GRADER_ERROR"):
            self.assertEqual(overall([{"passed": True}], {"status": verdict}), verdict)

    def test_trace_redacts_secrets_and_atomic_reports_survive_finish(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-synthetic-secret"}):
            self.log.event(
                "api_request",
                body={"Authorization": "Bearer secret", "text": "sk-synthetic-secret"},
            )
        self.log.finish({"status": "FAIL"})
        trace = (self.log.directory / "trace.jsonl").read_text()
        self.assertNotIn("sk-synthetic-secret", trace)
        self.assertNotIn("Bearer secret", trace)
        self.assertIn("[REDACTED]", trace)
        summary = json.loads((self.log.directory / "summary.json").read_text())
        manifest = json.loads((self.log.directory / "manifest.json").read_text())
        self.assertEqual(summary["status"], "FAIL")
        self.assertIsNotNone(manifest["finished_at"])
        self.assertTrue(all(json.loads(line)["at"] for line in trace.splitlines()))

    def test_git_metadata_records_head_branch_and_dirty_content_fingerprint(self):
        metadata = git_metadata()
        self.assertEqual(len(metadata["git_sha"]), 40)
        self.assertEqual(metadata["git_branch"], "stage3")
        self.assertIsInstance(metadata["dirty"], bool)
        if metadata["dirty"]:
            self.assertEqual(len(metadata["dirty_fingerprint"]), 64)

    def test_http_attempts_usage_and_budget_include_sdk_retries(self):
        from openai import OpenAI

        budget = RequestBudget(1)

        def failure(client, request, **kwargs):
            return httpx.Response(
                500, request=request, json={"error": {"message": "synthetic failure"}},
            )

        with patch.object(httpx.Client, "send", failure), observe_apis(self.log, budget):
            client = OpenAI(api_key="test-only", max_retries=1)
            with self.assertRaises(BudgetExceeded):
                client.embeddings.create(model="text-embedding-3-small", input=["synthetic"])
        self.assertEqual(budget.used, 1)
        requests = [event for event in self.log.events if event["kind"] == "api_request"]
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.log.events[-1]["api"], "embedding")
        self.assertEqual(self.log.events[-1]["status_code"], 500)

    def test_http_success_records_usage_and_latency_without_headers(self):
        def success(client, request, **kwargs):
            return httpx.Response(
                200, request=request, json={"usage": {"input_tokens": 7, "output_tokens": 3}},
            )

        with patch.object(httpx.Client, "send", success), observe_apis(self.log, RequestBudget(1)):
            httpx.Client().post(
                "https://example.invalid/responses", json={"model": "test", "input": []},
                headers={"Authorization": "secret"},
            )
        event = self.log.events[-1]
        self.assertEqual(event["usage"], {"input_tokens": 7, "output_tokens": 3})
        self.assertGreaterEqual(event["latency_seconds"], 0)
        self.assertNotIn("Authorization", json.dumps(self.log.events))

    def test_run_case_interruption_still_records_state_and_failure(self):
        self.prepare()
        with patch("loop.call_model", side_effect=KeyboardInterrupt()):
            result, interruption = run.run_case(
                self.cases["browse_jobs"], self.sandbox, self.fixtures, self.log, self.args,
            )
        self.assertIsInstance(interruption, KeyboardInterrupt)
        self.assertEqual(result["status"], "ABORTED")
        self.assertTrue(any(event["kind"] == "database_state" for event in self.log.events))
        self.assertEqual(self.log.events[-1]["kind"], "scenario_finished")

    def test_live_preparation_failure_and_interrupt_create_permanent_reports(self):
        for error in (RuntimeError("setup failed"), KeyboardInterrupt(), BudgetExceeded("budget")):
            with self.subTest(error=error):
                directory = Path(self.directory.name) / ("failure-" + type(error).__name__)

                def logger(metadata):
                    return RunLog(metadata, directory)

                with (
                    patch("evals.recording.RunLog", side_effect=logger),
                    patch.object(Sandbox, "prepare_indexes", side_effect=error),
                    redirect_stdout(io.StringIO()),
                ):
                    code = run.live(self.args, self.fixtures, [self.cases["browse_jobs"]])
                self.assertEqual(code, 1)
                summary = json.loads((directory / "summary.json").read_text())
                self.assertIn(summary["status"], {"ERROR", "ABORTED"})
                self.assertTrue((directory / "trace.jsonl").read_text())
                self.assertFalse((directory / "assertions.json").exists())
                self.assertFalse((directory / "grades.json").exists())

    def test_database_summary_hashes_are_stable_and_preserve_complete_contents(self):
        before = {
            "jobs": [{"id": "job_002", "title": "Beta"}, {"title": "Alpha", "id": "job_001"}],
            "empty": [],
            "opaque": [{"value": None}, {"value": "你好"}],
        }
        original = deepcopy(before)
        reordered = {
            table: [dict(reversed(list(row.items()))) for row in reversed(rows)]
            for table, rows in reversed(list(before.items()))
        }
        summary = summarize_database_changes(before, reordered)
        for table in before:
            self.assertFalse(summary[table]["changed"])
            self.assertEqual(summary[table]["before"], summary[table]["after"])
            self.assertEqual(len(summary[table]["before"]["sha256"]), 64)
        self.assertEqual(before, original)
        self.assertEqual(summary, summarize_database_changes(reordered, before))

        duplicate = deepcopy(before)
        duplicate["opaque"].append({"value": None})
        self.assertTrue(summarize_database_changes(before, duplicate)["opaque"]["changed"])
        changed_null = deepcopy(before)
        changed_null["opaque"][0]["value"] = ""
        self.assertTrue(summarize_database_changes(before, changed_null)["opaque"]["changed"])

    def test_database_summary_records_added_removed_and_modified_ids_and_fields(self):
        before = {"jobs": [
            {"id": "job_001", "location": None},
            {"id": "job_002", "location": "Melbourne"},
            {"id": "job_003", "location": "Sydney"},
        ]}
        after = {"jobs": [
            {"id": "job_001", "location": None},
            {"id": "job_003", "location": "Remote"},
            {"id": "job_004", "location": "Brisbane"},
        ]}
        table = summarize_database_changes(before, after)["jobs"]
        self.assertEqual(table["before"]["row_count"], 3)
        self.assertEqual(table["after"]["row_count"], 3)
        self.assertNotEqual(table["before"]["sha256"], table["after"]["sha256"])
        changes = table["changes"]
        self.assertEqual(changes["id_fields"], ["id"])
        self.assertEqual([row["id"] for row in changes["added"]], ["job_004"])
        self.assertEqual([row["id"] for row in changes["removed"]], ["job_002"])
        self.assertEqual(changes["modified"], [{
            "id": "job_003", "fields": {"location": {"before": "Sydney", "after": "Remote"}},
        }])
        self.assertEqual(changes["added"][0]["fields"]["location"], {"after": "Brisbane"})

    def test_database_summary_compacts_vectors_blobs_and_handles_tables_without_ids(self):
        embedding = {
            "content_hash": "synthetic-content-hash", "embedding_model": "text-embedding-3-small",
            "dimensions": 1536, "embedding": json.dumps([0.1] * 1536),
        }
        before = {
            "chunk_embeddings": [embedding],
            "chunk_fts_data": [{"id": 1, "block": {"sqlite_blob_hex": "ab" * 10000}}],
            "opaque": [{"value": 1}, {"value": 1}],
            "removed_empty": [],
        }
        after = deepcopy(before)
        after["chunk_embeddings"][0]["embedding"] = json.dumps([0.2] * 1536)
        after["chunk_fts_data"][0]["block"]["sqlite_blob_hex"] = "cd" * 10000
        after["opaque"].append({"value": 2})
        del after["removed_empty"]
        after["added_empty"] = []
        summary = summarize_database_changes(before, after)
        text = json.dumps(summary)
        self.assertLess(len(text), 4000)
        self.assertNotIn(embedding["embedding"], text)
        self.assertNotIn("ab" * 100, text)
        self.assertNotIn("cd" * 100, text)
        changed = summary["chunk_embeddings"]["changes"]["modified"][0]
        self.assertEqual(changed["id"], {
            "content_hash": "synthetic-content-hash", "embedding_model": "text-embedding-3-small",
        })
        self.assertNotEqual(
            changed["fields"]["embedding"]["before"]["sha256"],
            changed["fields"]["embedding"]["after"]["sha256"],
        )
        self.assertIsNone(summary["opaque"]["changes"])
        self.assertTrue(summary["opaque"]["changed"])
        self.assertFalse(summary["removed_empty"]["after"]["present"])
        self.assertFalse(summary["added_empty"]["before"]["present"])

    def test_database_assertions_still_compare_full_expected_and_unchanged_rows(self):
        self.prepare()
        case = self.cases["write_accepted"]
        before = self.sandbox.state(self.sandbox.snapshot)
        after = deepcopy(before)
        after["jobs"].append({"id": "job_009", **case.new_jobs[0]})
        approvals = Approvals(case.approvals, self.log)
        approvals.decide("initial", "create_job", case.new_jobs[0])
        # Even a misleading log summary cannot replace the complete assertion inputs.
        events = [{"kind": "database_state", "tables": summarize_database_changes(before, before)}]
        checks = assertions(case, events, before, after, self.fixtures, approvals)
        passed = {item["name"]: item["passed"] for item in checks}
        for name in ("existing_jobs_unchanged", "exact_job_changes", "non_job_tables_unchanged"):
            self.assertTrue(passed[name], name)

        for mutation in ("embedding", "modified_job", "removed_job", "unexpected_job"):
            with self.subTest(mutation=mutation):
                changed = deepcopy(after)
                if mutation == "embedding":
                    vector = json.loads(changed["chunk_embeddings"][0]["embedding"])
                    vector[-1] = 0.5
                    changed["chunk_embeddings"][0]["embedding"] = json.dumps(vector)
                    expected_failure = "non_job_tables_unchanged"
                elif mutation == "modified_job":
                    changed["jobs"][0]["description"] += " changed"
                    expected_failure = "existing_jobs_unchanged"
                elif mutation == "removed_job":
                    changed["jobs"].pop(0)
                    expected_failure = "existing_jobs_unchanged"
                else:
                    changed["jobs"].append({"id": "job_010", **case.new_jobs[0]})
                    expected_failure = "exact_job_changes"
                checks = assertions(case, events, before, changed, self.fixtures, approvals)
                passed = {item["name"]: item["passed"] for item in checks}
                self.assertFalse(passed[expected_failure])

    def test_run_case_logs_compact_states_but_asserts_complete_database(self):
        self.prepare()
        case = self.cases["write_accepted"]
        outputs = iter([
            self.model("create_job", case.new_jobs[0]),
            self.model(text="The approved job was created."),
        ])

        def agent(*args, **kwargs):
            self.log.event("api_request", api="main_agent")
            return next(outputs)

        with (
            patch("loop.call_model", side_effect=agent),
            patch("evals.assertions.assertions", wraps=assertions) as check,
        ):
            result, interruption = run.run_case(
                case, self.sandbox, self.fixtures, self.log, self.args,
            )
        self.assertIsNone(interruption)
        self.assertEqual(result["status"], "PASS", result["assertions"])
        before, after = check.call_args.args[2:4]
        self.assertGreater(len(before["chunk_embeddings"][0]["embedding"]), 1000)
        self.assertEqual(len(after["jobs"]), 9)
        event = next(event for event in self.log.events if event["kind"] == "database_state")
        self.assertEqual(event["format_version"], 2)
        self.assertNotIn("before", event)
        tables = event["tables"]
        self.assertEqual(tables["jobs"]["before"]["row_count"], 8)
        self.assertEqual(tables["jobs"]["after"]["row_count"], 9)
        self.assertEqual(tables["jobs"]["changes"]["added"][0]["id"], "job_009")
        self.assertTrue(all(
            not table["changed"] for name, table in tables.items() if name != "jobs"
        ))
        trace = (self.log.directory / "trace.jsonl").read_text()
        self.assertNotIn(before["chunk_embeddings"][0]["embedding"], trace)
        self.assertNotIn("sqlite_blob_hex", trace)

    def test_only_five_reports_keep_full_results_on_success_failure_and_interrupt(self):
        self.args.no_grader = False
        case = self.cases["browse_jobs"]
        for outcome in ("PASS", "FAIL", "ABORTED"):
            with self.subTest(outcome=outcome):
                directory = Path(self.directory.name) / outcome
                holder = {}
                progress = []
                outputs = iter([
                    self.model("list_jobs", {"limit": 2}),
                    self.model(text="First two stored jobs."),
                ])

                def logger(metadata):
                    log = RunLog(metadata, directory)
                    holder["log"] = log
                    write = log.write

                    def record_write(name, value):
                        if name == "summary.json":
                            progress.append(deepcopy(value))
                        return write(name, value)

                    log.write = record_write
                    return log

                def agent(*args, **kwargs):
                    if outcome == "ABORTED":
                        raise KeyboardInterrupt()
                    if outcome == "FAIL":
                        raise RuntimeError("synthetic scenario failure")
                    holder["log"].event("api_request", api="main_agent")
                    return next(outputs)

                def vectors(texts):
                    return [[1.0] + [0.0] * 1535 for _ in texts]

                grading = {"status": "PASS", "grade": {"verdict": "PASS", "reason": "synthetic"}}
                with (
                    patch("evals.recording.RunLog", side_effect=logger),
                    patch("rag_lab.indexing.embedding_texts", side_effect=vectors),
                    patch("rag_lab.job_indexing.embedding_texts", side_effect=vectors),
                    patch("loop.call_model", side_effect=agent),
                    patch("evals.grader.grade", return_value=grading),
                    patch.object(
                        httpx.Client, "send", side_effect=AssertionError("No paid calls"),
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    code = run.live(self.args, self.fixtures, [case])
                self.assertEqual(code, 0 if outcome == "PASS" else 1)
                self.assertEqual({path.name for path in directory.iterdir()}, {
                    "manifest.json", "summary.json", "trace.jsonl",
                    "prepared.sqlite", "prepared.json",
                })
                summary = json.loads((directory / "summary.json").read_text())
                self.assertEqual(summary["status"], outcome)
                result = summary["results"][0]
                self.assertEqual(result["status"], outcome)
                self.assertTrue(result["assertions"])
                self.assertEqual(result["grading"], None if outcome == "ABORTED" else grading)
                running = next(item for item in progress if item["status"] == "RUNNING")
                self.assertEqual(running["results"][0], result)
                trace_lines = (directory / "trace.jsonl").read_text().splitlines()
                trace = [json.loads(line) for line in trace_lines]
                self.assertTrue(any(event["kind"] == "database_state" for event in trace))
                self.assertEqual(trace[-1]["kind"], "run_finished")

    def test_existing_historical_report_directory_cannot_be_overwritten(self):
        directory = Path(self.directory.name) / "historical"
        directory.mkdir()
        files = {
            "summary.json": '{"status": "FAIL", "results": []}',
            "assertions.json": '{"legacy_case": []}',
            "grades.json": '{"legacy_case": {"status": "FAIL"}}',
            "trace.jsonl": '{"kind": "database_state", "before": {}, "after": {}}\n',
        }
        for name, content in files.items():
            (directory / name).write_text(content)
        with self.assertRaises(FileExistsError):
            RunLog({}, directory)
        for name, content in files.items():
            self.assertEqual((directory / name).read_text(), content)
        self.assertEqual(json.loads((directory / "summary.json").read_text())["status"], "FAIL")

    def test_safe_default_dry_run_and_case_limits_never_call_api(self):
        with (
            patch.object(httpx.Client, "send", side_effect=AssertionError("No paid calls")),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run.main([]), 0)
            self.assertEqual(run.main(["--suite", "full", "--dry-run"]), 0)
            self.assertEqual(run.main(["--max-cases", "1"]), 2)
            self.assertEqual(run.main(["--max-api-requests", "0"]), 2)
            self.assertEqual(run.main(["--case", "unknown"]), 2)
            self.assertEqual(run.main(["--reuse-snapshot", str(self.sandbox.snapshot)]), 2)


if __name__ == "__main__":
    unittest.main()
