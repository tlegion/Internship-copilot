from contextlib import redirect_stdout, closing
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx2 as httpx

from evals import run
from evals.assertions import assertions
from evals.fixtures import Job, load_fixtures, ROOT
from evals.grader import EvalGrade, grade, overall
from evals.instrumentation import Approvals, observe_apis, observe_checker
from evals.isolation import Sandbox, SafetyError
from evals.recording import BudgetExceeded, RequestBudget, RunLog, git_metadata
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
        self.addCleanup(lambda: self.log._stream.close() if not self.log._stream.closed else None)
        self.cases = {case.id: case for case in load_cases("full", self.fixtures)}
        self.args = run.parser().parse_args(["--no-grader"])

    def prepare(self):
        def vectors(texts):
            return [[1.0] + [0.0] * 1535 for _ in texts]
        with patch("rag_lab.indexing.embedding_texts", side_effect=vectors), \
                patch("rag_lab.job_indexing.embedding_texts", side_effect=vectors):
            self.sandbox.prepare_indexes(self.fixtures)

    def model(self, name=None, arguments=None, text=None):
        calls = [ModelToolCall(name, arguments or {}, "call-" + name)] if name else []
        return ModelResponse(text, calls, "completed", SimpleNamespace(output=[]), 1)

    def test_fixture_contracts_and_source_ownership(self):
        self.assertEqual(len(self.fixtures["jobs"]), 8)
        self.assertEqual(len(self.fixtures["evidence"]), 13)
        self.assertEqual({row["candidate_id"] for row in self.fixtures["evidence"]}, {"test_1", "synthetic_other"})
        self.assertIsNone(self.fixtures["jobs"][2]["location"])
        for invalid in ({**self.fixtures["jobs"][0], "extra": "bad"},
                        {**self.fixtures["jobs"][0], "location": 3}):
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
        with self.sandbox.bind(self.sandbox.snapshot), closing(sqlite3.connect(self.sandbox.snapshot)) as connection, connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 4)
            connection.execute("UPDATE evidence SET candidate_id='wrong' WHERE evidence_id=?",
                               (self.fixtures["evidence"][0]["evidence_id"],))
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
            for target in (outside, f"file:{outside}?mode=rw", ":memory:",
                           ROOT / "data/internship.db"):
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
        faults = ["DELETE FROM chunks", "DELETE FROM chunk_fts", "DELETE FROM chunk_embeddings",
                  "UPDATE chunks SET text='stale'", "DELETE FROM chunks_jobs",
                  "DELETE FROM chunk_fts_jobs", "DELETE FROM chunk_embeddings_jobs",
                  "UPDATE chunk_embeddings SET embedding='[0]'",
                  "UPDATE chunk_embeddings_jobs SET embedding='invalid'",
                  "UPDATE chunk_embeddings SET dimensions=4"]
        for index, statement in enumerate(faults):
            with self.subTest(statement=statement):
                path = self.sandbox.reset(f"fault{index}", self.fixtures)
                with self.sandbox.bind(path), closing(sqlite3.connect(path)) as connection, connection:
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
        for name in ("write_accepted", "write_rejected", "duplicate_job", "similar_accepted", "similar_rejected"):
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
                self.assertEqual([event["boundary"] for event in approvals.events], list(case.approvals))
                self.assertTrue(all(event["arguments"] == case.expected[0].arguments for event in approvals.events))

    def test_unscripted_approval_and_human_input_fail_closed(self):
        approvals = Approvals({}, self.log)
        with approvals.install():
            import approval
            with self.assertRaises(SafetyError):
                approval.request_tool_approval("create_job", {})
            with self.assertRaises(SafetyError):
                input("must not prompt")

    def test_scripted_production_loop_registry_and_trajectory_assertions(self):
        self.prepare()
        case = self.cases["browse_jobs"]
        outputs = iter([self.model("list_jobs", {"limit": 2}), self.model(text="Here are the first two jobs.")])
        def agent(*args, **kwargs):
            self.log.event("api_request", api="main_agent")
            return next(outputs)
        with patch("loop.call_model", side_effect=agent), patch("loop.run_tool", wraps=__import__('tool_call').run_tool) as registry:
            result, interruption = run.run_case(case, self.sandbox, self.fixtures, self.log, self.args)
        self.assertIsNone(interruption)
        self.assertEqual(result["status"], "PASS", result["assertions"])
        registry.assert_called_once()
        self.assertTrue(any(event["kind"] == "user" for event in self.log.events))
        self.assertTrue(any(event["kind"] == "tool_output" for event in self.log.events))

    def test_grader_payload_excludes_database_states_vectors_and_raw_api_bodies(self):
        self.prepare()
        self.args.no_grader = False
        outputs = iter([self.model("list_jobs", {"limit": 2}), self.model(text="First two jobs.")])
        def agent(*args, **kwargs):
            self.log.event("api_request", api="main_agent", body={"vector_blob":"MUST_STAY_LOCAL"})
            self.log.event("api_response", api="main_agent", body={"embedding":[1.0]*1536}, usage={"input_tokens":9})
            return next(outputs)
        with patch("loop.call_model", side_effect=agent), patch("evals.grader.grade", return_value={
                "status":"GRADER_ERROR", "error":"synthetic refusal"}) as grader:
            result, interruption = run.run_case(self.cases["browse_jobs"], self.sandbox,
                                               self.fixtures, self.log, self.args)
        self.assertIsNone(interruption)
        self.assertEqual(result["status"], "GRADER_ERROR")
        grader.assert_called_once()
        payload = grader.call_args.args[0]
        self.assertFalse(any(event["kind"] in {"database_state","api_request","api_response"}
                             for event in payload["events"]))
        self.assertNotIn("MUST_STAY_LOCAL", json.dumps(payload))
        self.assertNotIn("sqlite_blob_hex", json.dumps(payload))
        self.assertTrue(any(event["kind"] == "database_state" for event in self.log.events))
        self.assertTrue(any(event["kind"] == "grader_result" and event["result"]["status"] == "GRADER_ERROR"
                            for event in self.log.events))
        self.assertIn("MUST_STAY_LOCAL", (self.log.directory/"trace.jsonl").read_text())
        self.assertTrue(any(event.get("usage") == {"input_tokens":9} for event in payload["api_observations"]))

    def test_missing_tool_or_wrong_trajectory_is_hard_failure(self):
        self.prepare()
        with patch("loop.call_model", return_value=self.model(text="I guessed some jobs.")):
            result, _ = run.run_case(self.cases["browse_jobs"], self.sandbox, self.fixtures, self.log, self.args)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any(not item["passed"] and "execution_count" in item["name"] for item in result["assertions"]))

    def test_skill_activation_same_batch_is_detected(self):
        self.prepare()
        case = self.cases["verify_academic"]
        state = self.sandbox.state(self.sandbox.snapshot)
        events = [dict(kind="assistant", round=1, text=None, tool_calls=[
            dict(tool="load_skill", call_id="load", arguments={"name":"candidate-recognition"}),
            dict(tool="verify_candidate_claim", call_id="verify", arguments={"candidate_id":"test_1"})]),
            dict(kind="tool_output", tool="load_skill",call_id="load",round=1,step=1,output={"success":True,"result":"skill"}),
            dict(kind="tool_output", tool="verify_candidate_claim",call_id="verify",round=1,step=1,
                 output={"success":False,"error_type":"ToolError"})]
        checks = assertions(case, events, state, state, self.fixtures, Approvals({},self.log))
        self.assertTrue(any(not item["passed"] and "skill_loaded_before_batch" in item["name"] for item in checks))

    def test_cross_candidate_and_inexact_citations_fail_python_checks(self):
        self.prepare()
        case = self.cases["verify_academic"]
        state = self.sandbox.state(self.sandbox.snapshot)
        result = dict(candidate_id="test_1",claim="academic",verdict="supported",reason="A reason",
                      selected_evidence=[dict(evidence_id="ev_isolation_foreign_001",role="supports",quote="invented")])
        events = [dict(kind="assistant", round=1, text="verified", tool_calls=[
                    dict(tool="verify_candidate_claim",call_id="v",arguments={"candidate_id":"test_1"})]),
                  dict(kind="tool_output",tool="verify_candidate_claim",call_id="v",round=1,step=2,
                       output={"success":True,"result":result})]
        checks = assertions(case, events, state, state, self.fixtures, Approvals({},self.log))
        self.assertTrue(any(not item["passed"] and "citation_source" in item["name"] for item in checks))

    def test_checker_validation_failures_are_observed_without_changing_behavior(self):
        from tools import candidate
        invalid = candidate.ClaimAssessment(is_atomic=True, verdict="supported", reason=" ",selected_evidence=[])
        with observe_checker(self.log), self.assertRaises(Exception):
            candidate._validate_check_assessment(invalid, {})
        self.assertEqual(self.log.events[-1]["kind"], "checker_validation_failure")
        self.assertIn("must not be empty", self.log.events[-1]["error"])

    def test_grader_structured_output_pass_fail_review(self):
        with patch("openai.OpenAI") as factory:
            for verdict in ("PASS", "FAIL", "REVIEW"):
                expected = EvalGrade(verdict=verdict, reason="Grounded judgment",failed_criteria=[],evidence_refs=["job_001"])
                factory.return_value.responses.parse.return_value = SimpleNamespace(status="completed",output_parsed=expected)
                result = grade({"synthetic":True}, "test-grader")
                self.assertEqual(result["status"], verdict)
                request = factory.return_value.responses.parse.call_args.kwargs
                self.assertEqual(request["model"], "test-grader")
                self.assertIs(request["text_format"], EvalGrade)
                self.assertIs(request["store"], False)
                self.assertNotIn("tools", request)

    def test_grader_api_refusal_incomplete_and_schema_errors_are_not_review(self):
        with patch("openai.OpenAI") as factory:
            parse = factory.return_value.responses.parse
            for value in (SimpleNamespace(status="incomplete",output_parsed=None),
                          SimpleNamespace(status="completed",output_parsed=None),
                          SimpleNamespace(status="completed",output_parsed={"verdict":"PASS"})):
                parse.return_value = value
                self.assertEqual(grade({})["status"], "GRADER_ERROR")
            parse.side_effect = ValueError("provider schema error")
            self.assertEqual(grade({})["status"], "GRADER_ERROR")
            parse.side_effect = BudgetExceeded("budget")
            with self.assertRaises(BudgetExceeded):
                grade({})

    def test_positive_grade_cannot_override_hard_failure(self):
        self.assertEqual(overall([{"passed":False}], {"status":"PASS"}), "FAIL")
        for verdict in ("PASS", "FAIL", "REVIEW", "GRADER_ERROR"):
            self.assertEqual(overall([{"passed":True}], {"status":verdict}), verdict)

    def test_trace_redacts_secrets_and_atomic_reports_survive_finish(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY":"sk-synthetic-secret"}):
            self.log.event("api_request", body={"Authorization":"Bearer secret", "text":"sk-synthetic-secret"})
        self.log.finish({"status":"FAIL"})
        trace = (self.log.directory / "trace.jsonl").read_text()
        self.assertNotIn("sk-synthetic-secret", trace)
        self.assertNotIn("Bearer secret", trace)
        self.assertIn("[REDACTED]", trace)
        self.assertEqual(json.loads((self.log.directory / "summary.json").read_text())["status"], "FAIL")
        self.assertIsNotNone(json.loads((self.log.directory / "manifest.json").read_text())["finished_at"])
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
            return httpx.Response(500, request=request, json={"error":{"message":"synthetic failure"}})
        with patch.object(httpx.Client,"send",failure), observe_apis(self.log,budget):
            client = OpenAI(api_key="test-only", max_retries=1)
            with self.assertRaises(BudgetExceeded):
                client.embeddings.create(model="text-embedding-3-small",input=["synthetic"])
        self.assertEqual(budget.used,1)
        self.assertEqual(len([e for e in self.log.events if e["kind"]=="api_request"]),1)
        self.assertEqual(self.log.events[-1]["api"],"embedding")
        self.assertEqual(self.log.events[-1]["status_code"],500)

    def test_http_success_records_usage_and_latency_without_headers(self):
        def success(client, request, **kwargs):
            return httpx.Response(200,request=request,json={"usage":{"input_tokens":7,"output_tokens":3}})
        with patch.object(httpx.Client,"send",success), observe_apis(self.log,RequestBudget(1)):
            httpx.Client().post("https://example.invalid/responses",json={"model":"test","input":[]},
                                headers={"Authorization":"secret"})
        event = self.log.events[-1]
        self.assertEqual(event["usage"],{"input_tokens":7,"output_tokens":3})
        self.assertGreaterEqual(event["latency_seconds"],0)
        self.assertNotIn("Authorization",json.dumps(self.log.events))

    def test_run_case_interruption_still_records_state_and_failure(self):
        self.prepare()
        with patch("loop.call_model", side_effect=KeyboardInterrupt()):
            result, interruption = run.run_case(self.cases["browse_jobs"], self.sandbox,self.fixtures,self.log,self.args)
        self.assertIsInstance(interruption,KeyboardInterrupt)
        self.assertEqual(result["status"],"ABORTED")
        self.assertTrue(any(e["kind"]=="database_state" for e in self.log.events))
        self.assertEqual(self.log.events[-1]["kind"],"scenario_finished")

    def test_live_preparation_failure_and_interrupt_create_permanent_reports(self):
        for error in (RuntimeError("setup failed"), KeyboardInterrupt(), BudgetExceeded("budget")):
            with self.subTest(error=error):
                directory = Path(self.directory.name) / ("failure-" + type(error).__name__)
                real_log = RunLog
                def logger(metadata):
                    return real_log(metadata,directory)
                with patch("evals.recording.RunLog",side_effect=logger), \
                        patch.object(Sandbox,"prepare_indexes",side_effect=error), redirect_stdout(io.StringIO()):
                    code = run.live(self.args,self.fixtures,[self.cases["browse_jobs"]])
                self.assertEqual(code,1)
                summary = json.loads((directory/"summary.json").read_text())
                self.assertIn(summary["status"],{"ERROR","ABORTED"})
                self.assertTrue((directory/"trace.jsonl").read_text())
                self.assertTrue((directory/"assertions.json").exists())
                self.assertTrue((directory/"grades.json").exists())

    def test_safe_default_dry_run_and_case_limits_never_call_api(self):
        with patch.object(httpx.Client,"send",side_effect=AssertionError("No paid calls")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run.main([]),0)
            self.assertEqual(run.main(["--suite","full","--dry-run"]),0)
            self.assertEqual(run.main(["--max-cases","1"]),2)
            self.assertEqual(run.main(["--max-api-requests","0"]),2)
            self.assertEqual(run.main(["--case","unknown"]),2)
            self.assertEqual(run.main(["--reuse-snapshot",str(self.sandbox.snapshot)]),2)


if __name__ == "__main__":
    unittest.main()
