import io
import importlib.util
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import approval
from errors import StorageError
from tools import jobs


# Load the real dispatcher with unrelated evidence retrieval isolated: importing
# its tokenizer would otherwise download data before any job test runs.
evidence_stub = ModuleType("tools.evidence")
evidence_stub.search_evidence_hybrid = Mock()
spec = importlib.util.spec_from_file_location(
    "_job_review_tool_call", Path(__file__).resolve().parents[1] / "tool_call.py"
)
tool_call = importlib.util.module_from_spec(spec)
with patch.dict("sys.modules", {"tools.evidence": evidence_stub}):
    spec.loader.exec_module(tool_call)


PENDING = {
    "title": "Engineering Intern",
    "company": "Example",
    "description": "Build tools\nFull pending description",
    "location": None,
}
SIMILARS = [
    {"id": "job_001", **PENDING, "description": "Old duties\nFull existing description"},
    {"id": "job_002", **PENDING, "location": "Hong Kong"},
]


class ApprovalTests(unittest.TestCase):
    def test_generic_approval_argument_order_and_optional_call_id(self):
        for call_id in (None, "call_001"):
            with self.subTest(call_id=call_id), patch("builtins.input", return_value="yes"):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertTrue(approval.request_tool_approval("create_job", PENDING, call_id))
                self.assertIn("Tool: create_job", output.getvalue())
                self.assertIn(PENDING["description"], output.getvalue())
                if call_id:
                    self.assertIn(f"Call ID: {call_id}", output.getvalue())
                else:
                    self.assertNotIn("Call ID:", output.getvalue())

    def test_review_shows_full_values_and_equality_markers(self):
        with patch("builtins.input", return_value="yes") as prompt:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertTrue(approval.review_similar_job_creation("create_job", PENDING, SIMILARS))
        text = output.getvalue()
        self.assertIn(PENDING["description"], text)
        self.assertNotIn("Call ID:", text)
        for similar in SIMILARS:
            self.assertIn(f"Existing Job ID: {similar['id']}", text)
            for field in PENDING:
                marker = "SAME" if PENDING[field] == similar[field] else "DIFF"
                self.assertIn(f"[{marker}] {field}: {similar[field]}", text)
        prompt.assert_called_once_with(
            "Similar jobs exist. Do you still want to insert this job? [Y/N] "
        )

    def test_review_answers_and_safe_interrupts(self):
        for answer, expected in (("y", True), (" YES ", True), ("n", False), ("No", False), ("", False)):
            with self.subTest(answer=answer), patch("builtins.input", return_value=answer), redirect_stdout(io.StringIO()):
                self.assertEqual(approval.review_similar_job_creation("create_job", PENDING, SIMILARS), expected)
        for exception in (EOFError, KeyboardInterrupt):
            with self.subTest(exception=exception), patch("builtins.input", side_effect=exception), redirect_stdout(io.StringIO()):
                self.assertFalse(approval.review_similar_job_creation("create_job", PENDING, SIMILARS))

    def test_invalid_input_reprompts_then_enter_declines(self):
        with patch("builtins.input", side_effect=["invalid", ""]) as prompt:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertFalse(approval.review_similar_job_creation("create_job", PENDING, SIMILARS))
        self.assertEqual(prompt.call_count, 2)
        self.assertIn("Please enter y or n.", output.getvalue())


class CreateJobTests(unittest.TestCase):
    def setUp(self):
        # Mock every storage boundary so these tests cannot access the demo database.
        self.exists = self.start_patch("tools.jobs.check_exists", return_value=None)
        self.similar = self.start_patch("tools.jobs.check_similar", return_value=[])
        self.new_id = self.start_patch("tools.jobs.new_job_id", return_value="job_003")
        self.insert = self.start_patch("tools.jobs.insert_job", return_value={"msg": "Job creation successful", "extra": "preserved"})

    def start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def run_create(self, answers):
        with patch("builtins.input", side_effect=answers) as prompt, redirect_stdout(io.StringIO()):
            result = tool_call.run_tool("create_job", PENDING, "call_001")
        self.assertEqual(json.loads(json.dumps(result)), result)
        return result, prompt

    def test_exact_duplicate_keeps_error_and_skips_review_and_insert(self):
        self.exists.return_value = [{"id": "job_001"}]
        with patch.object(approval, "review_similar_job_creation") as review:
            result, prompt = self.run_create(["y"])
        self.assertEqual(result, {"success": False, "error_msg": "job already exists", "error_type": "ToolError", "error_args": []})
        self.similar.assert_not_called()
        review.assert_not_called()
        self.new_id.assert_not_called()
        self.insert.assert_not_called()
        self.assertEqual(prompt.call_count, 1)

    def test_no_similar_inserts_after_generic_approval(self):
        with patch.object(approval, "review_similar_job_creation") as review:
            result, prompt = self.run_create(["y"])
        self.assertTrue(tool_call.tool_list["create_job"]["requires_approval"])
        self.assertEqual(result, {"success": True, "result": {"msg": "Job creation successful", "extra": "preserved", "status": "created", "job_id": "job_003"}})
        self.exists.assert_called_once_with(*PENDING.values())
        self.similar.assert_called_once_with(*PENDING.values())
        review.assert_not_called()
        self.new_id.assert_called_once_with()
        self.insert.assert_called_once_with("job_003", *PENDING.values())
        self.assertEqual(prompt.call_count, 1)

    def test_similar_yes_requires_two_approvals_and_inserts_once(self):
        self.similar.return_value = SIMILARS
        with patch.object(approval, "review_similar_job_creation", wraps=approval.review_similar_job_creation) as review:
            result, prompt = self.run_create(["y", "yes"])
        review.assert_called_once_with("create_job", PENDING, SIMILARS)
        self.assertEqual(result, {"success": True, "result": {"msg": "Job creation successful", "extra": "preserved", "status": "created", "job_id": "job_003", "similar_job_ids": ["job_001", "job_002"], "user_selection": True}})
        self.new_id.assert_called_once_with()
        self.insert.assert_called_once_with("job_003", *PENDING.values())
        self.assertEqual(prompt.call_count, 2)

    def test_similar_decline_returns_successful_cancellation_without_writes(self):
        self.similar.return_value = SIMILARS
        for answer in ("no", "", EOFError, KeyboardInterrupt):
            with self.subTest(answer=answer):
                result, prompt = self.run_create(["y", answer])
                self.assertEqual(result, {"success": True, "result": {"status": "cancelled", "job_id": None, "similar_job_ids": ["job_001", "job_002"], "user_selection": False}})
                self.assertEqual(prompt.call_count, 2)
        self.new_id.assert_not_called()
        self.insert.assert_not_called()

    def test_generic_rejection_never_dispatches_business_function(self):
        with patch.dict(tool_call.tool_list["create_job"], {"handler": Mock(wraps=jobs.create_job)}):
            handler = tool_call.tool_list["create_job"]["handler"]
            result, prompt = self.run_create(["n"])
            handler.assert_not_called()
        self.assertFalse(result["success"])
        self.exists.assert_not_called()
        self.similar.assert_not_called()
        self.new_id.assert_not_called()
        self.insert.assert_not_called()
        self.assertEqual(prompt.call_count, 1)

    def test_database_error_preserves_generic_error_args(self):
        self.insert.side_effect = StorageError("Job creation failed", "storage detail")
        result, _ = self.run_create(["y"])
        self.assertEqual(result, {"success": False, "error_msg": "Job creation failed", "error_type": "StorageError", "error_args": ["storage detail"]})


if __name__ == "__main__":
    unittest.main()
