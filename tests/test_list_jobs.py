import importlib.util
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from errors import StorageError, ToolError
from storages import database
from tools import jobs


# Keep dispatcher tests offline without importing the unrelated RAG tokenizer.
evidence_stub = ModuleType("tools.evidence")
evidence_handler = Mock()


def search_evidence_hybrid(query: str, keywords: list[str], candidate_id: str):
    return evidence_handler(query=query, keywords=keywords, candidate_id=candidate_id)


evidence_stub.search_evidence_hybrid = search_evidence_hybrid
spec = importlib.util.spec_from_file_location(
    "_list_jobs_tool_call", Path(__file__).resolve().parents[1] / "tool_call.py"
)
tool_call = importlib.util.module_from_spec(spec)
with patch.dict("sys.modules", {"tools.evidence": evidence_stub}):
    spec.loader.exec_module(tool_call)


class ListJobsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "jobs.db"
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executescript(
                (Path(__file__).resolve().parents[1] / "sql/migrations/V000__initial.sql").read_text()
            )
            connection.executemany(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
                [(f"job_{i:03d}", f"Intern {i}", "Demo" if i % 2 else "Other",
                  f"Full description {i}\nSecond line", "Sydney" if i % 3 else None)
                 for i in [*range(1, 26), 999, 1000, 1002]],
            )
        patcher = patch.object(database, "DB_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def ids(self, result):
        return [job["job_id"] for job in result["jobs"]]

    def test_default_and_null_limit_and_complete_records(self):
        result = jobs.list_jobs()
        self.assertEqual(len(result["jobs"]), 20)
        self.assertTrue(result["has_more"])
        self.assertEqual(result["next_cursor"], "job_020")
        self.assertEqual(result, jobs.list_jobs(limit=None))
        self.assertEqual(json.loads(json.dumps(result)), result)
        for job in result["jobs"]:
            self.assertEqual(set(job), {"job_id", "title", "company", "description", "location"})
            self.assertIn("\nSecond line", job["description"])

    def test_two_item_pages_do_not_repeat(self):
        first = jobs.list_jobs(limit=2)
        second = jobs.list_jobs(limit=2, after_job_id=first["next_cursor"])
        self.assertEqual(self.ids(first), ["job_001", "job_002"])
        self.assertEqual(self.ids(second), ["job_003", "job_004"])
        self.assertTrue(second["has_more"])
        self.assertEqual(second["next_cursor"], "job_004")

    def test_numeric_order_gaps_and_nonexistent_cursor(self):
        result = jobs.list_jobs(limit=2, after_job_id="job_998")
        self.assertEqual(self.ids(result), ["job_999", "job_1000"])
        self.assertTrue(result["has_more"])
        final = jobs.list_jobs(limit=2, after_job_id=result["next_cursor"])
        self.assertEqual(self.ids(final), ["job_1002"])
        self.assertFalse(final["has_more"])
        self.assertIsNone(final["next_cursor"])
        self.assertEqual(self.ids(jobs.list_jobs(after_job_id="job_1001")), ["job_1002"])

    def test_exact_page_size_is_final(self):
        result = jobs.list_jobs(limit=2, after_job_id="job_999")
        self.assertEqual(self.ids(result), ["job_1000", "job_1002"])
        self.assertFalse(result["has_more"])
        self.assertIsNone(result["next_cursor"])

    def test_independent_and_combined_filters(self):
        for filters in ({"company": "Demo"}, {"location": "Sydney"},
                        {"company": "Demo", "location": "Sydney"}):
            with self.subTest(filters=filters):
                result = jobs.list_jobs(limit=50, **filters)
                self.assertTrue(result["jobs"])
                for job in result["jobs"]:
                    for field, value in filters.items():
                        self.assertEqual(job[field], value)
        page = jobs.list_jobs(limit=2, company="Demo", location="Sydney")
        following = jobs.list_jobs(limit=2, after_job_id=page["next_cursor"], company="Demo", location="Sydney")
        self.assertEqual(self.ids(page), ["job_001", "job_005"])
        self.assertEqual(self.ids(following), ["job_007", "job_011"])

    def test_empty_results_and_bound_filter_values(self):
        empty = {"jobs": [], "has_more": False, "next_cursor": None}
        for filters in ({"company": "Missing"}, {"location": "Missing"},
                        {"company": "Demo' OR 1=1 --"}, {"after_job_id": "job_9999"}):
            with self.subTest(filters=filters):
                self.assertEqual(jobs.list_jobs(**filters), empty)

    def test_invalid_limits_and_cursors_fail_before_storage(self):
        with patch.object(jobs, "get_jobs_page") as query:
            for limit in (0, -1, 51, "2", 2.0, True, [], {}):
                with self.subTest(limit=limit), self.assertRaises(ToolError):
                    jobs.list_jobs(limit=limit)
            for cursor in ("", "job_", "job_-1", "job_1x", "JOB_001", "job_1\n", "job_１", 1, []):
                with self.subTest(cursor=cursor), self.assertRaises(ToolError):
                    jobs.list_jobs(after_job_id=cursor)
            query.assert_not_called()

    def test_database_failures_follow_storage_error_contract(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("DROP TABLE jobs")
        with self.assertRaises(StorageError):
            jobs.list_jobs()

    def test_dispatch_is_read_only_and_serializable(self):
        with patch.object(tool_call, "request_tool_approval") as approve:
            result = tool_call.run_tool("list_jobs", dict(limit=None, after_job_id=None, company=None, location=None), "call_001")
        approve.assert_not_called()
        self.assertTrue(result["success"])
        self.assertEqual(len(result["result"]["jobs"]), 20)
        self.assertEqual(json.loads(json.dumps(result)), result)
        invalid = tool_call.run_tool("list_jobs", {"limit": 0}, "call_002")
        self.assertFalse(invalid["success"])
        self.assertEqual(invalid["error_type"], "ToolError")
        self.assertEqual(invalid["error_args"], [])

    def test_strict_schema_and_existing_registrations(self):
        schema = next(tool for tool in tool_call.tools if tool["name"] == "list_jobs")
        parameters = schema["parameters"]
        self.assertTrue(schema["strict"])
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(set(parameters["required"]), set(parameters["properties"]))
        self.assertEqual(parameters["properties"]["limit"]["anyOf"], [{"type": "integer", "minimum": 1, "maximum": 50}, {"type": "null"}])
        for field in ("after_job_id", "company", "location"):
            self.assertEqual(parameters["properties"][field]["anyOf"], [{"type": "string"}, {"type": "null"}])
        self.assertIs(tool_call.tool_list["list_jobs"]["handler"], jobs.list_jobs)
        self.assertFalse(tool_call.tool_list["list_jobs"]["requires_approval"])
        self.assertIs(tool_call.tool_list["lookup_job"]["handler"], jobs.lookup_job)
        self.assertIs(tool_call.tool_list["create_job"]["handler"], jobs.create_job)
        self.assertTrue(tool_call.tool_list["create_job"]["requires_approval"])
        arguments = dict(query="Python", keywords=["Python"], candidate_id="candidate_001")
        evidence_handler.return_value = [{"text": "evidence"}]
        self.assertEqual(tool_call.run_tool("search_evidence_hybrid", arguments, "call_003"),
                         {"success": True, "result": [{"text": "evidence"}]})
        evidence_handler.assert_called_once_with(**arguments)

    def test_lookup_job_retains_original_response(self):
        self.assertEqual(jobs.lookup_job("job_001"), {
            "id": "job_001", "title": "Intern 1", "company": "Demo",
            "description": "Full description 1\nSecond line", "location": "Sydney",
        })


if __name__ == "__main__":
    unittest.main()
