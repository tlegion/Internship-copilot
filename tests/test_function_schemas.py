import inspect
import json
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

import provider
import tool_call
from errors import StorageError
from tools import jobs


class Detail(BaseModel):
    label: str


class FunctionSchemaTests(unittest.TestCase):
    def test_base_schemas_follow_signatures_and_english_docstrings(self):
        schemas = tool_call.get_tool_schemas()
        self.assertEqual({s["name"] for s in schemas}, set(tool_call.BASE_TOOL_LIST))
        for schema in schemas:
            with self.subTest(name=schema["name"]):
                handler = tool_call.BASE_TOOL_LIST[schema["name"]]["handler"]
                params = schema["parameters"]
                self.assertTrue(schema["strict"])
                self.assertFalse(params["additionalProperties"])
                self.assertEqual(set(params["properties"]), set(inspect.signature(handler).parameters))
                self.assertEqual(set(params["required"]), set(params["properties"]))
                self.assertIsNone(re.search(r"[\u4e00-\u9fff]", json.dumps(schema, ensure_ascii=False)))
                self.assertTrue(schema["description"])
                for field in params["properties"].values():
                    self.assertTrue(field["description"])
        lookup = next(s for s in schemas if s["name"] == "lookup_job")
        self.assertEqual(lookup["parameters"]["properties"]["job_id"]["type"], "string")
        self.assertEqual(lookup["parameters"]["properties"]["job_id"]["description"], "Job ID, for example job_001")

    def test_invalid_arguments_do_not_reach_approval_or_business_handler(self):
        valid = dict(title="Intern", company="Demo", description="Build tools", location=None)
        invalid = [
            {}, {k: v for k, v in valid.items() if k != "location"},
            {**valid, "title": 123}, {**valid, "location": []},
            {**valid, "extra": True}, None, [], 1, True,
            "{", "null", "[]", '"{}"',
        ]
        with patch.object(tool_call, "request_tool_approval") as approve, \
                patch.object(jobs, "check_exists") as business:
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    result = tool_call.run_tool("create_job", arguments, "call")
                    self.assertFalse(result["success"])
                    self.assertEqual(result["error_type"], "ToolError")
                    self.assertTrue(result["error_msg"].startswith("Invalid arguments"))
            approve.assert_not_called()
            business.assert_not_called()

    def test_nested_wrong_types_and_constrained_values_fail_before_handler(self):
        with patch.object(jobs, "get_jobs_page") as page:
            for limit in (0, 51, True, "2", 2.0):
                self.assertFalse(tool_call.run_tool("list_jobs", {"limit": limit}, "call")["success"])
            page.assert_not_called()
        with patch("tools.evidence.hybrid_search") as search:
            result = tool_call.run_tool("search_evidence_hybrid", {
                "query": "Python", "keywords": [1], "candidate_id": "candidate_001",
            }, "call")
            self.assertFalse(result["success"])
            search.assert_not_called()

    def test_defaults_and_null_values_reach_handler_unchanged(self):
        with patch.object(jobs, "get_jobs_page", return_value={}) as page:
            self.assertTrue(tool_call.run_tool("list_jobs", {}, "call")["success"])
            page.assert_called_once_with(20, None, None, None)
        model = tool_call.get_function_schema("list_jobs", jobs.list_jobs).params_pydantic_model
        self.assertEqual(model.model_validate({}).limit, 20)
        self.assertIsNone(model.model_validate({"limit": None}).limit)

    def test_successful_approval_executes_and_business_errors_remain_distinct(self):
        arguments = dict(title="Intern", company="Demo", description="Build tools", location=None)
        with patch.object(tool_call, "request_tool_approval", return_value=True) as approve, \
                patch.object(jobs, "check_exists", side_effect=StorageError("Unavailable", "detail")) as business:
            result = tool_call.run_tool("create_job", json.dumps(arguments), "call")
            approve.assert_called_once_with("create_job", arguments, "call")
            business.assert_called_once_with(*arguments.values())
            self.assertEqual(result, {"success": False, "error_type": "StorageError",
                                      "error_msg": "Unavailable", "error_args": ["detail"]})

    def test_signature_changes_update_schema_and_validator_together(self):
        def original(value: str):
            """Original tool."""
            return value

        def changed(value: int, note: str | None = None):
            """Changed tool."""
            return value, note

        with patch.dict(tool_call.BASE_TOOL_LIST, {"example": {"handler": original, "requires_approval": False}}):
            before = next(s for s in tool_call.get_tool_schemas() if s["name"] == "example")
            self.assertEqual(before["parameters"]["properties"]["value"]["type"], "string")
            tool_call.BASE_TOOL_LIST["example"]["handler"] = changed
            after = next(s for s in tool_call.get_tool_schemas() if s["name"] == "example")
            self.assertEqual(after["parameters"]["properties"]["value"]["type"], "integer")
            self.assertIn("note", after["parameters"]["required"])
            self.assertFalse(tool_call.run_tool("example", {"value": "2"}, "call")["success"])
            self.assertEqual(tool_call.run_tool("example", {"value": 2}, "call")["result"], (2, None))

    def test_to_call_args_preserves_nested_python_types(self):
        def nested(detail: Detail, tags: list[str], note: str | None):
            """Inspect nested parameters."""
            self.assertIsInstance(detail, Detail)
            self.assertIsInstance(tags, list)
            self.assertIsNone(note)
            return detail.label

        with patch.dict(tool_call.BASE_TOOL_LIST, {"nested": {"handler": nested, "requires_approval": False}}):
            result = tool_call.run_tool("nested", {"detail": {"label": "ok"}, "tags": ["a"], "note": None}, "call")
            self.assertEqual(result, {"success": True, "result": "ok"})

    def test_multiple_skill_enums_validate_against_current_catalog(self):
        catalog = {"one": {"description": "First"}, "two": {"description": "Second"}}
        schema = next(s for s in tool_call.get_tool_schemas(catalog) if s["name"] == "load_skill")
        self.assertEqual(schema["parameters"]["properties"]["name"]["enum"], ["one", "two"])
        with patch.object(tool_call, "load_skill") as load:
            self.assertFalse(tool_call.run_tool("load_skill", {"name": "missing"}, "call", catalog=catalog)["success"])
            load.assert_not_called()

    def test_malformed_provider_arguments_return_tool_errors(self):
        for raw in ("{", "[]", "null", '"{}"'):
            with self.subTest(raw=raw):
                response = SimpleNamespace(status="completed", incomplete_details=None, output_text="",
                    output=[SimpleNamespace(type="function_call", name="lookup_job", arguments=raw, call_id="call")])
                with patch.object(provider, "OpenAI") as client:
                    client.return_value.responses.create.return_value = response
                    call = provider.call_openai("model", [], "system", 1, "auto").tool_calls[0]
                result = tool_call.run_tool(call.tool_name, call.arguments, call.call_id)
                self.assertFalse(result["success"])
                self.assertEqual(result["error_type"], "ToolError")
