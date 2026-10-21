import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError

import loop
import provider
import skill
import tool_call
from errors import ToolError
from models import ModelProvider, ModelResponse, ModelToolCall, UserInput


class SkillRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "job-comparison"
        self.directory.mkdir()
        self.instructions = """---
name: job-comparison
description: >-
  Compare stored jobs by responsibilities
  and technologies.
license: MIT
metadata:
  author: tester
allowed-tools: pretend_tool
---
Instructions only loaded on demand.
"""
        (self.directory / "SKILL.md").write_text(self.instructions, encoding="utf-8")
        (self.directory / "references").mkdir()
        (self.directory / "references" / "guide.md").write_text("Compare explicit requirements.", encoding="utf-8")
        self.catalog = skill.discover_skills(self.root)
        self.handler = Mock(return_value={"approved": True})
        def pretend_tool():
            """Test-only trusted tool."""
            return self.handler()

        self.start_patch(patch.dict(tool_call.SKILL_TOOL_LIST, {
            "job-comparison": {"pretend_tool": {"handler": pretend_tool, "requires_approval": True}},
        }))

    def start_patch(self, patcher):
        self.addCleanup(patcher.stop)
        return patcher.start()

    def call(self, name, arguments, loaded=None):
        return tool_call.run_tool(name, arguments, "call_001", catalog=self.catalog, loaded_skills=loaded)

    def names(self, loaded):
        return {item["name"] for item in tool_call.get_tool_schemas(self.catalog, loaded)}

    def response(self, calls=()):
        return ModelResponse(None, list(calls), "completed", SimpleNamespace(output=[]), 1)

    def test_discovery_yaml_optional_metadata_and_progressive_reads(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("Body/resource must not be loaded")):
            catalog = skill.discover_skills(self.root)
        entry = catalog["job-comparison"]
        self.assertEqual(entry["description"], "Compare stored jobs by responsibilities and technologies.")
        self.assertEqual(entry["path"], self.directory / "SKILL.md")
        self.assertEqual(entry["frontmatter"]["metadata"], {"author": "tester"})
        self.assertEqual(entry["frontmatter"]["license"], "MIT")

    def test_required_metadata_directory_consistency_and_safe_yaml(self):
        invalid = (
            "name: wrong\ndescription: test", "name: Job-comparison\ndescription: test",
            "name: job--comparison\ndescription: test", "name: job-comparison",
            "name: job-comparison\ndescription: ''", "name: job-comparison\ndescription: 123",
            "!!python/object/apply:builtins.eval ['1+1']", "- not a mapping",
        )
        for frontmatter in invalid:
            with self.subTest(frontmatter=frontmatter):
                (self.directory / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nbody", encoding="utf-8")
                with self.assertRaises(ToolError):
                    skill.discover_skills(self.root)

    def test_multiline_yaml_delimiter_text_is_not_a_frontmatter_boundary(self):
        text = "---\nname: job-comparison\ndescription: |\n  Compare jobs.\n  ---\n  Keep requirements explicit.\n---\nInstructions"
        (self.directory / "SKILL.md").write_text(text, encoding="utf-8")
        catalog = skill.discover_skills(self.root)
        self.assertIn("---", catalog["job-comparison"]["description"])
        self.assertIn("Keep requirements explicit.", catalog["job-comparison"]["description"])

    def test_dynamic_enum_description_and_strict_schema(self):
        schemas = tool_call.get_tool_schemas(self.catalog, set())
        schema = next(item for item in schemas if item["name"] == "load_skill")
        self.assertEqual(schema["parameters"]["properties"]["name"]["enum"], ["job-comparison"])
        self.assertIn(self.catalog["job-comparison"]["description"], schema["description"])
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["parameters"]["additionalProperties"])
        self.assertNotIn("Instructions only loaded", schema["description"])
        self.assertNotIn("job-comparison", loop.construct_system_prompt(1, 0, 10, 3))
        self.assertNotIn("load_skill", {item["name"] for item in tool_call.get_tool_schemas({}, set())})

    def test_dynamic_skill_names_with_zero_one_and_multiple_skills(self):
        second = self.root / "second-skill"
        second.mkdir()
        (second / "SKILL.md").write_text(
            "---\nname: second-skill\ndescription: Second test Skill.\n---\nInstructions.",
            encoding="utf-8",
        )
        catalogs = ({}, self.catalog, skill.discover_skills(self.root))
        for catalog in catalogs:
            for loaded in (set(), {"missing"}, {"job-comparison"}, set(catalog)):
                with self.subTest(catalog=list(catalog), loaded=loaded):
                    active = set(catalog) & loaded
                    available = tool_call.get_available_tools(catalog, loaded)
                    schemas = {s["name"]: s for s in tool_call.get_tool_schemas(catalog, loaded)}
                    self.assertEqual(set(available), set(schemas))
                    for tool_name, allowed in (("load_skill", set(catalog)),
                                               ("read_skill_resource", active)):
                        if not allowed:
                            self.assertNotIn(tool_name, schemas)
                            self.assertFalse(tool_call.run_tool(
                                tool_name, {"name": "job-comparison", "path": "references/guide.md"},
                                "call", catalog=catalog, loaded_skills=loaded,
                            )["success"])
                            continue
                        field = schemas[tool_name]["parameters"]["properties"]["name"]
                        self.assertEqual(field["type"], "string")
                        self.assertEqual(field["enum"], sorted(allowed))
                        self.assertNotIn("const", field)
                        contract = tool_call.get_function_schema(tool_name, available[tool_name]["handler"])
                        model = contract.params_pydantic_model
                        extra = {"path": "references/guide.md"} if tool_name == "read_skill_resource" else {}
                        for name in allowed:
                            self.assertEqual(model.model_validate({"name": name, **extra}).name, name)
                        invalid = ["missing", "", 1, None, [], *set(catalog).difference(allowed)]
                        with patch.object(tool_call, "load_skill") as load, \
                                patch.object(tool_call, "read_skill_resource") as read, \
                                patch.object(tool_call, "request_tool_approval") as approve:
                            for name in invalid:
                                arguments = {"name": name, **extra}
                                with self.assertRaises(ValidationError):
                                    model.model_validate(arguments)
                                for payload in (arguments, json.dumps(arguments)):
                                    result = tool_call.run_tool(
                                        tool_name, payload, "call", catalog=catalog, loaded_skills=loaded,
                                    )
                                    self.assertFalse(result["success"])
                                    self.assertEqual(result["error_type"], "ToolError")
                                    self.assertIn("Invalid arguments", result["error_msg"])
                            load.assert_not_called()
                            read.assert_not_called()
                            approve.assert_not_called()

    def test_load_returns_complete_text_and_resource_requires_activation(self):
        result = self.call("load_skill", {"name": "job-comparison"})
        self.assertEqual(result, {"success": True, "result": self.instructions})
        self.assertFalse(self.call("read_skill_resource", {"name": "job-comparison", "path": "references/guide.md"})["success"])
        loaded = {"job-comparison"}
        self.assertEqual(self.call("read_skill_resource", {"name": "job-comparison", "path": "references/guide.md"}, loaded),
                         {"success": True, "result": "Compare explicit requirements."})

    def test_unknown_skills_and_outside_paths_are_rejected(self):
        self.assertEqual(self.call("load_skill", {"name": "missing"})["error_type"], "ToolError")
        outside = self.root / "secret.txt"
        outside.write_text("outside", encoding="utf-8")
        for path in ("../secret.txt", str(outside.resolve()), "references/../../secret.txt", "missing.txt"):
            with self.subTest(path=path):
                result = self.call("read_skill_resource", {"name": "job-comparison", "path": path}, {"job-comparison"})
                self.assertFalse(result["success"])
                self.assertEqual(result["error_type"], "ToolError")

    def test_resource_symlink_cannot_escape_skill_directory(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        link = self.directory / "link.txt"
        try:
            link.symlink_to(outside)
        except OSError:
            # Some Windows environments lack symlink privilege; still exercise
            # the resolved-path boundary with the same outside target.
            with patch.object(Path, "resolve", autospec=True, side_effect=lambda path: outside if path.name == "link.txt" else path.absolute()):
                with self.assertRaises(ToolError):
                    skill.read_skill_resource("job-comparison", "link.txt", self.catalog, {"job-comparison"})
        else:
            with self.assertRaises(ToolError):
                skill.read_skill_resource("job-comparison", "link.txt", self.catalog, {"job-comparison"})

    def test_scripts_are_only_text_not_execution_or_registration(self):
        scripts = self.directory / "scripts"
        scripts.mkdir()
        script = "raise RuntimeError('This script must never execute')"
        (scripts / "action.py").write_text(script, encoding="utf-8")
        with patch("subprocess.run", side_effect=AssertionError("No script execution")):
            skill.discover_skills(self.root)
            self.assertEqual(skill.read_skill_resource("job-comparison", "scripts/action.py", self.catalog, {"job-comparison"}), script)
        with patch.dict(tool_call.SKILL_TOOL_LIST, {}, clear=True):
            self.assertNotIn("pretend_tool", tool_call.get_available_tools(self.catalog, {"job-comparison"}))

    def test_skill_arguments_cannot_override_trusted_session_context(self):
        result = self.call("load_skill", {"name": "job-comparison", "catalog": {}})
        self.assertFalse(result["success"])
        result = self.call("read_skill_resource", {"name": "job-comparison", "path": "../secret.txt", "loaded_skills": {"job-comparison"}}, {"job-comparison"})
        self.assertFalse(result["success"])

    def test_successful_load_activates_only_next_batch(self):
        loaded = set()
        calls = [ModelToolCall("load_skill", {"name": "job-comparison"}, "load"),
                 ModelToolCall("pretend_tool", {}, "gated"),
                 ModelToolCall("read_skill_resource", {"name": "job-comparison", "path": "references/guide.md"}, "read")]
        outputs = loop.tool_loop(calls, ModelProvider.OPENAI, 1, 1, catalog=self.catalog, loaded_skills=loaded)
        self.assertEqual([output.output["success"] for output in outputs], [True, False, False])
        self.assertEqual(loaded, {"job-comparison"})
        self.handler.assert_not_called()
        with patch.object(tool_call, "request_tool_approval", return_value=True):
            outputs = loop.tool_loop(calls[1:2], ModelProvider.OPENAI, 1, 2, catalog=self.catalog, loaded_skills=loaded)
        self.assertTrue(outputs[0].output["success"])
        self.handler.assert_called_once_with()

    def test_failed_or_budget_rejected_load_does_not_activate(self):
        loaded = set()
        (self.directory / "SKILL.md").unlink()
        outputs = loop.tool_loop([ModelToolCall("load_skill", {"name": "job-comparison"}, "load")],
                                 ModelProvider.OPENAI, 1, 1, catalog=self.catalog, loaded_skills=loaded)
        self.assertFalse(outputs[0].output["success"])
        self.assertEqual(loaded, set())
        outputs = loop.tool_loop([ModelToolCall("load_skill", {"name": "job-comparison"}, "load")],
                                 ModelProvider.OPENAI, 1, 1, max_tool_call=0, catalog=self.catalog, loaded_skills=loaded)
        self.assertEqual(outputs[0].output["error_type"], "ToolBudgetExceeded")
        self.assertEqual(loaded, set())

    def test_registry_schema_availability_and_approval_are_consistent(self):
        baseline = tool_call.BASE_TOOL_LIST.copy()
        self.assertNotIn("pretend_tool", self.names(set()))
        self.assertFalse(self.call("pretend_tool", {})["success"])
        self.assertIn("pretend_tool", self.names({"job-comparison"}))
        self.assertNotIn("pretend_tool", self.names({"missing"}))
        with patch.object(tool_call, "request_tool_approval", return_value=False) as approve:
            self.assertFalse(self.call("pretend_tool", {}, {"job-comparison"})["success"])
        approve.assert_called_once_with("pretend_tool", {}, "call_001")
        self.handler.assert_not_called()
        self.assertEqual(tool_call.BASE_TOOL_LIST, baseline)
        self.assertTrue(tool_call.BASE_TOOL_LIST["create_job"]["requires_approval"])
        for loaded in (set(), {"job-comparison"}):
            self.assertEqual(self.names(loaded), set(tool_call.get_available_tools(self.catalog, loaded)))

    def test_provider_obtains_session_schemas_and_preserves_tool_outputs(self):
        fake = SimpleNamespace(status="completed", incomplete_details=None, output=[], output_text="done")
        with patch.object(provider, "OpenAI") as client:
            client.return_value.responses.create.return_value = fake
            provider.call_openai("model", [], "system", 1, "auto", catalog=self.catalog, loaded_skills=set())
            first = client.return_value.responses.create.call_args.kwargs["tools"]
            provider.call_openai("model", [], "system", 1, "auto", catalog=self.catalog, loaded_skills={"job-comparison"})
            second = client.return_value.responses.create.call_args.kwargs["tools"]
        self.assertNotIn("pretend_tool", {item["name"] for item in first})
        self.assertIn("pretend_tool", {item["name"] for item in second})
        outputs = loop.tool_loop([ModelToolCall("load_skill", {"name": "job-comparison"}, "load")],
                                 ModelProvider.OPENAI, 1, 1, catalog=self.catalog, loaded_skills=set())
        history = loop.adapt_provider_format(ModelProvider.OPENAI, outputs)
        self.assertIn(self.instructions.replace("\n", "\\n"), history[0]["output"])

    def test_conversation_sessions_do_not_share_activation(self):
        snapshots = []
        responses = iter([
            self.response([ModelToolCall("load_skill", {"name": "job-comparison"}, "load")]),
            self.response(), self.response(),
        ])

        def model(*args, **kwargs):
            snapshots.append(kwargs["loaded_skills"].copy())
            return next(responses)

        with patch.object(loop, "discover_skills", return_value=self.catalog), \
                patch.object(loop, "get_user_input", side_effect=[UserInput("compare", None, 1), None, UserInput("browse", None, 1), None]), \
                patch.object(loop, "call_model", side_effect=model), redirect_stdout(io.StringIO()):
            loop.conversation_loop()
            loop.conversation_loop()
        self.assertEqual(snapshots, [set(), {"job-comparison"}, set()])


if __name__ == "__main__":
    unittest.main()
