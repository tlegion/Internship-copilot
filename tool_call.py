from typing import Literal

from agents.function_schema import function_schema
from pydantic import ValidationError

from tools.jobs import lookup_job, create_job, list_jobs, search_jobs_hybrid
from approval import request_tool_approval
from errors import HarnessError
from tools.evidence import search_evidence_hybrid
from skill import load_skill, read_skill_resource

tool_list= {
    "search_jobs_hybrid": {
        "handler": search_jobs_hybrid,
        "requires_approval": False,
    },
    "list_jobs": {
        "handler": list_jobs,
        "requires_approval": False,
    },
    "lookup_job": {
        "handler": lookup_job,
        "requires_approval": False
    },
    "create_job": {
        "handler": create_job,
        "requires_approval": True
    },
    "search_evidence_hybrid": {
        "handler": search_evidence_hybrid,
        "requires_approval": False,
    },
}


BASE_TOOL_LIST = tool_list

# Developer-maintained executable registrations; Skill files cannot add these.
SKILL_TOOL_LIST = {}


def get_function_schema(name, handler):
    """Derive both the API contract and runtime validator from a trusted handler."""
    schema = function_schema(handler, name_override=name, docstring_style="google")
    # The SDK makes JSON strict, but its generated model otherwise ignores extras
    # and coerces values. Enforce the same boundary at runtime as well.
    schema.params_pydantic_model.model_config.update(extra="forbid", strict=True)
    schema.params_pydantic_model.model_rebuild(force=True)
    # Pydantic emits const for a single Literal; retain the existing Skill enum format.
    for field in schema.params_json_schema.get("properties", {}).values():
        if "const" in field:
            field["enum"] = [field.pop("const")]
    return schema


def get_available_tools(catalog: dict | None = None, loaded_skills: set | None = None) -> dict:
    catalog = catalog or {}
    loaded = set(loaded_skills or ()) & catalog.keys()
    available = BASE_TOOL_LIST.copy()
    if catalog:
        def load(name: Literal[tuple(sorted(catalog))]) -> str:
            """Load relevant Skill instructions before using them. Available Skills:"""
            return load_skill(name, catalog)

        descriptions = "\n".join(
            f"{name}: {entry['description']}" for name, entry in sorted(catalog.items())
        )
        load.__doc__ += "\n" + descriptions

        available["load_skill"] = {
            "handler": load,
            "requires_approval": False,
        }
    if loaded:
        def read(name: Literal[tuple(sorted(loaded))], path: str) -> str:
            """Read a supporting UTF-8 text file inside an already loaded Skill. Scripts are text only and never execute.

            Args:
                path: Relative path inside the Skill directory.
            """
            return read_skill_resource(name, path, catalog, loaded)

        available["read_skill_resource"] = {
            "handler": read,
            "requires_approval": False,
        }
    for name in sorted(loaded):
        available.update(SKILL_TOOL_LIST.get(name, {}))
    return available


def get_tool_schemas(catalog: dict | None = None, loaded_skills: set | None = None) -> list[dict]:
    schemas = []
    for name, tool in get_available_tools(catalog, loaded_skills).items():
        schema = get_function_schema(name, tool["handler"])
        schemas.append({
            "type": "function",
            "name": name,
            "description": schema.description,
            "parameters": schema.params_json_schema,
            "strict": True,
        })
    return schemas


# Backward-compatible base-tool views; parameter definitions are generated.
tools = get_tool_schemas()
BASE_TOOL_SCHEMAS = tools


def run_tool(tool_name:str, tool_input:dict | str, call_id:str, *,
             catalog: dict | None = None, loaded_skills: set | None = None):
    tool = get_available_tools(catalog, loaded_skills).get(tool_name)

    if tool:
        schema = get_function_schema(tool_name, tool["handler"])
        try:
            model = schema.params_pydantic_model
            if isinstance(tool_input, str):
                validated = model.model_validate_json(tool_input, strict=True)
            else:
                validated = model.model_validate(tool_input, strict=True)
            args, kwargs = schema.to_call_args(validated)
        except ValidationError as error:
            return {
                "success": False,
                "error_msg": f"Invalid arguments for {tool_name}: {error}",
                "error_type": "ToolError",
                "error_args": [],
            }
        # Use field values directly; model_dump would turn nested models into dicts.
        tool_input = {name: getattr(validated, name) for name in model.model_fields}
        if tool["requires_approval"] and not request_tool_approval(tool_name, tool_input, call_id):
            return {
                "success": False,
                "error_msg": f"Authorization error: User approval failed for {tool_name} at {call_id} with arguments {tool_input}",
                "error_type": "ToolError"
            }
        try:
            result = tool["handler"](*args, **kwargs)
        except HarnessError as error:
            return {
                "success": False,
                "error_msg": str(error.args[0]),
                "error_type": type(error).__name__,
                "error_args": list(error.args[1:])
            }

        return {
            "success": True,
            "result": result
        }
    else:
        return { 
            "success": False,
            "error_msg": f"tool name not found: {tool_name}",
            "error_type": "ToolError"
        }
