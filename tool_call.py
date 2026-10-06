from tools.jobs import lookup_job, create_job
import json

tool_list= {
    "lookup_job": {
        "handler": lookup_job,
        "requires_approval": False
    },
    "create_job": {
        "handler": create_job,
        "requires_approval": True
    }
}


tools = [
    {
        "type": "function",
        "name": "lookup_job",
        "description": "根据岗位 ID 查询岗位详情。",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"}
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

def run_tool(tool_name, tool_input:dict):
    tool = tool_list.get(tool_name)

    if tool:
        if tool["requires_approval"]:
            if 
        return tool(**tool_input)
    else:
        return { "error_msg": f"tool name not found: {tool_name}" }



