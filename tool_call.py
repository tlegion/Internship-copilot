from tools.jobs import lookup_job, create_job
import json
from conversation import request_tool_approval

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
                "job_id": {
                    "type": "string",
                    "description": "岗位 ID，例如 job_001"
                }
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "create_job",
        "description": (
            "根据用户提供的岗位信息创建一个新的岗位记录。"
            "只有当输入中能够明确提取岗位标题、公司、岗位描述时才调用。"
            "不要自行生成 job_id; job_id 由系统内部生成。"
            "如果地点没有提供，则 location 传 null。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "岗位标题，例如 Software Engineering Intern"
                },
                "company": {
                    "type": "string",
                    "description": "公司名称"
                },
                "description": {
                    "type": "string",
                    "description": "岗位职责或岗位描述，只使用用户提供的信息，不要补充未提供的要求"
                },
                "location": {
                    "type": ["string", "null"],
                    "description": "岗位地点；如果原始信息没有明确提供，则传 null"
                }
            },
            "required": [
                "title",
                "company",
                "description",
                "location"
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }
]
def run_tool(tool_name:str, tool_input:dict, call_id:str):
    tool = tool_list.get(tool_name)

    if tool:
        if tool["requires_approval"] and not request_tool_approval(tool_name, tool_input, call_id):
            return {
                "success": False,
                "error_msg": f"Authorization error: User approval failed for {tool_name} at {call_id} with arguments {tool_input}"
            }
        return tool["handler"](**tool_input)
    else:
        return { 
            "success": False,
            "error_msg": f"tool name not found: {tool_name}" 
        }



