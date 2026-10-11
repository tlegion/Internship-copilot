from tools.jobs import lookup_job, create_job
import json
from approval import request_tool_approval
from errors import HarnessError
from tools.evidence import search_evidence_hybrid

tool_list= {
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
    },
    {
        "type": "function",
        "name": "search_evidence_hybrid",
        "description": (
            "Search the candidate's evidence when answering questions "
            "about their experience, skills, projects, or qualifications. "
            "Use query as a standalone semantic description of the user's "
            "current retrieval intent. If the user's current message is "
            "already self-contained, preserve it closely. If it depends on "
            "earlier conversation, include only the context needed to make "
            "the query standalone. "
            "keywords are lexical search anchors for exact matching. "
            "Use only distinctive technical terms, technology names, "
            "skills, identifiers, error codes, product names, or short "
            "phrases relevant to the user's request. Exclude generic words "
            "such as 'I', 'have', 'did', 'use', 'experience', and 'about'. "
            "Do not invent terms that assume the answer. "
            "If there are no useful exact lexical anchors, pass an empty list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Standalone semantic search query representing the user's current "
                        "retrieval intent. The current evidence corpus is in English, so "
                        "write this query in English even if the user speaks another language. "
                        "Preserve the user's meaning and do not add unsupported facts. "
                        "If the current user message depends on earlier conversation, include "
                        "only the context needed to make the query standalone."
                    ),
                },
                "keywords": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": (
                        "Distinctive English lexical search anchors for the current English "
                        "evidence corpus. Preserve exact technical identifiers, API names, "
                        "technology names, error codes, and code terms as written. "
                        "Exclude generic conversational words. Do not invent likely answer "
                        "terms. If there are no useful exact lexical anchors, use an empty list."
                    ),
                },
                "candidate_id": {
                    "type": "string",
                    "description": (
                        "Candidate whose evidence should be searched, "
                        "for example candidate_001."
                    ),
                },
            },
            "required": [
                "query",
                "keywords",
                "candidate_id",
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
                "error_msg": f"Authorization error: User approval failed for {tool_name} at {call_id} with arguments {tool_input}",
                "error_type": "ToolError"
            }
        try:
            result = tool["handler"](**tool_input)
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



