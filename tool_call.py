from tools.jobs import lookup_job, create_job, list_jobs, search_jobs_hybrid
import json
from approval import request_tool_approval
from errors import HarnessError
from tools.evidence import search_evidence_hybrid

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


tools = [
    {
        "type": "function",
        "name": "search_jobs_hybrid",
        "description": (
            "Find stored jobs by natural-language role preferences, technologies and skills. "
            "Use meaningful English semantic queries for English job records and distinctive "
            "technical keywords for lexical retrieval. Rank is retrieval relevance, not candidate fit. "
            "Use list_jobs for browsing with exact filters and pagination; lookup_job for one known ID. "
            "Company and location are optional exact filters. An outdated index must be synchronized "
            "by the developer before searching."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Nonempty English semantic search query."},
                "keywords": {"type": "array", "items": {"type": "string"},
                             "description": "Distinctive lexical anchors such as Python, RAG, LLM APIs; [] is allowed."},
                "company": {"type": ["string", "null"], "description": "Exact company filter; null disables it."},
                "location": {"type": ["string", "null"], "description": "Exact location filter; null disables it."},
                "top_k": {"type": ["integer", "null"], "description": "Number of distinct jobs, 1 to 20; null defaults to 10."},
            },
            "required": ["query", "keywords", "company", "location", "top_k"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_jobs",
        "description": (
            "浏览已有岗位，不需要预先知道岗位 ID；已知单个 ID 时使用 lookup_job。"
            "按岗位 ID 的数字后缀升序分页。has_more 为 true 时，"
            "将 next_cursor 作为下一次的 after_job_id，并保持相同筛选条件。"
            "company 和 location 是可独立使用或组合使用的精确筛选条件；"
            "传 null 表示不筛选。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": ["integer", "null"],
                    "description": "每页数量，1 到 50；null 时默认 20。",
                },
                "after_job_id": {
                    "type": ["string", "null"],
                    "description": "返回数字后缀大于此 ID 的岗位，例如 job_010；null 从头开始。ID 无须存在。",
                },
                "company": {
                    "type": ["string", "null"],
                    "description": "公司名称精确筛选；null 不筛选。",
                },
                "location": {
                    "type": ["string", "null"],
                    "description": "地点精确筛选；null 不筛选。",
                },
            },
            "required": ["limit", "after_job_id", "company", "location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
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



