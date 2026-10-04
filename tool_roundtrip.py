import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))
client = OpenAI(timeout=30, max_retries=1)

JOBS = {
    "job_001": {
        "title": "AI Engineering Intern",
        "requirements": ["Python", "API integration", "SQL"]
    }
}

def lookup_job(input):
    job_id = input.get("job_id", "")
    if not job_id:
        return { "error_msg": f"parameter error: job_id not found"}
    job = JOBS.get(job_id)
    if job is None:
        return {"error_msg": f"Job not found: {job_id}"}
    return job

tools = [{
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
}]

def run_tool(tool_name, tool_input):
    if tool_name == "lookup_job":
        return lookup_job(tool_input)

    return { "error_msg": f"tool name not found: {tool_name}" }


history = [{
    "role": "user",
    "content": "查询 job_001，告诉我这个岗位需要什么技能。",
}]

response = client.responses.create(
    model= "gpt-6-luna",
    input = history,
    tools= tools,
    tool_choice= "auto",
    parallel_tool_calls= True,
    reasoning={"effort": "none"},
    max_output_tokens= 256,
    store= False
)

history.extend(response.output)


for item in response.output:
    if item.type != "function_call":
        continue

    print("工具请求：", item.name, item.arguments)
    parameters= json.loads(item.arguments)
    print(parameters)
    result = run_tool(item.name, parameters)
    print("工具输出：", result)

    history.append({
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": json.dumps(result, ensure_ascii=False)
    })

final = client.responses.create(
    model="gpt-6-luna",
    input=history,
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls= True,
    reasoning={"effort": "none"},
    max_output_tokens= 256,
    store= False
)

print("Response:", final.output_text)