from models import ModelProvider, ModelResponse, ModelToolCall, UserInput, ToolOutput

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, APIError
import os
import json
from tool_call import tools

def call_model(
        provider: ModelProvider, 
        model:str, history:list, 
        system_prompt:str,
        round:int,
        max_output_tokens:int = 128, 
        reasoning_effort:str = "none",
    ):
    load_dotenv(Path(__file__).with_name(".env"))
    if provider == ModelProvider.OPENAI and os.environ.get("OPENAI_API_KEY"):
        return call_openai(model, history, system_prompt, round, max_output_tokens, reasoning_effort)

    return {
        "error_msg": f"missing API key: {provider}"
    }

    
def call_openai(       
        model:str, history:list, 
        system_prompt:str,
        round:int,
        max_output_tokens:int = 128, 
        reasoning_effort:str = "none"
    ):
    client =OpenAI(timeout=30, max_retries=1)
    try:
        response= client.responses.create(
            model=model,
            input=history,
            instructions=system_prompt,
            tools=tools,
            tool_choice="auto",
            reasoning={"effort":reasoning_effort},
            max_output_tokens=max_output_tokens,
            store=False
        )
    except APIError as error:
        return { 
            "error_msg": error.message,
            "error_body": error.body,
            "code": error.code
        }

    tool_calls:list[ModelToolCall] = []

    for item in response.output:
        if item.type == "function_call":
            tool_calls.append(ModelToolCall(
                tool_name=item.name,
                arguments=item.arguments,
                call_id= item.call_id
            ))


    return ModelResponse(
        text=response.output_text or None,
        tool_calls= tool_calls,
        status=response.status,
        raw=response,
        round=round
    )


def load_tool_arguments(arguments, provider: ModelProvider):
    if provider == ModelProvider.OPENAI:
        return json.loads(arguments)

def tool_output_OpenAI(output:ToolOutput):
    return {
        "type": "function_call_output",
        "call_id": output.call_id,
        "output": json.dumps(output.output)
    }

def format_tool_output(tool_outputs:ToolOutput, provider: ModelProvider):

    if provider == ModelProvider.OPENAI:
        return tool_output_OpenAI(tool_outputs)


def user_input_OpenAI(user_input:UserInput):
    return {
        "type": "message",
        "role": user_input.role,
        "content": [
            {
                "type": "input_text",
                "text": user_input.text
            }
        ]
    }

def format_user_input(user_input:UserInput, provider: ModelProvider):
    if provider == ModelProvider.OPENAI:
        return user_input_OpenAI(user_input)


# def tool_call_OpenAI(tool_input):
#     return {
#         "type": "functioncall"
#     }


# def format_tool_call(tool_calls:list[ModelToolCall], provider: ModelProvider):
#     formatted = []
#     for item in tool_calls:
#         if provider == ModelProvider.OPENAI:
            # formatted.append(tool_call_OpenAI(item))
