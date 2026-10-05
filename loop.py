import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from enum import Enum
from typing import Any
from provider import call_model, format_tool_output, format_user_input, load_tool_arguments
from tool_call import run_tool
from conversation import get_user_input

load_dotenv(Path(__file__).with_name('.env'))

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthroopic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"

class UserInput:
    role = "user"
    text: str
    file: Any

    def __init__(self, text, file):
        self.text = text
        self.file = file

class ModelToolCall:
    tool_name: str
    arguments: dict
    call_id: str

    def __init__(
        self,
        tool_name: str,
        arguments: dict,
        call_id: str,
    ):
        self.tool_name = tool_name
        self.arguments = arguments
        self.call_id = call_id


class ModelResponse:
    role = "assistant"
    text: str | None
    tool_calls: list[ModelToolCall]
    status: str
    raw: Any

    def __init__(
        self,
        text: str | None,
        tool_calls: list[ModelToolCall],
        status: str,
        raw: Any,
    ):
        self.text = text
        self.tool_calls = tool_calls
        self.status = status
        self.raw = raw

class ToolOutput:
    call_id: str
    outputs: Any
    tool_name: str

    def __init__(
        self,
        tool_name: str,
        call_id: str,
        output: Any,
    ):
        self.tool_name = tool_name
        self.call_id = call_id
        self.output = output



def tool_loop(tool_calls:list[ModelToolCall], provider:ModelProvider, max_tool_call:int = 10):
    tool_outputs:list[ToolOutput] = []
    call_count = 0
    for call in tool_calls:
        result= run_tool(call.tool_name, load_tool_arguments(call.arguments, provider))
        tool_calls.append(ToolOutput(
            tool_name=call.tool_name,
            call_id=call.call_id,
            output=result
        ))
        call_count += 1
        if call_count >= max_tool_call:
            break
    return tool_outputs


def adapt_provider_format(provider:ModelProvider, history:list):
    formatted = []

    for item in history:
        if isinstance(item, UserInput):
            formatted.append(format_user_input(item, provider))
        elif isinstance(item,ModelResponse):
            if provider == ModelProvider.OPENAI:
                formatted.extend(item.raw.output)
        elif isinstance(item, list[ToolOutput]):
            formatted.append(format_tool_output(item, provider))

    return formatted



def conversation_loop(provider:ModelProvider, model:str= "gpt-6-luna", max_round:int = 10):
    history = []
    user_ended = False
    round = 0


    while not user_ended and round <= max_round:
        input = get_user_input()
        response = call_model(provider, model, adapt_provider_format(provider, history))
        if getattr(response, "error_msg", None):
            #  place holder
            history.extend(response)
            continue

        history.extend(response)

        if tool_calls := getattr(response, "tool_calls", None):
           tool_outputs= tool_loop(tool_calls, provider)

        history.extend(format_tool_output(tool_outputs, provider))

        round += 1



