import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from enum import Enum
from typing import Any
from provider import call_model, format_tool_output, format_user_input, load_tool_arguments
from tool_call import run_tool
from conversation import get_user_input, display_history
from models import ModelProvider, ModelResponse, ModelToolCall, UserInput, ToolOutput

load_dotenv(Path(__file__).with_name('.env'))



def tool_loop(tool_calls:list[ModelToolCall], provider:ModelProvider, round:int,  max_tool_call:int = 10):
    tool_outputs:list[ToolOutput] = []
    call_count = 0
    for call in tool_calls:
        result= run_tool(call.tool_name, load_tool_arguments(call.arguments, provider))
        tool_outputs.append(ToolOutput(
            tool_name=call.tool_name,
            call_id=call.call_id,
            output=result,
            round= round
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
        elif isinstance(item, ToolOutput):
            formatted.append(format_tool_output(item, provider))

    return formatted



def conversation_loop(provider:ModelProvider= ModelProvider.OPENAI, model:str= "gpt-6-luna", max_round:int = 10):
    history = []
    user_ended = False
    round = 1


    while not user_ended and round <= max_round:
        display_history(history)
        user_input = get_user_input(round)
        history.append(user_input)
        response = call_model(provider, model, round, adapt_provider_format(provider, history))
        if getattr(response, "error_msg", None):
            #  place holder
            history.extend(response)
            continue

        history.extend(response)

        if tool_calls := getattr(response, "tool_calls", None):
            tool_outputs= tool_loop(tool_calls, provider)

            history.extend(format_tool_output(tool_outputs, provider))

        round += 1



def start_conversation_OpenAI():
    conversation_loop()

if __name__ == "__main__":
    start_conversation_OpenAI()