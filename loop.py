import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from enum import Enum
from typing import Any
from provider import call_model, format_tool_output, format_user_input, load_tool_arguments
from tool_call import run_tool, tool_list, tools
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

def construct_system_prompt():
    return """
You are an internship assistant agent.

Your job is to help the user retrieve and reason about internship-related information accurately.

Rules:
- Use the available tools whenever the user's request requires information provided by a tool.
- Never invent tool results or claim that a tool returned information that it did not return.
- Use tool outputs as the source of truth for tool-provided data.
- When a tool call is required, provide only the arguments required by that tool.
- After receiving tool output, continue the task and give the user a clear final answer.
- If a tool fails or returns an error, do not fabricate a replacement result. Explain the limitation clearly.
- Keep responses concise and directly relevant to the user's request.
""".strip()


def conversation_loop(provider:ModelProvider= ModelProvider.OPENAI, model:str= "gpt-6-luna", max_round:int = 10):
    history = []
    user_ended = False
    round = 1


    while not user_ended and round <= max_round:
        display_history(history)
        user_input = get_user_input(round)
        history.append(user_input)
        response = call_model(
            provider, model, 
            adapt_provider_format(provider, history), 
            construct_system_prompt(), round
        )
        if isinstance(response, dict) and response.get("error_msg"):
            #  place holder
            history.append(response)
            continue

        history.append(response)

        if tool_calls := getattr(response, "tool_calls", None):
            round += 1
            tool_outputs= tool_loop(tool_calls, provider, round)

            history.extend(tool_outputs)

            while round <= max_round:
                round += 1
                response = call_model(
                    provider, model, 
                    adapt_provider_format(provider, history), 
                    construct_system_prompt(), round
                )
                history.append(response)
                if tool_calls := getattr(response, "tool_calls", None):
                    round += 1
                    tool_outputs= tool_loop(tool_calls, provider, round)
                    history.extend(tool_outputs)




        round += 1



def start_conversation_OpenAI():
    conversation_loop()

if __name__ == "__main__":
    start_conversation_OpenAI()