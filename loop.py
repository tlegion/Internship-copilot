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



def tool_loop(tool_calls:list[ModelToolCall], provider:ModelProvider, round:int, step:int, max_tool_call:int = 10):
    tool_outputs:list[ToolOutput] = []
    call_count = 0
    for call in tool_calls:
        result= run_tool(call.tool_name, load_tool_arguments(call.arguments, provider))
        tool_outputs.append(ToolOutput(
            tool_name=call.tool_name,
            call_id=call.call_id,
            output=result,
            round= round,
            step=step
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

Your scope is limited to:
- internship and job information
- job search and comparison
- resume and interview preparation
- career-related reasoning
- tasks supported by the tools available to you

Rules:
- Do not answer requests unrelated to internships, jobs, careers, or your available tools.
- For out-of-scope requests, briefly explain that the request is outside your scope and redirect the user to an internship-related task.
- Use tools whenever authoritative information must come from a tool.
- Never fabricate tool results.
- Treat tool outputs as the source of truth for tool-provided data.
- After receiving tool output, continue until you can give a final answer.
- If a tool fails, report the failure clearly.
- Keep answers concise and relevant.
""".strip()


def conversation_loop(provider:ModelProvider= ModelProvider.OPENAI, model:str= "gpt-6-luna", max_round:int = 10, max_step:int = 3):
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
            step = 1
            tool_outputs= tool_loop(tool_calls, provider, round, step)

            history.extend(tool_outputs)
            response = call_model(
                provider, model, 
                adapt_provider_format(provider, history), 
                construct_system_prompt(), round
            )

            while step <= max_step:

                history.append(response)
                if tool_calls := getattr(response, "tool_calls", None):
                    step += 1
                    tool_outputs= tool_loop(tool_calls, provider, round, step)
                    history.extend(tool_outputs)
                else:
                    break

                response = call_model(
                    provider, model, 
                    adapt_provider_format(provider, history), 
                    construct_system_prompt(), round
                )




        round += 1



def start_conversation_OpenAI():
    conversation_loop()

if __name__ == "__main__":
    start_conversation_OpenAI()