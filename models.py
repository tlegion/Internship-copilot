from enum import Enum
from typing import Any

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthroopic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"

class UserInput:
    role = "user"
    text: str
    file: Any
    round:int

    def __init__(self, text, file, round:int):
        self.text = text
        self.file = file
        self.round = round

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
    round:int

    def __init__(
        self,
        text: str | None,
        tool_calls: list[ModelToolCall],
        status: str,
        raw: Any,
        round:int,
    ):
        self.text = text
        self.tool_calls = tool_calls
        self.status = status
        self.raw = raw
        self.round = round


class ToolOutput:
    call_id: str
    outputs: Any
    tool_name: str
    round:int

    def __init__(
        self,
        tool_name: str,
        call_id: str,
        output: Any,
        round:int,
    ):
        self.tool_name = tool_name
        self.call_id = call_id
        self.output = output
        self.round = round

