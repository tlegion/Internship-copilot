from models import ModelProvider, ModelResponse, ModelToolCall, UserInput, ToolOutput

def display_history(history: list):
    for item in history:
        if isinstance(item, UserInput):
            print(f"You: {item.text}")

        elif isinstance(item, ModelResponse) and item.text:
            print(f"Assistant: {item.text}")

def get_user_input(round:int) -> UserInput:
    user_input = input()
    return UserInput(user_input, None, round=round)

