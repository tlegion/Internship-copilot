from models import ModelResponse, UserInput, ToolOutput


def display_item(item):
    if isinstance(item, UserInput):
        print(f"[User] {item.text}")

    elif isinstance(item, ModelResponse):
        if item.text:
            print(f"\n[Assistant] {item.text}")

        for call in item.tool_calls:
            print(
                f"\n[Tool Call] {call.tool_name}"
                f"\n  call_id: {call.call_id}"
                f"\n  args: {call.arguments}"
            )

    elif isinstance(item, ToolOutput):
        print(
            f"\n[Tool Output] {item.tool_name}"
            f"\n  call_id: {item.call_id}"
            f"\n  output: {item.output}"
        )

    elif isinstance(item, dict) and item.get("error_msg"):
        print(f"\n[Runtime Error] {item['error_msg']}")


def get_user_input(round: int) -> UserInput:
    user_input = input("\nEnter your prompt: ")
    return UserInput(user_input, None, round=round)



