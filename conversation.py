from models import ModelProvider, ModelResponse, ModelToolCall, UserInput, ToolOutput

def display_history(history: list):
    current_round = None
    model_step = 0

    print("\n========== HISTORY ==========")

    for item in history:
        item_round = getattr(item, "round", None)

        if item_round != current_round:
            current_round = item_round
            model_step = 0
            print(f"\n--- Round {current_round} ---")

        if isinstance(item, UserInput):
            print(f"[User] {item.text}")

        elif isinstance(item, ModelResponse):
            model_step += 1
            print(f"[Model Step {model_step}]")

            if item.text:
                print(f"  Text: {item.text}")

            for call in item.tool_calls:
                print(
                    f"  Tool Call: {call.tool_name}"
                    f" | call_id={call.call_id}"
                    f" | args={call.arguments}"
                )

        elif isinstance(item, ToolOutput):
            print(
                f"[Tool Output] {item.tool_name}"
                f" | call_id={item.call_id}"
                f" | output={item.output}"
            )

    print("=============================\n")

def get_user_input(round:int) -> UserInput:
    user_input = input("Enter your prompt:")
    return UserInput(user_input, None, round=round)

