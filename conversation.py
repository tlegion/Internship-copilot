from loop import UserInput, conversation_loop, ModelProvider

def display_history(history: list):
    for item in history:
        if isinstance(item, UserInput):
            print(f"You: {item.text}")

        elif isinstance(item, ModelResponse) and item.text:
            print(f"Assistant: {item.text}")

def get_user_input() -> UserInput:
    user_input = input()
    return UserInput(user_input, None)

def start_conversation_OpenAI():
    conversation_loop()

if __name__ == "__main__":
    start_conversation_OpenAI()