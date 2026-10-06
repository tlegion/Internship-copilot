class CLI:
    def __init__(self):
        self.rendered_history_count = 0

    def get_user_input(self, round: int):
        from models import UserInput

        text = input("You > ")
        return UserInput(
            text=text,
            file=None,
            round=round,
        )

    def request_tool_approval(
        self,
        tool_name: str,
        tool_input: dict,
        call_id: str,
    ) -> bool:
        print("\n--- Approval Required ---")
        print(f"Tool: {tool_name}")
        print(f"Call ID: {call_id}")

        for key, value in tool_input.items():
            print(f"{key}: {value}")

        while True:
            answer = input(
                "Approve? [y/N]: "
            ).strip().lower()

            if answer in {"y", "yes"}:
                return True

            if answer in {"", "n", "no"}:
                return False

            print("Please enter y or n.")