def print_approval_details(tool_name:str, tool_input:dict, call_id:str | None = None):
    print("\n========== APPROVAL REQUIRED ==========")
    print(f"Tool: {tool_name}")
    if call_id:
        print(f"Call ID: {call_id}")
    print("Arguments:")

    for key, value in tool_input.items():
        print(f"  {key}: {value}")

    print("=======================================")


def request_tool_approval(
    tool_name: str,
    tool_input: dict,
    call_id: str | None = None,
) -> bool:
    print_approval_details(tool_name, call_id, tool_input)

    while True:
        answer = input(
            "Approve this tool call? [y/N]: "
        ).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"", "n", "no"}:
            return False

        print("Please enter y or n.")


