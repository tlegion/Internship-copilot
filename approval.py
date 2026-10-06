def request_tool_approval(
    tool_name: str,
    tool_input: dict,
    call_id: str,
) -> bool:
    print("\n========== APPROVAL REQUIRED ==========")
    print(f"Tool: {tool_name}")
    print(f"Call ID: {call_id}")
    print("Arguments:")

    for key, value in tool_input.items():
        print(f"  {key}: {value}")

    print("=======================================")

    while True:
        answer = input(
            "Approve this tool call? [y/N]: "
        ).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"", "n", "no"}:
            return False

        print("Please enter y or n.")