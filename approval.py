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
    print_approval_details(tool_name, tool_input, call_id)

    while True:
        answer = input(
            "Approve this tool call? [Y/N]: "
        ).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"", "n", "no"}:
            return False

        print("Please enter y or n.")


def review_similar_job_creation(
    tool_name: str,
    tool_input: dict,
    similars: list[dict],
) -> bool:
    print_approval_details(tool_name, tool_input)

    for job in similars:
        print(f"\nExisting Job ID: {job['id']}")
        for field in ("title", "company", "description", "location"):
            marker = "SAME" if tool_input[field] == job[field] else "DIFF"
            print(f"  [{marker}] {field}: {job[field]}")

    while True:
        try:
            answer = input(
                "Similar jobs exist. Do you still want to insert this job? [Y/N] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("Please enter y or n.")


