"""Hard checks independent of the semantic grader."""
import json


def select(value, path):
    parts = path.split(".") if path else []
    if not parts:
        return value
    head, *tail = parts
    if head == "*":
        return [select(item, ".".join(tail)) for item in value]
    child = value[int(head)] if isinstance(value, list) else value[head]
    return select(child, ".".join(tail))


def _check_expected_tools(case, executions, calls, check):
    for expected in case.expected:
        matching = []
        for event in executions:
            if event["tool"] != expected.tool:
                continue
            arguments = calls.get(event["call_id"], {}).get("arguments", {})
            matches_arguments = all(
                arguments.get(key) == value for key, value in expected.arguments.items()
            )
            if matches_arguments:
                matching.append(event)

        check(
            f"{expected.tool}:execution_count",
            expected.min_count <= len(matching) <= expected.max_count,
            {"actual": len(matching), "min": expected.min_count, "max": expected.max_count},
        )
        for index, event in enumerate(matching):
            output = event["output"]
            prefix = f"{expected.tool}:{index}"
            check(f"{prefix}:success", output.get("success") == expected.success)
            operations = (("equals", expected.equals), ("contains", expected.contains))
            for operation, conditions in operations:
                for path, value in conditions.items():
                    try:
                        actual = select(output, path)
                        passed = actual == value if operation == "equals" else value in actual
                    except (KeyError, IndexError, ValueError, TypeError):
                        actual, passed = None, False
                    check(
                        f"{prefix}:{operation}:{path}", passed,
                        {"expected": value, "actual": actual},
                    )


def _check_candidate_assessments(result, candidate_id, documents, call_id, check):
    from tools.candidate import ClaimAssessment

    for assessment in result.get("assessments", [result]):
        citations = assessment.get("selected_evidence", [])
        try:
            ClaimAssessment.model_validate({
                "is_atomic": True,
                "verdict": assessment["verdict"],
                "reason": assessment["reason"],
                "selected_evidence": [
                    {key: item[key] for key in ("evidence_id", "role", "quote")}
                    for item in assessment["selected_evidence"]
                ],
            })
            valid = (
                assessment["candidate_id"] == candidate_id
                and bool(assessment["reason"].strip())
            )
        except Exception:
            valid = False
        check(f"{call_id}:assessment_contract", valid)

        seen_ids, seen_quotes = set(), set()
        for citation in citations:
            document = documents.get(citation.get("evidence_id"))
            valid_source = False
            if document and document["candidate_id"] == candidate_id:
                exact_quote = (
                    bool(citation.get("quote", "").strip())
                    and citation["quote"] in document["text"]
                )
                same_metadata = all(
                    citation.get(key) == document[key] for key in ("source_id", "category", "date")
                )
                valid_source = exact_quote and same_metadata
            check(f"{call_id}:citation_source", valid_source)
            unique = (
                citation.get("evidence_id") not in seen_ids
                and citation.get("quote") not in seen_quotes
            )
            check(f"{call_id}:citation_unique", unique)
            seen_ids.add(citation.get("evidence_id"))
            seen_quotes.add(citation.get("quote"))

        required_roles = {"supported": "supports", "contradicted": "contradicts"}
        role = required_roles.get(assessment.get("verdict"))
        has_required_role = not role or any(item.get("role") == role for item in citations)
        check(f"{call_id}:decisive_role", has_required_role)


def _check_tool_results(case, executions, calls, fixtures, check):
    documents = {row["evidence_id"]: row for row in fixtures["evidence"]}
    jobs = {row["id"]: row for row in fixtures["jobs"]}
    activations = []
    for event in executions:
        call_id = event["call_id"]
        tool = event["tool"]
        output = event["output"]
        arguments = calls.get(call_id, {}).get("arguments", {})
        position = (event["round"], event["step"])
        if tool == "load_skill" and output.get("success"):
            activations.append((arguments.get("name"), position))
        if tool in {"verify_candidate_claim", "build_candidate_profile", "read_skill_resource"}:
            name = arguments.get("name", "candidate-recognition")
            # Activation occurs after a whole batch, never within that same batch.
            loaded_before = any(skill == name and batch < position for skill, batch in activations)
            check(f"{call_id}:skill_loaded_before_batch", loaded_before)
        if "candidate_id" in arguments and case.candidate_id:
            check(f"{call_id}:candidate_argument", arguments["candidate_id"] == case.candidate_id)

        if output.get("success"):
            has_output = "result" in output
        else:
            has_output = bool(output.get("error_type"))
        check(f"{call_id}:output_contract", isinstance(output.get("success"), bool) and has_output)
        if not output.get("success"):
            continue
        result = output["result"]
        if tool in {"verify_candidate_claim", "build_candidate_profile"}:
            _check_candidate_assessments(result, case.candidate_id, documents, call_id, check)
        if tool == "search_evidence_hybrid":
            owned = True
            for row in result:
                document = documents.get(row.get("evidence_id"))
                if (document is None or row.get("candidate_id") != case.candidate_id
                        or document["candidate_id"] != case.candidate_id):
                    owned = False
                    break
            check(f"{call_id}:retrieval_ownership", owned)
        if tool in {"list_jobs", "lookup_job", "search_jobs_hybrid"}:
            original_jobs = True
            for row in result.get("jobs", [result]):
                original = jobs.get(row.get("job_id", row.get("id")))
                if original is None or any(
                    row.get(field) != original[field]
                    for field in ("title", "company", "description", "location")
                ):
                    original_jobs = False
                    break
            check(f"{call_id}:original_jobs", original_jobs)


def _check_database_and_approvals(case, before, after, approvals, check):
    # Always compare complete in-memory rows; logging summaries never determine a pass.
    actual_new = [row for row in after["jobs"] if row not in before["jobs"]]
    unchanged = [row for row in after["jobs"] if row in before["jobs"]]
    check("existing_jobs_unchanged", unchanged == before["jobs"])
    actual_fields = [
        {key: row[key] for key in ("title", "company", "description", "location")}
        for row in actual_new
    ]
    sort_key = lambda row: json.dumps(row, sort_keys=True)
    expected_jobs = sorted(case.new_jobs, key=sort_key)
    check("exact_job_changes", sorted(actual_fields, key=sort_key) == expected_jobs)
    before_other = {key: value for key, value in before.items() if key != "jobs"}
    after_other = {key: value for key, value in after.items() if key != "jobs"}
    check("non_job_tables_unchanged", before_other == after_other)
    check("all_scripted_approvals_used", not any(approvals.pending.values()))
    for boundary in ("initial", "similar"):
        decisions = [
            event["decision"] for event in approvals.events if event["boundary"] == boundary
        ]
        check(f"{boundary}:decisions", decisions == case.approvals.get(boundary, []))
    if any(not event["decision"] for event in approvals.events):
        check("rejection_did_not_write", not actual_new)


def _check_completion(case, events, check):
    completed_turns = {
        event.get("round") for event in events
        if event["kind"] == "assistant" and event.get("text") and not event.get("tool_calls")
    }
    all_turns_answered = all(turn in completed_turns for turn in range(1, len(case.prompts) + 1))
    check("completed_assistant_answers", all_turns_answered)
    apis = {event.get("api") for event in events if event["kind"] == "api_request"}
    for api in case.required_apis:
        check(f"api_path:{api}", api in apis)


def assertions(case, events, before, after, fixtures, approvals):
    checks = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    executions = [event for event in events if event["kind"] == "tool_output"]
    calls = {}
    for event in events:
        if event["kind"] == "assistant":
            for call in event.get("tool_calls", []):
                calls[call["call_id"]] = call

    _check_expected_tools(case, executions, calls, check)
    check("bounded_tool_trajectory", len(executions) <= case.max_tool_calls)
    runtime_errors = any(event["kind"] in {"runtime_error", "scenario_error"} for event in events)
    check("no_runtime_errors", not runtime_errors)
    _check_tool_results(case, executions, calls, fixtures, check)
    _check_database_and_approvals(case, before, after, approvals, check)
    _check_completion(case, events, check)
    return checks
