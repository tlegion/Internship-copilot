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


def assertions(case, events, before, after, fixtures, approvals):
    checks = []
    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
    executions = [event for event in events if event["kind"] == "tool_output"]
    calls = {call["call_id"]: call for event in events if event["kind"] == "assistant"
             for call in event.get("tool_calls", [])}
    for expected in case.expected:
        matching = [event for event in executions if event["tool"] == expected.tool
                    and all(calls.get(event["call_id"], {}).get("arguments", {}).get(key) == value
                            for key, value in expected.arguments.items())]
        check(f"{expected.tool}:execution_count", expected.min_count <= len(matching) <= expected.max_count,
              {"actual": len(matching), "min": expected.min_count, "max": expected.max_count})
        for index, event in enumerate(matching):
            output = event["output"]
            check(f"{expected.tool}:{index}:success", output.get("success") == expected.success)
            for operation, conditions in (("equals", expected.equals), ("contains", expected.contains)):
                for path, value in conditions.items():
                    try:
                        actual = select(output, path)
                        passed = actual == value if operation == "equals" else value in actual
                    except (KeyError, IndexError, ValueError, TypeError):
                        actual, passed = None, False
                    check(f"{expected.tool}:{index}:{operation}:{path}", passed,
                          {"expected": value, "actual": actual})
    check("bounded_tool_trajectory", len(executions) <= case.max_tool_calls)
    check("no_runtime_errors", not any(event["kind"] in {"runtime_error", "scenario_error"} for event in events))
    # Activation occurs after an entire tool batch. A same-batch gated call is invalid.
    activations = []
    documents = {row["evidence_id"]: row for row in fixtures["evidence"]}
    jobs = {row["id"]: row for row in fixtures["jobs"]}
    for event in executions:
        output = event["output"]
        arguments = calls.get(event["call_id"], {}).get("arguments", {})
        position = (event["round"], event["step"])
        if event["tool"] == "load_skill" and output.get("success"):
            activations.append((arguments.get("name"), position))
        if event["tool"] in {"verify_candidate_claim", "build_candidate_profile", "read_skill_resource"}:
            name = arguments.get("name", "candidate-recognition")
            check(f"{event['call_id']}:skill_loaded_before_batch",
                  any(skill == name and batch < position for skill, batch in activations))
        if "candidate_id" in arguments and case.candidate_id:
            check(f"{event['call_id']}:candidate_argument", arguments["candidate_id"] == case.candidate_id)
        check(f"{event['call_id']}:output_contract", isinstance(output.get("success"), bool)
              and (("result" in output) if output.get("success") else bool(output.get("error_type"))))
        if not output.get("success"):
            continue
        result = output["result"]
        if event["tool"] in {"verify_candidate_claim", "build_candidate_profile"}:
            from tools.candidate import ClaimAssessment
            for assessment in result.get("assessments", [result]):
                try:
                    ClaimAssessment.model_validate({"is_atomic": True, "verdict": assessment["verdict"],
                        "reason": assessment["reason"], "selected_evidence": [
                            {key: item[key] for key in ("evidence_id", "role", "quote")}
                            for item in assessment["selected_evidence"]]})
                    valid = assessment["candidate_id"] == case.candidate_id and bool(assessment["reason"].strip())
                except Exception:
                    valid = False
                check(f"{event['call_id']}:assessment_contract", valid)
                seen_ids, seen_quotes = set(), set()
                for citation in assessment.get("selected_evidence", []):
                    document = documents.get(citation.get("evidence_id"))
                    valid = bool(document and document["candidate_id"] == case.candidate_id
                                 and citation.get("quote", "").strip()
                                 and citation["quote"] in document["text"]
                                 and all(citation.get(key) == document[key] for key in ("source_id", "category", "date")))
                    check(f"{event['call_id']}:citation_source", valid)
                    check(f"{event['call_id']}:citation_unique",
                          citation.get("evidence_id") not in seen_ids and citation.get("quote") not in seen_quotes)
                    seen_ids.add(citation.get("evidence_id")); seen_quotes.add(citation.get("quote"))
                role = {"supported": "supports", "contradicted": "contradicts"}.get(assessment.get("verdict"))
                check(f"{event['call_id']}:decisive_role", not role or
                      any(item.get("role") == role for item in assessment.get("selected_evidence", [])))
        if event["tool"] == "search_evidence_hybrid":
            check(f"{event['call_id']}:retrieval_ownership",
                  all(row.get("evidence_id") in documents and row.get("candidate_id") == case.candidate_id
                      and documents[row["evidence_id"]]["candidate_id"] == case.candidate_id for row in result))
        if event["tool"] in {"list_jobs", "lookup_job", "search_jobs_hybrid"}:
            rows = result.get("jobs", [result])
            check(f"{event['call_id']}:original_jobs", all(
                row.get("job_id", row.get("id")) in jobs and all(row.get(field) == jobs[row.get("job_id", row.get("id"))][field]
                    for field in ("title", "company", "description", "location")) for row in rows))
    actual_new = [row for row in after["jobs"] if row not in before["jobs"]]
    unchanged = [row for row in after["jobs"] if row in before["jobs"]]
    check("existing_jobs_unchanged", unchanged == before["jobs"])
    actual_fields = [{key: row[key] for key in ("title", "company", "description", "location")} for row in actual_new]
    key = lambda row: json.dumps(row, sort_keys=True)
    check("exact_job_changes", sorted(actual_fields, key=key) == sorted(case.new_jobs, key=key))
    check("non_job_tables_unchanged", {k: v for k, v in before.items() if k != "jobs"}
          == {k: v for k, v in after.items() if k != "jobs"})
    check("all_scripted_approvals_used", not any(approvals.pending.values()))
    for boundary in ("initial", "similar"):
        check(f"{boundary}:decisions", [e["decision"] for e in approvals.events if e["boundary"] == boundary]
              == case.approvals.get(boundary, []))
    if any(not event["decision"] for event in approvals.events):
        check("rejection_did_not_write", not actual_new)
    check("completed_assistant_answers", all(any(event["kind"] == "assistant" and event.get("round") == turn
          and event.get("text") and not event.get("tool_calls") for event in events)
          for turn in range(1, len(case.prompts) + 1)))
    apis = {event.get("api") for event in events if event["kind"] == "api_request"}
    for api in case.required_apis:
        check(f"api_path:{api}", api in apis)
    return checks
