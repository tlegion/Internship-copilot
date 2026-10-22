"""Scoped observation of actual HTTP attempts, business validation and approvals.

The runner is synchronous and process-local. This must not be used alongside an
interactive conversation or parallel evaluation in the same Python process.
"""
from contextlib import contextmanager
import json
import time
from unittest.mock import patch

import httpx2 as httpx

from evals.isolation import SafetyError


def api_kind(body, url):
    if url.endswith("/embeddings"):
        return "embedding"
    name = body.get("text", {}).get("format", {}).get("name", "")
    return {"ClaimBatch": "candidate_extraction", "ClaimAssessment": "candidate_checker",
            "RerankOutput": "reranking", "JobRerankOutput": "reranking", "EvalGrade": "grader"}.get(name, "main_agent")


@contextmanager
def observe_apis(log, budget):
    send = httpx.Client.send

    def observed_send(client, request, *args, **kwargs):
        # Only the OpenAI SDK calls run within this scope. Count every transport
        # attempt, including SDK retries. Headers and URLs with query strings are never logged.
        number = budget.take()
        try:
            body = json.loads(request.content)
        except (ValueError, UnicodeDecodeError):
            body = {"unparsed_body": True}
        kind = api_kind(body, request.url.path)
        log.event("api_request", request_number=number, api=kind,
                  endpoint=request.url.path, body=body)
        started = time.perf_counter()
        try:
            response = send(client, request, *args, **kwargs)
            response.read()
            try:
                result = response.json()
            except ValueError:
                result = {"unparsed_response": True}
            log.event("api_response", request_number=number, api=kind,
                      status_code=response.status_code, latency_seconds=time.perf_counter() - started,
                      body=result, usage=result.get("usage") if isinstance(result, dict) else None)
            return response
        except BaseException as error:
            log.event("api_error", request_number=number, api=kind, error_type=type(error).__name__,
                      error=str(error), latency_seconds=time.perf_counter() - started)
            raise

    with patch.object(httpx.Client, "send", observed_send):
        yield


class Approvals:
    def __init__(self, decisions, log):
        self.pending = {key: list(decisions.get(key, [])) for key in ("initial", "similar")}
        self.log = log
        self.events = []

    def decide(self, boundary, tool_name, arguments, **details):
        queue = self.pending[boundary]
        if not queue:
            self.log.event("approval_unexpected", boundary=boundary, tool=tool_name, arguments=arguments)
            raise SafetyError(f"Unscripted {boundary} approval request")
        decision = queue.pop(0)
        event = self.log.event("approval", boundary=boundary, tool=tool_name,
                               arguments=arguments, decision=decision, **details)
        self.events.append(event)
        return decision

    @contextmanager
    def install(self):
        import approval
        import tool_call
        def initial(tool_name, tool_input, call_id=None):
            return self.decide("initial", tool_name, tool_input, call_id=call_id)
        def similar(tool_name, tool_input, similars):
            return self.decide("similar", tool_name, tool_input, similars=similars)
        with patch.object(tool_call, "request_tool_approval", initial), \
                patch.object(approval, "request_tool_approval", initial), \
                patch.object(approval, "review_similar_job_creation", similar), \
                patch("builtins.input", side_effect=SafetyError("Interactive input is forbidden in evaluations")):
            yield self


@contextmanager
def observe_checker(log):
    from tools import candidate
    validate = candidate._validate_check_assessment
    def observed(assessment, context):
        try:
            return validate(assessment, context)
        except Exception as error:
            log.event("checker_validation_failure", assessment=assessment, error=str(error))
            raise
    with patch.object(candidate, "_validate_check_assessment", observed):
        yield
