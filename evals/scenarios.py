"""Declarative cases keep oracle data out of the tested conversation."""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CASES_PATH = Path(__file__).with_name("cases") / "test_1.json"
TOOLS = {"list_jobs", "lookup_job", "search_jobs_hybrid", "create_job", "search_evidence_hybrid",
         "verify_candidate_claim", "build_candidate_profile", "load_skill", "read_skill_resource"}


class Expectation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tool: str
    arguments: dict = Field(default_factory=dict)
    success: bool = True
    equals: dict = Field(default_factory=dict)
    contains: dict = Field(default_factory=dict)
    min_count: int = Field(default=1, ge=1)
    max_count: int = Field(default=1, ge=1)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    suite: Literal["smoke", "full"]
    description: str = Field(min_length=1)
    prompts: list[str] = Field(min_length=1, max_length=8)
    expected: list[Expectation] = Field(min_length=1)
    ground_truth: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    job_refs: list[str] = Field(default_factory=list)
    candidate_id: str | None = None
    approvals: dict[str, list[bool]] = Field(default_factory=dict)
    new_jobs: list[dict] = Field(default_factory=list)
    required_apis: list[str] = Field(default_factory=lambda: ["main_agent"])
    fault: Literal["stale_job_index"] | None = None
    max_step: int = Field(default=8, ge=1, le=20)
    max_tool_calls: int = Field(default=20, ge=1, le=100)


def load_cases(suite, fixtures):
    cases = [Scenario.model_validate(row) for row in json.loads(CASES_PATH.read_text(encoding="utf-8"))]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    evidence = {row["evidence_id"] for row in fixtures["evidence"]}
    jobs = {row["id"] for row in fixtures["jobs"]}
    candidates = {row["candidate_id"] for row in fixtures["evidence"]}
    for case in cases:
        if not set(case.evidence_refs) <= evidence or not set(case.job_refs) <= jobs:
            raise ValueError(f"Unknown fixture reference in {case.id}")
        if case.candidate_id is not None and case.candidate_id not in candidates:
            raise ValueError("Unknown candidate")
        if not set(case.approvals) <= {"initial", "similar"}:
            raise ValueError("Unknown approval boundary")
        for expected in case.expected:
            if expected.tool not in TOOLS or expected.max_count < expected.min_count:
                raise ValueError("Invalid tool expectation")
            for path in (*expected.equals, *expected.contains):
                if not path or any(not part for part in path.split(".")):
                    raise ValueError("Invalid assertion path")
        for job in case.new_jobs:
            if set(job) != {"title", "company", "description", "location"}:
                raise ValueError("Expected job must include exact fields")
        if case.new_jobs and not all(case.approvals.get("initial", [])):
            raise ValueError("Write expectation conflicts with rejected approval")
    selected = [case for case in cases if suite == "full" or case.suite == "smoke"]
    if {expected.tool for case in cases if case.suite == "smoke" for expected in case.expected} != TOOLS:
        raise ValueError("Smoke must cover all executable tools")
    return selected
