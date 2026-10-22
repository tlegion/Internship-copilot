"""Strict fixture contracts; no business module, tokenizer or API imports."""
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name("fixtures")


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    date: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^job_[0-9]+$")
    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    description: str = Field(min_length=1)
    location: str | None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixtures():
    paths = [FIXTURES / "test_1/evidence.json", FIXTURES / "test_1/jobs.json",
             FIXTURES / "isolation/evidence.json"]
    evidence, jobs, foreign = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for records, schema in ((evidence, Evidence), (jobs, Job), (foreign, Evidence)):
        if not isinstance(records, list) or not records:
            raise ValueError("Fixtures must be nonempty arrays")
        for record in records:
            schema.model_validate(record)
    if {row["candidate_id"] for row in evidence} != {"test_1"}:
        raise ValueError("test_1 evidence ownership mismatch")
    if {row["candidate_id"] for row in foreign} != {"synthetic_other"}:
        raise ValueError("Isolation evidence ownership mismatch")
    evidence = evidence + foreign
    for records, key in ((evidence, "evidence_id"), (jobs, "id")):
        if len({row[key] for row in records}) != len(records):
            raise ValueError("Duplicate fixture IDs")
    return {"evidence": evidence, "jobs": jobs,
            "hashes": {str(path.relative_to(ROOT)): digest(path) for path in paths}}
