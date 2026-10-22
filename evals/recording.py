"""Durable, redacted events and reproducibility metadata."""
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

from evals.fixtures import ROOT


def utc():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return plain(vars(value))
    return value


def redact(value):
    value = plain(value)
    if isinstance(value, dict):
        return {key: "[REDACTED]" if re.search(r"api.?key|authorization|password|secret|access.token", key, re.I)
                else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        for key, secret in os.environ.items():
            if re.search(r"KEY|TOKEN|PASSWORD|SECRET", key) and len(secret) >= 8:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value


def git_metadata():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    fingerprint = hashlib.sha256(git("diff", "HEAD", "--binary") + status)
    for name in git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0"):
        if not name:
            continue
        path = ROOT / name
        if path.is_file() and not path.is_symlink() and not name.startswith("evals/runs/"):
            fingerprint.update(name.encode())
            fingerprint.update(path.read_bytes())
    return {"git_branch": git("branch", "--show-current").decode().strip(),
            "git_sha": git("rev-parse", "HEAD").decode().strip(),
            "dirty": bool(status), "dirty_fingerprint": fingerprint.hexdigest() if status else None}


class RunLog:
    def __init__(self, metadata, directory=None):
        self.run_id = uuid.uuid4().hex
        git = git_metadata()
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git["git_sha"][:10] + "_" + self.run_id[:8]
        self.directory = Path(directory) if directory else ROOT / "evals/runs" / name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.scenario = None
        self.events = []
        self.manifest = {"run_id": self.run_id, "started_at": utc(), "finished_at": None, **git, **metadata}
        self.write("manifest.json", self.manifest)
        for filename in ("assertions.json", "grades.json", "summary.json"):
            self.write(filename, {})
        self._stream = (self.directory / "trace.jsonl").open("a", encoding="utf-8")

    def write(self, name, value):
        temporary = self.directory / (name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(redact(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.directory / name)

    def event(self, kind, **data):
        event = redact({"at": utc(), "scenario": self.scenario, "kind": kind, **data})
        self._stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self.events.append(event)
        return event

    def finish(self, summary):
        self.manifest["finished_at"] = utc()
        self.write("manifest.json", self.manifest)
        self.write("summary.json", summary)
        self.event("run_finished", summary=summary)
        self._stream.close()


class BudgetExceeded(BaseException):
    pass


class RequestBudget:
    def __init__(self, maximum):
        if maximum < 1:
            raise ValueError("API request limit must be positive")
        self.maximum = maximum
        self.used = 0

    def take(self):
        if self.used >= self.maximum:
            raise BudgetExceeded(f"Evaluation API request budget exhausted ({self.used}/{self.maximum})")
        self.used += 1
        return self.used
