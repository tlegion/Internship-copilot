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
        redacted = {}
        for key, item in value.items():
            sensitive = re.search(
                r"api.?key|authorization|password|secret|access.token", key, re.I,
            )
            redacted[key] = "[REDACTED]" if sensitive else redact(item)
        return redacted
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
    return {
        "git_branch": git("branch", "--show-current").decode().strip(),
        "git_sha": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(status),
        "dirty_fingerprint": fingerprint.hexdigest() if status else None,
    }


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    )


def _content_hash(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _diagnostic_value(field, value):
    encoded = _canonical_json(value).encode("utf-8")
    is_blob = isinstance(value, dict) and "sqlite_blob_hex" in value
    if field == "embedding" or is_blob or len(encoded) > 500:
        summary = {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}
        if isinstance(value, str) and field != "embedding":
            summary["preview"] = value[:200]
        return summary
    return value


def _record_changes(before, after):
    # Public tables and FTS shadow tables use these single or composite IDs.
    candidates = (
        ("chunk_id",), ("content_hash", "embedding_model"), ("id",),
        ("evidence_id",), ("segid", "term"), ("k",),
    )
    for fields in candidates:
        if not all(all(field in row for field in fields) for row in before + after):
            continue
        before_by_id = {_canonical_json([row[field] for field in fields]): row for row in before}
        after_by_id = {_canonical_json([row[field] for field in fields]): row for row in after}
        if len(before_by_id) == len(before) and len(after_by_id) == len(after):
            break
    else:
        # A hash still covers every row when the table has no reliable unique ID.
        return None

    changes = {"id_fields": list(fields), "added": [], "removed": [], "modified": []}
    for key in sorted(before_by_id.keys() | after_by_id.keys()):
        old = before_by_id.get(key)
        new = after_by_id.get(key)
        if old == new:
            continue
        row = new if new is not None else old
        record_id = row[fields[0]] if len(fields) == 1 else {field: row[field] for field in fields}
        field_changes = {}
        for field in sorted((old or {}).keys() | (new or {}).keys()):
            if (old is not None and new is not None and field in old and field in new
                    and old[field] == new[field]):
                continue
            values = {}
            if old is not None and field in old:
                values["before"] = _diagnostic_value(field, old[field])
            if new is not None and field in new:
                values["after"] = _diagnostic_value(field, new[field])
            field_changes[field] = values
        if old is None:
            kind = "added"
        elif new is None:
            kind = "removed"
        else:
            kind = "modified"
        changes[kind].append({"id": record_id, "fields": field_changes})
    return changes


def summarize_database_changes(before, after):
    """Describe full in-memory states without persisting repeated rows or vectors.

    Hashes cover complete table contents, including duplicate rows, independently
    of row order and dictionary insertion order. Assertions still use the originals.
    """
    tables = {}
    for table in sorted(before.keys() | after.keys()):
        old_rows = before.get(table, [])
        new_rows = after.get(table, [])
        old_hash = _content_hash(sorted(old_rows, key=_canonical_json))
        new_hash = _content_hash(sorted(new_rows, key=_canonical_json))
        changed = table not in before or table not in after or old_hash != new_hash
        tables[table] = {
            "before": {"present": table in before, "row_count": len(old_rows), "sha256": old_hash},
            "after": {"present": table in after, "row_count": len(new_rows), "sha256": new_hash},
            "changed": changed,
        }
        if changed:
            tables[table]["changes"] = _record_changes(old_rows, new_rows)
    return tables


class RunLog:
    def __init__(self, metadata, directory=None):
        self.run_id = uuid.uuid4().hex
        git = git_metadata()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{timestamp}_{git['git_sha'][:10]}_{self.run_id[:8]}"
        self.directory = Path(directory) if directory else ROOT / "evals/runs" / name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.scenario = None
        self.events = []
        self.manifest = {
            "run_id": self.run_id, "started_at": utc(), "finished_at": None, **git, **metadata,
        }
        self.write("manifest.json", self.manifest)
        self.write("summary.json", {})
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
            raise BudgetExceeded(
                f"Evaluation API request budget exhausted ({self.used}/{self.maximum})"
            )
        self.used += 1
        return self.used
