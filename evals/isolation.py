"""Process-local SQLite sandbox; fixed-path migration/seed entrypoints are never run."""
from contextlib import contextmanager, closing
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

from evals.fixtures import ROOT


class SafetyError(RuntimeError):
    pass


class Sandbox:
    def __init__(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="agent-eval-")
        self.root = Path(self._temporary.name).resolve()
        self.snapshot = self.root / "prepared.sqlite"

    def close(self):
        self._temporary.cleanup()

    def checked(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise SafetyError("Eval database must remain inside its private sandbox")
        return path

    @contextmanager
    def bind(self, path):
        from storages import database
        path = self.checked(path)
        if not path.is_file():
            raise SafetyError("Eval database must already exist; no fallback is allowed")
        connect = sqlite3.connect

        def guarded_connect(database_path, *args, **kwargs):
            name = str(database_path)
            if name.startswith("file:"):
                parsed = urlsplit(name)
                if parsed.netloc:
                    raise SafetyError("Remote SQLite URI forbidden")
                name = unquote(parsed.path)
            self.checked(name)
            if Path(database.DB_PATH).resolve() != path:
                raise SafetyError("Eval database override was lost")
            return connect(database_path, *args, **kwargs)

        with patch.object(database, "DB_PATH", path), patch.object(sqlite3, "connect", guarded_connect):
            yield path

    def create(self, fixtures):
        self.checked(self.snapshot)
        # Exclusive creation prevents accidental reuse of a populated database.
        with self.snapshot.open("xb"):
            pass
        from scripts.migrate_db import MIGRATIONS
        with closing(sqlite3.connect(self.snapshot)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            for version, migration in enumerate(MIGRATIONS, 1):
                connection.executescript(migration.read_text(encoding="utf-8"))
                connection.execute(f"PRAGMA user_version = {version}")
            connection.executemany(
                "INSERT INTO evidence (evidence_id,candidate_id,source_id,category,date,text) VALUES (?,?,?,?,?,?)",
                [tuple(row[field] for field in ("evidence_id", "candidate_id", "source_id", "category", "date", "text"))
                 for row in fixtures["evidence"]])
            connection.executemany("INSERT INTO jobs (id,title,company,description,location) VALUES (?,?,?,?,?)",
                                   [tuple(row[field] for field in ("id", "title", "company", "description", "location"))
                                    for row in fixtures["jobs"]])
        self.validate_records(self.snapshot, fixtures)

    def validate_records(self, path, fixtures):
        from scripts.migrate_db import MIGRATIONS
        with self.bind(path), closing(sqlite3.connect(path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise SafetyError("SQLite integrity check failed")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise SafetyError("Foreign-key validation failed")
            if connection.execute("PRAGMA user_version").fetchone()[0] != len(MIGRATIONS):
                raise SafetyError("Migration version mismatch")
            for table, fields, key in (
                ("evidence", "evidence_id,candidate_id,source_id,category,date,text", "evidence_id"),
                ("jobs", "id,title,company,description,location", "id"),
            ):
                rows = [dict(row) for row in connection.execute(f"SELECT {fields} FROM {table} ORDER BY {key}")]
                if rows != sorted(fixtures[table], key=lambda row: row[key]):
                    raise SafetyError(f"{table} differs from fixture records")

    def prepare_indexes(self, fixtures):
        from rag_lab.indexing import sync_rag_index
        from rag_lab.job_indexing import sync_job_index
        with self.bind(self.snapshot):
            sync_rag_index()
            sync_job_index()
        self.validate_records(self.snapshot, fixtures)
        self.validate_indexes(self.snapshot, fixtures)

    def validate_indexes(self, path, fixtures):
        from rag_lab.chunking import EMBEDDING_MODEL, chunk_documents
        from rag_lab.job_chunking import chunk_jobs
        specs = [("chunks", "chunk_fts", "chunk_embeddings", "evidence_id",
                  chunk_documents(sorted(fixtures["evidence"], key=lambda row: row["evidence_id"]))),
                 ("chunks_jobs", "chunk_fts_jobs", "chunk_embeddings_jobs", "job_id",
                  chunk_jobs([{**row, "job_id": row["id"]} for row in sorted(fixtures["jobs"], key=lambda row: row["id"])]))]
        with self.bind(path), closing(sqlite3.connect(path)) as connection, connection:
            for chunks, fts, embeddings, parent, expected in specs:
                expected_rows = sorted((row["chunk_id"], row[parent], row["chunk_index"], row["text"],
                                        hashlib.sha256(row["text"].encode()).hexdigest()) for row in expected)
                actual = connection.execute(
                    f"SELECT chunk_id,{parent},chunk_index,text,content_hash FROM {chunks} ORDER BY chunk_id").fetchall()
                expected_fts = sorted((row["chunk_id"], row["text"]) for row in expected)
                actual_fts = connection.execute(f"SELECT chunk_id,text FROM {fts} ORDER BY chunk_id").fetchall()
                if actual != expected_rows or actual_fts != expected_fts:
                    raise SafetyError(f"Missing, incomplete or outdated {chunks}/FTS index")
                connection.execute(f"INSERT INTO {fts}({fts}) VALUES ('integrity-check')")
                dimensions = set()
                for content_hash in {row[-1] for row in expected_rows}:
                    cached = connection.execute(
                        f"SELECT dimensions,embedding FROM {embeddings} WHERE content_hash=? AND embedding_model=?",
                        (content_hash, EMBEDDING_MODEL)).fetchone()
                    if cached is None:
                        raise SafetyError("Missing embedding")
                    dimension, raw = cached
                    try:
                        vector = json.loads(raw)
                        valid = (isinstance(vector, list) and dimension > 0 and len(vector) == dimension
                                 and any(vector) and all(type(value) in (int, float) and math.isfinite(value) for value in vector))
                    except (ValueError, TypeError):
                        valid = False
                    if not valid:
                        raise SafetyError("Corrupt embedding")
                    dimensions.add(dimension)
                if len(dimensions) != 1:
                    raise SafetyError("Inconsistent embedding dimensions")
                if dimensions != {1536}:
                    raise SafetyError("Unexpected text-embedding-3-small dimensions")

    def reset(self, name, fixtures):
        self.validate_records(self.snapshot, fixtures)
        self.validate_indexes(self.snapshot, fixtures)
        path = self.checked(self.root / f"{name}.sqlite")
        if path.exists():
            raise SafetyError("Scenario database already exists")
        shutil.copyfile(self.snapshot, path)
        self.validate_records(path, fixtures)
        self.validate_indexes(path, fixtures)
        return path

    def state(self, path):
        with self.bind(path), closing(sqlite3.connect(path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            tables = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            return {table: sorted([{key: ({"sqlite_blob_hex": value.hex()} if isinstance(value, bytes) else value)
                                    for key, value in dict(row).items()}
                                   for row in connection.execute(f'SELECT * FROM "{table}"')],
                                  key=lambda row: json.dumps(row, sort_keys=True)) for table in tables}
