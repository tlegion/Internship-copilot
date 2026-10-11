"""SQLite storage for the V003 job index; Evidence tables are untouched."""
import json
import math
import sqlite3

from errors import StorageError, ToolError
from storages.database import db_connection_ro, db_connection_rw

SYNC_COMMAND = "python scripts/sync_job_index.py"


def _read_jobs(connection):
    return [dict(row) for row in connection.execute(
        "SELECT id AS job_id, title, company, description, location FROM jobs ORDER BY id"
    )]


def get_cached_job_hashes(hashes: list[str], model: str) -> set[str]:
    connection = db_connection_ro()
    try:
        cached = set()
        # Stay below SQLite's bound-variable limit for large indexes.
        for start in range(0, len(hashes), 500):
            batch = hashes[start:start + 500]
            placeholders = ",".join("?" for _ in batch)
            cached.update(row[0] for row in connection.execute(
                f"SELECT content_hash FROM chunk_embeddings_jobs WHERE embedding_model = ? "
                f"AND content_hash IN ({placeholders})", [model, *batch]
            ))
        return cached
    except sqlite3.Error as e:
        raise StorageError(f"Unable to read Job embedding cache: {e}. Run python scripts/migrate_db.py.") from e
    finally:
        connection.close()


def save_job_index(jobs: list[dict], chunks: list[dict], embeddings: list[dict]) -> None:
    connection = db_connection_rw()
    try:
        connection.execute("BEGIN IMMEDIATE")
        if _read_jobs(connection) != sorted(jobs, key=lambda job: job["job_id"]):
            raise ToolError(f"Jobs changed during indexing. Run {SYNC_COMMAND} again.")
        connection.execute("DELETE FROM chunks_jobs")
        connection.execute("DELETE FROM chunk_fts_jobs")
        connection.executemany(
            "INSERT INTO chunks_jobs (chunk_id, job_id, chunk_index, text, content_hash) VALUES (?, ?, ?, ?, ?)",
            [(c["chunk_id"], c["job_id"], c["chunk_index"], c["text"], c["content_hash"]) for c in chunks],
        )
        connection.executemany(
            "INSERT INTO chunk_fts_jobs (chunk_id, text) VALUES (?, ?)",
            [(c["chunk_id"], c["text"]) for c in chunks],
        )
        connection.executemany(
            "INSERT INTO chunk_embeddings_jobs (content_hash, embedding_model, dimensions, embedding) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(content_hash, embedding_model) DO NOTHING",
            [(e["content_hash"], e["embedding_model"], e["dimensions"], json.dumps(e["embedding"])) for e in embeddings],
        )
        connection.commit()
    except sqlite3.Error as e:
        connection.rollback()
        raise StorageError(f"Unable to save Job index: {e}") from e
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def load_job_search_data(model: str, build_chunks, fts_query: str,
                         company: str | None, location: str | None) -> tuple[list, list]:
    """Check freshness and retrieve both branches in one SQLite read snapshot."""
    connection = db_connection_ro()
    try:
        connection.execute("BEGIN")
        expected = build_chunks(_read_jobs(connection))
        fields = ("chunk_id", "job_id", "chunk_index", "text", "content_hash")
        expected_rows = sorted(tuple(c[field] for field in fields) for c in expected)
        actual = connection.execute(
            "SELECT chunk_id, job_id, chunk_index, text, content_hash FROM chunks_jobs ORDER BY chunk_id"
        ).fetchall()
        fts = connection.execute("SELECT chunk_id, text FROM chunk_fts_jobs ORDER BY chunk_id").fetchall()
        missing = connection.execute(
            "SELECT 1 FROM chunks_jobs c LEFT JOIN chunk_embeddings_jobs e "
            "ON e.content_hash = c.content_hash AND e.embedding_model = ? "
            "WHERE e.content_hash IS NULL LIMIT 1", (model,)
        ).fetchone()
        if (expected_rows != [tuple(row) for row in actual]
                or sorted((c["chunk_id"], c["text"]) for c in expected) != [tuple(row) for row in fts]
                or missing):
            raise ToolError(f"Job index is stale or incomplete. Run {SYNC_COMMAND} before searching.")

        filters, parameters = "", []
        for field, value in (("company", company), ("location", location)):
            if value is not None:
                filters += f" AND j.{field} = ?"
                parameters.append(value)
        columns = "c.chunk_id, c.job_id, c.text, j.title, j.company, j.location, j.description"
        dense = [dict(row) for row in connection.execute(
            f"SELECT {columns}, e.embedding, e.dimensions FROM chunks_jobs c "
            "JOIN jobs j ON j.id = c.job_id JOIN chunk_embeddings_jobs e "
            "ON e.content_hash = c.content_hash AND e.embedding_model = ? "
            f"WHERE 1=1 {filters} ORDER BY c.chunk_id", [model, *parameters]
        )]
        for row in dense:
            row["embedding"] = json.loads(row["embedding"])
            vector = row["embedding"]
            if (not isinstance(vector, list) or not vector or len(vector) != row["dimensions"]
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
                    or not any(vector)):
                raise StorageError(f"Invalid Job embedding cache. Run {SYNC_COMMAND}.")
        bm25 = []
        if fts_query:
            bm25 = [dict(row) for row in connection.execute(
                f"SELECT {columns}, bm25(chunk_fts_jobs) AS score FROM chunk_fts_jobs "
                "JOIN chunks_jobs c ON c.chunk_id = chunk_fts_jobs.chunk_id "
                "JOIN jobs j ON j.id = c.job_id "
                f"WHERE chunk_fts_jobs MATCH ? {filters} ORDER BY score ASC, c.chunk_id",
                [fts_query, *parameters],
            )]
        return dense, bm25
    except (sqlite3.Error, json.JSONDecodeError) as e:
        raise StorageError(
            f"Unable to search Job index: {e}. Run python scripts/migrate_db.py, then {SYNC_COMMAND}."
        ) from e
    finally:
        connection.close()
