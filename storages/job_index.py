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
    if not hashes:
        return set()
    connection = db_connection_ro()
    try:
        cached_hashes = set()
        # Each batch stays below SQLite's bound-parameter limit.
        for start in range(0, len(hashes), 500):
            batch = hashes[start:start + 500]
            placeholders = ", ".join("?" for _ in batch)
            rows = connection.execute(
                f"""
                SELECT content_hash FROM chunk_embeddings_jobs
                WHERE embedding_model = ? AND content_hash IN ({placeholders})
                """,
                (model, *batch),
            )
            cached_hashes.update(row["content_hash"] for row in rows)
        return cached_hashes
    except sqlite3.Error as e:
        raise StorageError(f"Unable to read Job embedding cache: {e}. Run python scripts/migrate_db.py.") from e
    finally:
        connection.close()


def save_job_index(jobs: list[dict], chunks: list[dict], embeddings: list[dict]) -> None:
    chunk_rows = [
        (chunk["chunk_id"], chunk["job_id"], chunk["chunk_index"], chunk["text"], chunk["content_hash"])
        for chunk in chunks
    ]
    embedding_rows = [
        (item["content_hash"], item["embedding_model"], item["dimensions"], json.dumps(item["embedding"]))
        for item in embeddings
    ]
    fts_rows = [(chunk["chunk_id"], chunk["text"]) for chunk in chunks]
    connection = db_connection_rw()
    try:
        connection.execute("BEGIN IMMEDIATE")
        if _read_jobs(connection) != sorted(jobs, key=lambda job: job["job_id"]):
            raise ToolError(f"Jobs changed during indexing. Run {SYNC_COMMAND} again.")
        connection.execute("DELETE FROM chunks_jobs")
        connection.execute("DELETE FROM chunk_fts_jobs")
        connection.executemany(
            """
            INSERT INTO chunks_jobs (chunk_id, job_id, chunk_index, text, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )
        connection.executemany(
            "INSERT INTO chunk_fts_jobs (chunk_id, text) VALUES (?, ?)",
            fts_rows,
        )
        connection.executemany(
            """
            INSERT INTO chunk_embeddings_jobs (content_hash, embedding_model, dimensions, embedding)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(content_hash, embedding_model) DO NOTHING
            """,
            embedding_rows,
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


def check_job_index(connection, embedding_model: str) -> None:
    from rag_lab.job_chunking import chunk_jobs

    expected_chunks = sorted(chunk_jobs(_read_jobs(connection)), key=lambda chunk: chunk["chunk_id"])
    actual_chunks = [dict(row) for row in connection.execute("""
        SELECT chunk_id, job_id, chunk_index, text, content_hash
        FROM chunks_jobs ORDER BY chunk_id
    """)]
    expected_fts = [(chunk["chunk_id"], chunk["text"]) for chunk in expected_chunks]
    actual_fts = [tuple(row) for row in connection.execute(
        "SELECT chunk_id, text FROM chunk_fts_jobs ORDER BY chunk_id"
    )]
    missing_embedding = connection.execute("""
        SELECT 1 FROM chunks_jobs c
        LEFT JOIN chunk_embeddings_jobs e
          ON e.content_hash = c.content_hash AND e.embedding_model = ?
        WHERE e.content_hash IS NULL LIMIT 1
    """, (embedding_model,)).fetchone()
    if expected_chunks != actual_chunks or expected_fts != actual_fts or missing_embedding:
        raise ToolError(f"Job index is stale or incomplete. Run {SYNC_COMMAND} before searching.")


def load_indexed_job_chunks(connection, embedding_model: str,
                            company: str | None, location: str | None) -> list[dict]:
    query = """
        SELECT c.chunk_id, c.job_id, c.text,
               j.title, j.company, j.location, j.description,
               e.embedding, e.dimensions
        FROM chunks_jobs c
        JOIN jobs j ON j.id = c.job_id
        JOIN chunk_embeddings_jobs e
          ON e.content_hash = c.content_hash AND e.embedding_model = ?
        WHERE 1=1
    """
    parameters = [embedding_model]
    if company is not None:
        query += " AND j.company = ?"
        parameters.append(company)
    if location is not None:
        query += " AND j.location = ?"
        parameters.append(location)
    query += " ORDER BY c.chunk_id"

    chunks = []
    for row in connection.execute(query, parameters):
        chunk = dict(row)
        vector = json.loads(chunk["embedding"])
        if (not isinstance(vector, list) or not vector or not any(vector)
                or len(vector) != chunk["dimensions"]
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
            raise StorageError(f"Invalid Job embedding cache. Run {SYNC_COMMAND}.")
        chunk["embedding"] = vector
        chunks.append(chunk)
    return chunks


def bm25_job_search(connection, fts_query: str,
                    company: str | None, location: str | None) -> list[dict]:
    if not fts_query:
        return []
    query = """
        SELECT c.chunk_id, c.job_id, c.text,
               j.title, j.company, j.location, j.description,
               bm25(chunk_fts_jobs) AS score
        FROM chunk_fts_jobs
        JOIN chunks_jobs c ON c.chunk_id = chunk_fts_jobs.chunk_id
        JOIN jobs j ON j.id = c.job_id
        WHERE chunk_fts_jobs MATCH ?
    """
    parameters = [fts_query]
    if company is not None:
        query += " AND j.company = ?"
        parameters.append(company)
    if location is not None:
        query += " AND j.location = ?"
        parameters.append(location)
    query += " ORDER BY score ASC, c.chunk_id"
    return [dict(row) for row in connection.execute(query, parameters)]


def load_job_search_data(embedding_model: str, fts_query: str,
                         company: str | None, location: str | None) -> tuple[list, list]:
    """Verify freshness and load both branches in one read snapshot."""
    connection = db_connection_ro()
    try:
        connection.execute("BEGIN")
        check_job_index(connection, embedding_model)
        chunks = load_indexed_job_chunks(connection, embedding_model, company, location)
        bm25_results = bm25_job_search(connection, fts_query, company, location)
        return chunks, bm25_results
    except (sqlite3.Error, json.JSONDecodeError) as e:
        raise StorageError(
            f"Unable to search Job index: {e}. Run python scripts/migrate_db.py, then {SYNC_COMMAND}."
        ) from e
    finally:
        connection.close()
