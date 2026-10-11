import json
import sqlite3

from pathlib import Path
from errors import StorageError


DB_PATH = (
    Path(__file__).resolve().parents[1]
    /"data"
    /"internship.db"
)

def db_connection_ro():
    try:
        connection = sqlite3.connect(
            f"file:{DB_PATH}?mode=ro",
            uri=True,
        )
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )
    except sqlite3.Error as e:
        raise StorageError(f"Unable to open database: {e}") from e
    connection.row_factory = sqlite3.Row
    return connection

def db_connection_rw():
    try:
        connection = sqlite3.connect(
            f"file:{DB_PATH}?mode=rw",
            uri=True,
        )
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )
    except sqlite3.Error as e:
        raise StorageError(f"Unable to open database: {e}") from e
    connection.row_factory = sqlite3.Row
    return connection


def get_job(job_id: str) -> dict | None:
    connection = db_connection_ro()

    try:
        row = connection.execute(
            """
            SELECT id, title, company, description, location
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()

        return dict(row) if row is not None else None
    except sqlite3.Error as e:
        raise StorageError(f"Unable to look up job {job_id}: {e}") from e
    finally:
        connection.close()


def get_all_jobs() -> dict | None:
    connection = db_connection_ro()
    try:
        rows = connection.execute(
            """
            SELECT id, title, company, description, location
            FROM jobs
            """
        ).fetchall()
        return [
            {
            "job_id": row[0],
            "title": row[1],
            "company": row[2],
            "description": row[3],
            "location": row[4],
            }
            for row in rows
        ]
    except sqlite3.Error as e:
        raise StorageError(f"Unable to load jobs: {e}") from e
    finally:
        connection.close()

def get_jobs_page(
    limit: int,
    after_job_id: str | None = None,
    company: str | None = None,
    location: str | None = None,
) -> dict:
    connection = db_connection_ro()
    try:
        conditions = []
        parameters = []
        if after_job_id is not None:
            conditions.append("CAST(SUBSTR(id, 5) AS INTEGER) > CAST(? AS INTEGER)")
            parameters.append(after_job_id[4:])
        if company is not None:
            conditions.append("company = ?")
            parameters.append(company)
        if location is not None:
            conditions.append("location = ?")
            parameters.append(location)

        query = "SELECT id AS job_id, title, company, description, location FROM jobs"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY CAST(SUBSTR(id, 5) AS INTEGER) ASC LIMIT ?"
        parameters.append(limit + 1)
        rows = connection.execute(query, parameters).fetchall()
        has_more = len(rows) > limit
        jobs = [dict(row) for row in rows[:limit]]
        return {
            "jobs": jobs,
            "has_more": has_more,
            "next_cursor": jobs[-1]["job_id"] if has_more else None,
        }
    except sqlite3.Error as e:
        raise StorageError(f"Unable to list jobs: {e}") from e
    finally:
        connection.close()


def insert_job(
        id:str, title:str, company:str, 
        description:str, location:str
):
    connection = db_connection_rw()
    try:
        row = connection.execute(
            """
            INSERT INTO jobs
            (id, title, company, description, location)
            VALUES (?, ?, ?, ?, ?)
            """,
            (id, title, company, description, location)
        )
        connection.commit()
        return {
            "msg": "Job creation successful"
        }
    except sqlite3.Error as e:
        connection.rollback()

        raise StorageError(f"Job creation failed: {e}") from e
    finally:
        connection.close()

def check_exists(
        title:str, company:str, 
        description:str, location:str
):
    connection = db_connection_ro()
    try:
        rows = connection.execute(
            """
            SELECT *
            FROM jobs
            WHERE title = ?
            AND company = ?
            AND description = ?
            AND location IS ?
            """,
            (title, company, description, location)
        ).fetchall()
        return rows if rows else None
    except sqlite3.Error as e:
        raise StorageError(f"Unable to check whether job exists: {e}") from e
    finally:
        connection.close()


def check_similar(
        title:str, company:str, 
        description:str, location:str
):
    connection = db_connection_ro()
    similars = []
    key_enumerate = {"title", "company", "description", "location"}
    search = {
        "title": title,
        "company": company,
        "description": description,
        "location": location
    }
    try:
        for key in  key_enumerate:
            remaining = [k for k in key_enumerate if k != key]
            where_str = " AND ".join(f"{k} IS ?" for k in remaining)
            where_str += f" AND {key} IS NOT ?"
            remaining_search = [search[k] for k in remaining]
            remaining_search.append(search[key])
            rows = connection.execute(
                f"""
                SELECT *
                FROM jobs
                WHERE {where_str}
                """,
                remaining_search
            ).fetchall()
            if rows:
                similars.extend({
                    **dict(row),
                    "similar_reason": f"only {key} different"
                    } for row in rows)

        return similars


    except sqlite3.Error as e:
        raise StorageError(f"Unable to check whether job exists: {e}") from e
    finally:
        connection.close()

def get_last_job() -> dict | None:
    connection = db_connection_ro()
    try:
        row = connection.execute(
            """
            SELECT id
            FROM jobs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        raise StorageError(f"Unable to load last job: {e}") from e
    finally:
        connection.close()

def get_evidence_documents(candidate_id: str | None = None) -> list[dict]:
    connection = db_connection_ro()
    try:
        query = """
            SELECT evidence_id, candidate_id, source_id, category, date, text
            FROM evidence
        """
        parameters = ()
        if candidate_id is not None:
            query += " WHERE candidate_id = ?"
            parameters = (candidate_id,)
        query += " ORDER BY evidence_id"
        return [dict(row) for row in connection.execute(query, parameters)]
    except sqlite3.Error as e:
        raise StorageError(f"Unable to load evidence documents: {e}") from e
    finally:
        connection.close()


def get_cached_embedding_hashes(
    content_hashes: list[str], embedding_model: str
) -> set[str]:
    if not content_hashes:
        return set()

    connection = db_connection_ro()
    try:
        placeholders = ", ".join("?" for _ in content_hashes)
        rows = connection.execute(
            f"""
            SELECT content_hash
            FROM chunk_embeddings
            WHERE embedding_model = ?
              AND content_hash IN ({placeholders})
            """,
            (embedding_model, *content_hashes),
        )
        return {row["content_hash"] for row in rows}
    except sqlite3.Error as e:
        raise StorageError(f"Unable to look up cached embeddings: {e}") from e
    finally:
        connection.close()


def save_rag_index(chunks: list[dict], new_embeddings: list[dict]) -> None:
    chunk_rows = [
        (
            chunk["chunk_id"], chunk["evidence_id"], chunk["chunk_index"],
            chunk["text"], chunk["content_hash"],
        )
        for chunk in chunks
    ]
    embedding_rows = [
        (
            item["content_hash"], item["embedding_model"], item["dimensions"],
            json.dumps(item["embedding"]),
        )
        for item in new_embeddings
    ]
    connection = db_connection_rw()
    try:
        connection.execute("BEGIN")
        connection.execute("DELETE FROM chunks")
        connection.executemany(
            """
            INSERT INTO chunks (chunk_id, evidence_id, chunk_index, text, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )
        connection.executemany(
            """
            INSERT INTO chunk_embeddings
                (content_hash, embedding_model, dimensions, embedding)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(content_hash, embedding_model) DO NOTHING
            """,
            embedding_rows,
        )

        connection.execute(
            """
            DELETE FROM chunk_fts
            """
        )
        fts_rows = [
            (chunk["chunk_id"], chunk["text"],)
            for chunk in chunks
        ]
        connection.executemany(
            """
            INSERT INTO chunk_fts (
                chunk_id,
                text
            )
            VALUES (?, ?)
            """,
            fts_rows,
        )

        connection.commit()
    except sqlite3.Error as e:
        connection.rollback()
        raise StorageError(f"Unable to save RAG index: {e}") from e
    finally:
        connection.close()


def load_indexed_chunks(
    embedding_model: str, candidate_id: str | None = None
) -> list[dict]:
    connection = db_connection_ro()
    try:
        query = """
            SELECT c.chunk_id, c.evidence_id, e.candidate_id, e.source_id,
                   e.category, e.date, c.text, ce.embedding
            FROM chunks c
            JOIN evidence e ON e.evidence_id = c.evidence_id
            JOIN chunk_embeddings ce
              ON ce.content_hash = c.content_hash AND ce.embedding_model = ?
        """
        parameters = (embedding_model,)
        if candidate_id is not None:
            query += " WHERE e.candidate_id = ?"
            parameters += (candidate_id,)
        query += " ORDER BY c.evidence_id, c.chunk_index"
        result = []
        for row in connection.execute(query, parameters):
            chunk = dict(row)
            chunk["embedding"] = json.loads(chunk["embedding"])
            result.append(chunk)
        return result
    except (sqlite3.Error, json.JSONDecodeError) as e:
        raise StorageError(f"Unable to load indexed chunks: {e}") from e
    finally:
        connection.close()


if __name__ == "__main__":
    # print(get_all_jobs()) 
    # print(get_job("job_002"))  
    print(get_last_job())

#     with db_connection_ro() as connection:
#         row = connection.execute(
#             """
#             SELECT id, title, company, location
#             FROM jobs
#             WHERE id = ?
#             """,
#             ("job_002",),
#         ).fetchone()
#         print(dict(row) if row else None)

