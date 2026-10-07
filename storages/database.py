import sqlite3

from pathlib import Path
from errors import HarnessError


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
    except sqlite3.Error as e:
        raise HarnessError(f"Unable to open database: {e}") from e
    connection.row_factory = sqlite3.Row
    return connection

def db_connection_rw():
    try:
        connection = sqlite3.connect(
            f"file:{DB_PATH}?mode=rw",
            uri=True,
        )
    except sqlite3.Error as e:
        raise HarnessError(f"Unable to open database: {e}") from e
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
        raise HarnessError(f"Unable to look up job {job_id}: {e}") from e
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
        raise HarnessError(f"Unable to load jobs: {e}") from e
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

        raise HarnessError(f"Job creation failed: {e}") from e
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
            AND location = ?
            """,
            (title, company, description, location)
        ).fetchall()
        return True if rows else False
    except sqlite3.Error as e:
        raise HarnessError(f"Unable to check whether job exists: {e}") from e
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
        raise HarnessError(f"Unable to load last job: {e}") from e
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

