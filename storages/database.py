import sqlite3

from pathlib import Path


DB_PATH = (
    Path(__file__).resolve().parents[1]
    /"data"
    /"internship.db"
)

def db_connection_ro():
    connection = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
    )
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
    finally:
        connection.close()


if __name__ == "__main__":
    print(get_job("job_002"))    
    print(get_job("job_999"))
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

