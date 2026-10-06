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


if __name__ == "__main__":
    with db_connection_ro() as connection:
        row = connection.execute(
            """
            SELECT id, title, company, location
            FROM jobs
            WHERE id = ?
            """,
            ("job_002",),
        )
        print(dict(row) if row else None)