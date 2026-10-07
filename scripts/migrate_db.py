import sqlite3

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "internship.db"

LATEST_VERSION = 1

def get_db_version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "PRAGMA user_version"
    ).fetchone()

    return row[0]

def migrate_db():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f'Database does not exist: {DB_PATH}'
        )

    connection = sqlite3.connect(DB_PATH)

    try:
        current_version = get_db_version(connection)
        print(
            f"Database version: "
            f"{current_version} -> {LATEST_VERSION}"
        )

        if current_version > LATEST_VERSION:
            raise RuntimeError(
                "Database is newer than this application"
            )

        if current_version == LATEST_VERSION:
            print("Database is already up to date.")
            return

    finally:
        connection.close()

if __name__ == "__main__":
    migrate_db()