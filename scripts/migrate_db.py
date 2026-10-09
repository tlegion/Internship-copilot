import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "internship.db"

MIGRATIONS = [
    ROOT / "sql" / "migrations" / "V000__initial.sql",
    ROOT / "sql" / "migrations" / "V001__add_rag_storage.sql",
    ROOT / "sql" / "migrations" / "V002__add_chunks_fts.sql",
]

LATEST_VERSION = len(MIGRATIONS)


def get_db_version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "PRAGMA user_version"
    ).fetchone()

    return row[0]


def migrate_db():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database does not exist: {DB_PATH}"
        )

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")

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

        for next_version in range(
            current_version + 1,
            LATEST_VERSION + 1,
        ):
            migration_path = MIGRATIONS[next_version - 1]

            migration_sql = migration_path.read_text(
                encoding="utf-8"
            )

            print(
                f"Applying migration: "
                f"{migration_path.name}"
            )

            connection.executescript(
                f"""
                BEGIN;

                {migration_sql}

                PRAGMA user_version = {next_version};

                COMMIT;
                """
            )

        print("Database migration complete.")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    migrate_db()