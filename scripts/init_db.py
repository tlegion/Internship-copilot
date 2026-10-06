import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "internship.db"

SCHEMA_PATH = ROOT / "sql" / "schema.sql"
SEED_PATH = ROOT / "sql" / "seed_demo.sql"


def init_db():
    if DB_PATH.exists():
        raise FileExistsError(
            f"Database already exists: {DB_PATH}"
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    seed_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.executescript(
            f"""
            BEGIN;

            {schema_sql}

            {seed_sql}

            COMMIT;
            """
        )
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print(f"Database initialized: {DB_PATH}")


if __name__ == "__main__":
    init_db()
