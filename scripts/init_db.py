import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.migrate_db import migrate_db
from scripts.seed_rag import seed_rag

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "internship.db"

SEED_PATH = ROOT / "sql" / "seed_demo.sql"


def init_db():
    if DB_PATH.exists():
        raise FileExistsError(
            f"Database already exists: {DB_PATH}"
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sqlite3.connect(DB_PATH).close()

    try:
        migrate_db()

        seed_sql = SEED_PATH.read_text(
            encoding="utf-8"
        )

        connection = sqlite3.connect(DB_PATH)

        try:
            connection.executescript(
                f"""
                BEGIN;

                {seed_sql}

                COMMIT;
                """
            )

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

        seed_rag()

    except Exception:
        DB_PATH.unlink(missing_ok=True)
        raise

    print(f"Database initialized: {DB_PATH}")


if __name__ == "__main__":
    init_db()
