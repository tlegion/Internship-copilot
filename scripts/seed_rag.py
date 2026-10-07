import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data" / "internship.db"
EVIDENCE_PATH = (
    ROOT
    / "rag_lab"
    / "data"
    / "evidence.json"
)


def seed_rag():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database does not exist: {DB_PATH}"
        )

    documents = json.loads(
        EVIDENCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    rows = [
        (
            document["evidence_id"],
            document["candidate_id"],
            document["source_id"],
            document["category"],
            document.get("date"),
            document["text"],
        )
        for document in documents
    ]

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.executemany(
            """
            INSERT INTO evidence (
                evidence_id,
                candidate_id,
                source_id,
                category,
                date,
                text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(evidence_id)
            DO UPDATE SET
                candidate_id = excluded.candidate_id,
                source_id = excluded.source_id,
                category = excluded.category,
                date = excluded.date,
                text = excluded.text
            """,
            rows,
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    print(
        f"Seeded/synced {len(rows)} evidence documents."
    )


if __name__ == "__main__":
    seed_rag()
