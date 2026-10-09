import sqlite3

import re
from errors import StorageError
from storages.database import db_connection_ro

def build_fts_query(query:str) -> str:
    terms = re.findall(r"\w+", query)

    return " OR ".join((f'"{term}"' for term in terms))


def bm25_search( keywords:list[str], candidate_id:str | None = None, top_k:int = 3) ->list[dict]:
    # fts_query = keywords

    # if not fts_query: return []

    connection = db_connection_ro()

    try:
        sql = """
            SELECT
                c.chunk_id,
                c.evidence_id, 
                e.candidate_id,
                e.source_id,
                c.text,
                bm25(chunk_fts) AS score
                
            FROM chunk_fts
            
            JOIN chunks c
                ON c.chunk_id = chunk_fts.chunk_id

            JOIN evidence e
                ON e.evidence_id = c.evidence_id

            WHERE chunk_fts MATCH ?
        """

        query = " OR ".join(f"{word}" for word in keywords)

        parameter = [query]

        if candidate_id is not None:
            sql += """
                AND e.candidate_id = ?
            """
            parameter.append(candidate_id)

        sql += """
            ORDER BY score
            LIMIT ?
        """
        parameter.append(top_k)

        rows = connection.execute(
            sql,
            parameter,
        ).fetchall()

        return [ dict(row) for row in rows]

    except sqlite3.Error as e:
        raise StorageError(
            f"BM25 search failed: {e}"
        ) from e
    finally:
        connection.close()