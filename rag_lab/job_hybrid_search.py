import math

from errors import ProviderError, StorageError
from rag_lab.bm25_search import build_fts_query
from rag_lab.job_chunking import EMBEDDING_MODEL, chunk_jobs
from rag_lab.job_indexing import embedding_texts
from storages.job_index import load_job_search_data

DENSE_WEIGHT = 0.8
BM25_WEIGHT = 0.2


def fuse_job_chunks(dense: list[dict], bm25: list[dict]) -> list[dict]:
    fused = {}
    for results, weight, branch in ((dense, DENSE_WEIGHT, "dense"), (bm25, BM25_WEIGHT, "bm25")):
        for rank, row in enumerate(results, 1):
            item = fused.setdefault(row["chunk_id"], {**row, "rrf_score": 0.0})
            item["rrf_score"] += weight / rank
            item[f"{branch}_rank"] = rank
            item[f"{branch}_score"] = row["score"]
    return sorted(fused.values(), key=lambda row: (-row["rrf_score"], row["chunk_id"]))


def hybrid_job_search(query: str, keywords: list[str], company: str | None,
                      location: str | None, top_k: int) -> dict:
    chunks, bm25 = load_job_search_data(EMBEDDING_MODEL, chunk_jobs,
                                       build_fts_query(keywords), company, location)
    if not chunks:
        return {"jobs": []}
    from rag_lab.dense_search import cosine_similarity
    vectors = embedding_texts([query])
    if len(vectors) != 1:
        raise ProviderError("Expected one Job search query embedding.")
    if (not isinstance(vectors[0], list) or not vectors[0]
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in vectors[0])
            or not any(vectors[0])):
        raise ProviderError("Invalid Job search query embedding returned by provider.")
    try:
        dense = [{**chunk, "score": cosine_similarity(vectors[0], chunk["embedding"])} for chunk in chunks]
    except (ValueError, TypeError) as e:
        raise StorageError(f"Invalid Job search embeddings: {e}") from e
    dense.sort(key=lambda row: (-row["score"], row["chunk_id"]))
    # Scan all eligible chunks so a long job cannot exhaust the job budget.
    seen_job_ids = set()
    jobs = []
    for row in fuse_job_chunks(dense, bm25):
        if row["job_id"] in seen_job_ids:
            continue
        seen_job_ids.add(row["job_id"])
        jobs.append({**{field: row[field] for field in
                        ("job_id", "title", "company", "location", "description")},
                     "rank": len(jobs) + 1})
        if len(jobs) == top_k:
            break
    return {"jobs": jobs}
