import math

from errors import ProviderError, StorageError
from rag_lab.bm25_search import build_fts_query
from rag_lab.job_chunking import EMBEDDING_MODEL
from rag_lab.job_indexing import embedding_texts
from storages.job_index import load_job_search_data
from rag_lab.hybrid_search import reciprocal_rank_fusion


def fuse_job_chunks(dense: list[dict], bm25: list[dict]) -> list[dict]:
    # Keep every candidate until after job deduplication.
    results = reciprocal_rank_fusion(dense, bm25, top_k=len(dense) + len(bm25))
    dense_scores = {row["chunk_id"]: row["score"] for row in dense}
    bm25_scores = {row["chunk_id"]: row["score"] for row in bm25}
    for row in results:
        row["dense_score"] = dense_scores.get(row["chunk_id"])
        row["bm25_score"] = bm25_scores.get(row["chunk_id"])
    return results


def dense_job_search(query: str, chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []
    from rag_lab.dense_search import cosine_similarity
    vectors = embedding_texts([query])
    if len(vectors) != 1:
        raise ProviderError("Expected one Job search query embedding.")
    if (not isinstance(vectors[0], list) or not vectors[0]
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in vectors[0])
            or not any(vectors[0])):
        raise ProviderError("Invalid Job search query embedding returned by provider.")
    try:
        results = []
        for chunk in chunks:
            score = cosine_similarity(vectors[0], chunk["embedding"])
            results.append({**chunk, "score": score})
    except (ValueError, TypeError) as e:
        raise StorageError(f"Invalid Job search embeddings: {e}") from e
    results.sort(key=lambda row: row["score"], reverse=True)
    return results


def hybrid_job_search(query: str, keywords: list[str], company: str | None,
                      location: str | None, top_k: int) -> dict:
    indexed_chunks, bm25_results = load_job_search_data(
        EMBEDDING_MODEL, build_fts_query(keywords), company, location
    )
    dense_results = dense_job_search(query, indexed_chunks)
    fused_results = fuse_job_chunks(dense_results, bm25_results)
    # Scan all eligible chunks so a long job cannot exhaust the job budget.
    seen_job_ids = set()
    jobs = []
    for row in fused_results:
        if row["job_id"] in seen_job_ids:
            continue
        seen_job_ids.add(row["job_id"])
        jobs.append({
            "job_id": row["job_id"],
            "title": row["title"],
            "company": row["company"],
            "location": row["location"],
            "description": row["description"],
            "rank": len(jobs) + 1,
        })
        if len(jobs) == top_k:
            break
    return {"jobs": jobs}
