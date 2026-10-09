from rag_lab.chunking import EMBEDDING_MODEL
from rag_lab.dense_search import dense_search
from rag_lab.bm25_search import bm25_search
from storages.database import load_indexed_chunks


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 3,
    rank_constant: int = 60,
) -> list[dict]:

    fused = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]

        fused[chunk_id] = {
            **result,
            "rrf_score": 0.0,
            "dense_rank": rank,
            "bm25_rank": None,
        }

        fused[chunk_id]["rrf_score"] += (
            1 / (rank_constant + rank)
        )

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                **result,
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": rank,
            }
        else:
            fused[chunk_id]["bm25_rank"] = rank

        fused[chunk_id]["rrf_score"] += (
            1 / (rank_constant + rank)
        )

    results = list(fused.values())

    results.sort(
        key=lambda result: result["rrf_score"],
        reverse=True,
    )

    return results[:top_k]


def hybrid_search(
    query: str,
    keywords: list[str] | None,
    candidate_id: str | None = None,
    top_k: int = 3,
    retrieval_k: int = 10,
) -> list[dict]:

    indexed_chunks = load_indexed_chunks(
        EMBEDDING_MODEL,
        candidate_id=candidate_id,
    )

    dense_results = dense_search(
        query,
        indexed_chunks,
        top_k=retrieval_k,
    )

    bm25_results = bm25_search(
        query,
        candidate_id=candidate_id,
        top_k=retrieval_k,
    )

    return reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=top_k,
    )