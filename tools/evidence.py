from rag_lab.hybrid_search import (
    hybrid_search,
)

from rag_lab.hybrid_search import hybrid_search
from rag_lab.reranker import rerank_results


def search_evidence_hybrid(
    query: str,
    keywords: list[str],
    candidate_id: str,
) -> list[dict]:

    candidates = hybrid_search(
        query=query,
        keywords=keywords,
        candidate_id=candidate_id,
        top_k=10,
    )

    print(candidates)

    return rerank_results(
        query=query,
        candidates=candidates,
        top_k=3,
    )