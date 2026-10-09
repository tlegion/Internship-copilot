from rag_lab.hybrid_search import (
    hybrid_search,
)


def search_evidence_hybrid(
    query: str,
    keywords: list[str],
    candidate_id: str,
) -> list[dict]:

    return hybrid_search(
        query=query,
        keywords=keywords,
        candidate_id=candidate_id,
        top_k=3,
    )