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

    """Search the candidate's evidence when answering questions about their experience, skills,
    projects, or qualifications. Use query as a standalone semantic description of the user's
    current retrieval intent. If the user's current message is already self-contained, preserve it
    closely. If it depends on earlier conversation, include only the context needed to make the
    query standalone. keywords are lexical search anchors for exact matching. Use only distinctive
    technical terms, technology names, skills, identifiers, error codes, product names, or short
    phrases relevant to the user's request. Exclude generic words such as 'I', 'have', 'did', 'use',
    'experience', and 'about'. Do not invent terms that assume the answer. If there are no useful
    exact lexical anchors, pass an empty list.

    Args:
        query: Standalone semantic search query representing the user's current retrieval
            intent. The current evidence corpus is in English, so write this query in English
            even if the user speaks another language. Preserve the user's meaning and do not add
            unsupported facts. If the current user message depends on earlier conversation,
            include only the context needed to make the query standalone.
        keywords: Distinctive English lexical search anchors for the current English evidence
            corpus. Preserve exact technical identifiers, API names, technology names, error
            codes, and code terms as written. Exclude generic conversational words. Do not
            invent likely answer terms. If there are no useful exact lexical anchors, use an
            empty list.
        candidate_id: Candidate whose evidence should be searched, for example candidate_001.
    """
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