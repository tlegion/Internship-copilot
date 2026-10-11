from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI, APIError
from pydantic import BaseModel

from errors import ProviderError


ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")

RERANK_MODEL = "gpt-6-luna"


class RerankOutput(BaseModel):
    ranked_indices: list[int]


def rerank_results(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:

    if not candidates:
        return []

    if len(candidates) == 1:
        return [
            {
                **candidates[0],
                "rerank_rank": 1,
            }
        ]

    final_k = min(
        top_k,
        len(candidates),
    )

    candidate_text = "\n\n".join(
        f"[{index}] {candidate['text']}"
        for index, candidate in enumerate(candidates)
    )

    prompt = f"""
Query:
{query}

Candidate evidence:
{candidate_text}

Rank the candidate evidence by how directly and completely it helps answer the query.

Return exactly {final_k} unique candidate indices,
ordered from most relevant to least relevant.

Only select from the provided candidate indices.
Do not answer the query itself.
""".strip()

    if not os.environ.get("OPENAI_API_KEY"):
        return candidates[:final_k]

    client = OpenAI(
        timeout=30,
        max_retries=1,
    )

    try:
        response = client.responses.parse(
            model=RERANK_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a retrieval reranker. "
                        "Rank candidate evidence only by relevance "
                        "to the supplied search query."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            text_format=RerankOutput,
            reasoning={
                "effort": "none",
            },
            store=False,
        )

    except APIError:
        return candidates[:final_k]

    parsed = response.output_parsed

    if parsed is None:
        return candidates[:final_k]

    ranked_indices = parsed.ranked_indices

    if (
        len(ranked_indices) != final_k
        or len(set(ranked_indices)) != final_k
        or any(
            index < 0 or index >= len(candidates)
            for index in ranked_indices
        )
    ):
        return candidates[:final_k]

    return [
        {
            **candidates[index],
            "rerank_rank": rank,
        }
        for rank, index in enumerate(
            ranked_indices,
            start=1,
        )
    ]