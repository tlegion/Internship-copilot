from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIError
from errors import ProviderError

from rag_lab.chunking import EMBEDDING_MODEL
import math
import os

ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")




def embedding_texts(
        texts: list[str],
) -> list[list[float]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ProviderError("missing API key: OpenAI embeddings")
    client = OpenAI()
    try:
        response = client.embeddings.create(
            model = EMBEDDING_MODEL,
            input=texts,
        )

    except APIError as e:
        raise ProviderError(str(e)) from e

    return [item.embedding for item in response.data]


def adding_embeddings(
        chunks: list[dict],
) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_texts(texts)

    if len(chunks) != len(embeddings):
        raise ProviderError("Number of chunks and embeddings do not match.")

    result = []
    for chunk, embedding in zip(chunks, embeddings, strict = True):
        result.append(
            {
                **chunk,
                "embedding": embedding
            }
        )

    return result


def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(
        a*b for a,b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        raise ValueError("cannot compare zero vectors")

    return dot_product / (norm_a * norm_b)

def dense_search(
        query: str,
        chunks: list[dict],
        top_k: int = 3,
) -> list[dict]:

    if not chunks:
        return []

    query_embedding = embedding_texts([query])[0]

    results = []

    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        results.append(
            
            {
                "chunk_id": chunk["chunk_id"],
                "evidence_id": chunk["evidence_id"],
                "candidate_id": chunk["candidate_id"],
                "source_id": chunk["source_id"],
                "category": chunk["category"],
                "text": chunk["text"],
                "score": score,
            }
        )

    results.sort(
        key=lambda result:result["score"],
        reverse=True,
    )

    return results[:top_k]




# if __name__ == "__main__":
    # print(embedding_texts(input()))
