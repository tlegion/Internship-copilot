from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from chunking import EMBEDDING_MODEL
import math

ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")

client = OpenAI()



def embedding_texts(
        texts: list[str],
) -> dict:
    try:
        response = client.embeddings.create(
            model = EMBEDDING_MODEL,
            input=texts,
        )

        return {
            "embedding" : [item.embedding for item in response.data],
            "success": True
        }
    except Exception as e:
        return {
            "success": False,
            "error_msg": str(e)
        }


def adding_embeddings(
        chunks: list[dict],
) -> dict:
    texts = [chunk["text"] for chunk in chunks]

    embedding_results = embedding_texts(texts)

    if not embedding_results["success"]:
        return embedding_results

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
        chunk_embeddings: list[list[float]],
        top_k: int = 3,
) -> dict:
    if len(chunks) != len(chunk_embeddings):
        return {
            "success": False,
            "error_msg": (
                "Number of chunks and embeddings do not match."
            ),
        }
    query_result = embedding_texts([query])

    if not query_result["success"]:
        return query_result

    query_embedding = query_result["embedding"][0]

    results = []
    for chunk, chunk_embeddings in zip(
        chunks,
        chunk_embeddings,
        strict=True
    )


# if __name__ == "__main__":
    # print(embedding_texts(input()))