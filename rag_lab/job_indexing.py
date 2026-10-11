import math

from errors import ProviderError
from rag_lab.job_chunking import EMBEDDING_MODEL, chunk_jobs
from storages.database import get_all_jobs
from storages.job_index import get_cached_job_hashes, save_job_index


def embedding_texts(texts: list[str]) -> list[list[float]]:
    # Lazy import keeps browsing tools independent of tokenizer initialization.
    from rag_lab.dense_search import embedding_texts as embed
    return embed(texts)


def sync_job_index() -> dict:
    jobs = get_all_jobs()
    chunks = chunk_jobs(jobs)
    hash_to_text = {}
    for chunk in chunks:
        hash_to_text.setdefault(chunk["content_hash"], chunk["text"])

    hashes = list(hash_to_text)
    cached_hashes = get_cached_job_hashes(hashes, EMBEDDING_MODEL)
    missing_hashes = [value for value in hashes if value not in cached_hashes]
    new_embeddings = []
    if missing_hashes:
        vectors = embedding_texts([hash_to_text[value] for value in missing_hashes])
        if len(vectors) != len(missing_hashes):
            raise ProviderError("Number of missing job chunks and embeddings do not match.")
        for vector in vectors:
            if (not isinstance(vector, list) or not vector or not any(vector)
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
                raise ProviderError("Invalid Job chunk embeddings returned by provider.")
            if len(vector) != len(vectors[0]):
                raise ProviderError("Job chunk embedding dimensions do not match.")
        new_embeddings = [
            {
                "content_hash": value,
                "embedding_model": EMBEDDING_MODEL,
                "dimensions": len(vector),
                "embedding": vector,
            }
            for value, vector in zip(missing_hashes, vectors, strict=True)
        ]
    save_job_index(jobs, chunks, new_embeddings)
    return {
        "jobs": len(jobs),
        "chunks": len(chunks),
        "cached_embeddings": len(cached_hashes),
        "new_embeddings": len(new_embeddings),
    }
