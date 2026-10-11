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
    hash_to_text = {chunk["content_hash"]: chunk["text"] for chunk in chunks}
    cached = get_cached_job_hashes(list(hash_to_text), EMBEDDING_MODEL)
    missing = [value for value in hash_to_text if value not in cached]
    new_embeddings = []
    if missing:
        vectors = embedding_texts([hash_to_text[value] for value in missing])
        if len(vectors) != len(missing):
            raise ProviderError("Number of missing job chunks and embeddings do not match.")
        if any(not isinstance(vector, list) or not vector
               or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
               or not any(vector) for vector in vectors) or len({len(vector) for vector in vectors}) != 1:
            raise ProviderError("Invalid Job chunk embeddings returned by provider.")
        new_embeddings = [
            {"content_hash": value, "embedding_model": EMBEDDING_MODEL,
             "dimensions": len(vector), "embedding": vector}
            for value, vector in zip(missing, vectors, strict=True)
        ]
    save_job_index(jobs, chunks, new_embeddings)
    return {"jobs": len(jobs), "chunks": len(chunks),
            "cached_embeddings": len(cached), "new_embeddings": len(new_embeddings)}
