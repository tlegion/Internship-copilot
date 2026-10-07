import hashlib

from errors import ProviderError
from rag_lab.chunking import EMBEDDING_MODEL, chunk_documents
from rag_lab.dense_search import embedding_texts
from storages.database import (
    get_cached_embedding_hashes,
    get_evidence_documents,
    save_rag_index,
)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sync_rag_index() -> dict:
    documents = get_evidence_documents()
    chunks = chunk_documents(documents)
    hash_to_text = {}
    for chunk in chunks:
        chunk["content_hash"] = content_hash(chunk["text"])
        hash_to_text.setdefault(chunk["content_hash"], chunk["text"])

    hashes = list(hash_to_text)
    cached_hashes = get_cached_embedding_hashes(hashes, EMBEDDING_MODEL)
    missing_hashes = [value for value in hashes if value not in cached_hashes]

    new_embeddings = []
    if missing_hashes:
        vectors = embedding_texts([hash_to_text[value] for value in missing_hashes])
        if len(vectors) != len(missing_hashes):
            raise ProviderError("Number of missing chunks and embeddings do not match.")
        new_embeddings = [
            {
                "content_hash": value,
                "embedding_model": EMBEDDING_MODEL,
                "dimensions": len(vector),
                "embedding": vector,
            }
            for value, vector in zip(missing_hashes, vectors, strict=True)
        ]

    save_rag_index(chunks, new_embeddings)
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "cached_embeddings": len(cached_hashes),
        "new_embeddings": len(new_embeddings),
    }
