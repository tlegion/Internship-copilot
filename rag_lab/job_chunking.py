"""Deterministic job documents with bounded chunks and overlapping context."""
import hashlib
import json
import re
from functools import lru_cache

import tiktoken

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CHUNK_TOKENS = 800
OVERLAP_TOKENS = 50


@lru_cache(maxsize=1)
def encoding():
    return tiktoken.encoding_for_model(EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(encoding().encode(text, disallowed_special=()))


def job_header(job: dict) -> str:
    # Quoting preserves field boundaries and distinguishes null from empty text.
    return "\n".join(
        f"{field.title()}: {json.dumps(job[field], ensure_ascii=False)}"
        for field in ("title", "company", "location")
    ) + "\nDescription:\n"


def _fitting_end(text: str, start: int, prefix: str, budget: int) -> int:
    low, high = start, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(prefix + text[start:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def chunk_job(job: dict, max_tokens: int = MAX_CHUNK_TOKENS,
              overlap_tokens: int = OVERLAP_TOKENS) -> list[dict]:
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("Chunk overlap must be smaller than the token budget")
    header = job_header(job)
    document = header + job["description"]
    if count_tokens(document) <= max_tokens:
        texts = [document]
    else:
        # Repeat metadata in long descriptions; unusually large metadata falls
        # back to splitting the whole document without an oversized prefix.
        prefix = header if count_tokens(header) < max_tokens // 2 else ""
        body = job["description"] if prefix else document
        texts = []
        start = 0
        while start < len(body):
            end = _fitting_end(body, start, prefix, max_tokens)
            if end == start:
                raise ValueError("Token budget cannot fit a character")
            if end < len(body):
                boundaries = list(re.finditer(r"\n\n|(?<=[.!?])\s+", body[start:end]))
                if boundaries and boundaries[-1].end() > (end - start) // 2:
                    end = start + boundaries[-1].end()
            # A new text boundary can change BPE tokenization slightly.
            while count_tokens(prefix + body[start:end]) > max_tokens:
                end -= 1
            texts.append(prefix + body[start:end])
            if end == len(body):
                break
            # Find an approximately 50-token suffix, without breaking Unicode.
            low, high = start + 1, end
            while low < high:
                middle = (low + high) // 2
                if count_tokens(body[middle:end]) <= overlap_tokens:
                    high = middle
                else:
                    low = middle + 1
            start = low
    return [
        {"chunk_id": f"{job['job_id']}_chunk_{index:03d}",
         "job_id": job["job_id"], "chunk_index": index, "text": text,
         "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        for index, text in enumerate(texts)
    ]


def chunk_jobs(jobs: list[dict]) -> list[dict]:
    return [chunk for job in jobs for chunk in chunk_job(job)]
