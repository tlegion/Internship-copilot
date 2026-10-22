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


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    tokens = encoding().encode(text, disallowed_special=())
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        # A token boundary can fall inside a Unicode character. Back up until
        # decoding is lossless; the next piece starts at the same boundary.
        while end > start:
            try:
                piece = encoding().decode(tokens[start:end], errors="strict")
                break
            except UnicodeDecodeError:
                end -= 1
        if end == start:
            raise ValueError("Token budget cannot fit a character")
        chunks.append(piece)
        start = end
    return chunks


def split_long_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])(?=\s)", paragraph)
    chunks = []
    current = ""
    for sentence in sentences:
        if count_tokens(sentence) > max_tokens:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_by_tokens(sentence, max_tokens))
        elif count_tokens(current + sentence) <= max_tokens:
            current += sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def trailing_overlap(text: str, max_tokens: int) -> str:
    if max_tokens == 0:
        return ""
    tokens = encoding().encode(text, disallowed_special=())[-max_tokens:]
    while tokens:
        try:
            overlap = encoding().decode(tokens, errors="strict")
            break
        except UnicodeDecodeError:
            tokens = tokens[1:]
    else:
        return ""
    while count_tokens(overlap) > max_tokens:
        overlap = overlap[1:]
    return overlap


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
        # Reserve room for headers and overlap when splitting long sentences.
        piece_limit = max_tokens - count_tokens(prefix) - overlap_tokens - 2
        if piece_limit <= 0:
            raise ValueError("Token budget cannot fit headers and overlap")
        pieces = []
        for paragraph in re.split(r"(?<=\n\n)(?=\S)", body):
            if count_tokens(paragraph) <= piece_limit:
                pieces.append(paragraph)
            else:
                pieces.extend(split_long_paragraph(paragraph, piece_limit))

        texts = []
        current = ""
        for piece in pieces:
            if count_tokens(prefix + piece) > max_tokens:
                raise ValueError("Decoded chunk exceeds the token budget")
            if count_tokens(prefix + current + piece) > max_tokens:
                texts.append(prefix + current)
                current = trailing_overlap(current, overlap_tokens)
                while count_tokens(prefix + current + piece) > max_tokens:
                    current = current[1:]
            current += piece
        if current:
            texts.append(prefix + current)

    chunks = []
    for index, text in enumerate(texts):
        chunks.append({
            "chunk_id": f"{job['job_id']}_chunk_{index:03d}",
            "job_id": job["job_id"],
            "chunk_index": index,
            "text": text,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return chunks


def chunk_jobs(jobs: list[dict]) -> list[dict]:
    chunks = []
    for job in jobs:
        chunks.extend(chunk_job(job))
    return chunks
