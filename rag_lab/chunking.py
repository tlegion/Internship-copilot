from typing import Any
import tiktoken
import re

MAX_CHUNK_TOKEN = 120

EMBEDDING_MODEL = "text-embedding-3-small"
ENCODING = tiktoken.encoding_for_model(EMBEDDING_MODEL)

def count_tokens(text:str) ->int:
    return len(ENCODING.encode(text))

def split_by_tokens(text:str, max_token:int) -> list[str]:
    tokens = ENCODING.encode(text)
    chunks = []

    for start in range(0, len(tokens), max_token):
        chunk_tokens = tokens[start:start + max_token]
        chunks.append(ENCODING.decode(chunk_tokens))

    return chunks


def split_long_paragraph(
        paragraph: str,
        max_tokens: int,
)-> list[str]:
    sentences = [
        sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
        if sentence.strip()
    ]

    chunks = []
    current_sentences = []

    for sentence in sentences:
        if count_tokens(sentence) > max_tokens:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []

            chunks.extend(
                split_by_tokens(sentence, max_tokens)
            )
            continue

        candidate = " ".join(
            current_sentences + [sentence]
        )

        if count_tokens(candidate) <= max_tokens:
            current_sentences.append(sentence)

        else:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks

def chunk_document(
        document:dict[str, Any], 
        max_tokens:int=MAX_CHUNK_TOKEN
) -> list[dict[str, Any]]:
    paragraphs = [
        paragraph.strip() for paragraph in document["text"].split("\n\n")
        if paragraph.strip()
    ]

    chunks = []

    chunk_texts = []

    for paragraph in paragraphs:
        if count_tokens(paragraph) <= max_tokens:
            chunk_texts.append(paragraph)

        else:
            chunk_texts.extend(
                split_long_paragraph(
                    paragraph,
                    max_tokens,
                )
            )


    for index, text in enumerate(chunk_texts):
        chunk = {
            "chunk_id": f"{document['evidence_id']}_chunk_{index:03d}",
            "evidence_id": document["evidence_id"],
            "candidate_id": document["candidate_id"],
            "source_id": document["source_id"],
            "category": document["category"],
            "date": document.get("date"),
            "text": text,
        }

        chunks.append(chunk)

    return chunks

def chunk_documents(
        documents:list[dict[str, Any]],
        max_tokens:int = MAX_CHUNK_TOKEN,
) -> list[dict[str, Any]]:
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, max_tokens))
    return chunks