import json
from pathlib import Path
import sys

# Keep the shared runtime error available when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from errors import HarnessError

from chunking import (
    MAX_CHUNK_TOKEN,
    chunk_documents,
    count_tokens,
    split_long_paragraph,
)
import tiktoken
from dense_search import embedding_texts, adding_embeddings,dense_search

DATA_PATH = Path(__file__).parent / "data" / "evidence.json"
EMBED_PATH = DATA_PATH.with_name("embed_text.json")


def load_documents() -> list[dict]:
    try:
        with DATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeError) as e:
        raise HarnessError(f"Unable to load documents from {DATA_PATH}: {e}") from e

def storage_documents_embed(documents, storage_path:Path = EMBED_PATH) -> Path:
    chunks = chunk_documents(documents)
    embed_text = adding_embeddings(chunks)
    try:
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(embed_text, f, indent=4)
    except OSError as e:
        raise HarnessError(f"Unable to store embeddings at {storage_path}: {e}") from e
    return storage_path
    

def load_documents_embed(storage_path:Path = EMBED_PATH) -> list[dict]:
    try:
        with open(storage_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeError) as e:
        raise HarnessError(f"Unable to load embeddings from {storage_path}: {e}") from e


    

def main():
    try:
        documents = load_documents()

        chunks = chunk_documents(documents)
        texts = [
            chunk["text"] for chunk in chunks
        ]
        embeddings = embedding_texts(texts)

        storage_documents_embed(documents)

        evidence_embed = load_documents_embed()
        query_text = "What was the job that requires SQL skill"

        search_results = dense_search(query_text, evidence_embed)
        for index, result in enumerate(search_results):
            print(f'number {index}: {result["text"]}')
    except HarnessError as e:
        print(f"[Runtime Error] {e}")
        return
                  

    # search_results = dense_search(query_text, embed_text)



    # print(f"documents: {len(documents)}")
    # print(f"chunks: {len(chunks)}")
    # print()
    # print(f"embeddings: {len(embeddings["embedding"])}")
    # print(f"dimensions: {len(embeddings["embedding"][0])}") 
    

    # for chunk in chunks:
    #     word_count = len(chunk["text"].split())
    #     token_count = count_tokens(chunk["text"])
    #     print(
    #         f"{chunk['chunk_id']} "
    #         f"| source={chunk['source_id']} "
    #         f"| words={word_count} "
    #         f"| tokens={token_count}"
    #     )
    #     print(chunk["text"])
    #     print("-" * 80)

    # print("\n=== long paragraph fallback test ===")

    # first_paragraph = documents[0]["text"].split("\n\n")[0]

    # long_paragraph = " ".join(
    #     [first_paragraph] * 10
    # )

    # # print(
    # #     "long paragraph tokens:",
    # #     count_tokens(long_paragraph),
    # # )

    # parts = split_long_paragraph(
    #     long_paragraph,
    #     max_tokens=MAX_CHUNK_TOKEN,
    # )



    # for index, part in enumerate(parts):
    #     print(
    #         f"part {index}: "
    #         f"{count_tokens(part)} tokens"
    #     )
    # assert len(documents) == 3
    # assert len(chunks) == 12

    # assert all(chunk["text"] for chunk in chunks)
    # assert all(chunk["candidate_id"] == "candidate_001" for chunk in chunks)


if __name__ == "__main__":
    main()

