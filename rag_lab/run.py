import json
from pathlib import Path

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
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def storage_documents_embed(documents, storage_path:Path = EMBED_PATH):
    chunks = chunk_documents(documents)
    embed_text = adding_embeddings(chunks)
    if not embed_text["success"]:
        return embed_text
    embed_text = embed_text["chunks"]
    with open(storage_path, "w", encoding="utf-8") as f:
        try:
            json.dump(embed_text, f, indent=4)
            return {
                "success": True,
                "storage_path": storage_path,
            }
        except Exception as e:
            return {
                "success": False,
                "error_msg": str(e)
            }
    

def load_documents_embed(storage_path:Path = EMBED_PATH):
    try:
        with open(storage_path, "r", encoding="utf-8") as f:
            result = json.load(f)
            return {
                "success": True,
                "load_result": result,
            }
    except Exception as e:
        return {
            "success": False,
            "error_msg": str(e)
        }


    

def main():
    documents = load_documents()

    chunks = chunk_documents(documents)
    texts = [
        chunk["text"] for chunk in chunks
    ]
    embeddings = embedding_texts(texts)

    storage_documents_embed(documents)

    evidence_embed = load_documents_embed()
    if evidence_embed['success']:
        evidence_embed = evidence_embed["load_result"]
    else:
        print("1", evidence_embed)
        return
    query_text = "What was the job that requires SQL skill"

    search_results = dense_search(query_text, evidence_embed)
    if search_results["success"]:
        for index, result in enumerate(search_results["results"]):
            print(f'number {index}: {result["text"]}')
                  

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

