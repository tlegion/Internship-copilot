import json
from pathlib import Path

from chunking import (
    MAX_CHUNK_TOKEN,
    chunk_documents,
    count_tokens,
    split_long_paragraph,
)
import tiktoken

DATA_PATH = Path(__file__).parent / "data" / "evidence.json"

def load_documents() -> list[dict]:
  with DATA_PATH.open("r", encoding="utf-8") as f:
      return json.load(f)

def main():
    documents = load_documents()

    chunks = chunk_documents(documents)
    
    print(f"documents: {len(documents)}")
    print(f"chunks: {len(chunks)}")
    print()

    for chunk in chunks:
        word_count = len(chunk["text"].split())
        token_count = count_tokens(chunk["text"])
        print(
            f"{chunk['chunk_id']} "
            f"| source={chunk['source_id']} "
            f"| words={word_count} "
            f"| tokens={token_count}"
        )
        print(chunk["text"])
        print("-" * 80)

    print("\n=== long paragraph fallback test ===")

    first_paragraph = documents[0]["text"].split("\n\n")[0]

    long_paragraph = " ".join(
        [first_paragraph] * 10
    )

    print(
        "long paragraph tokens:",
        count_tokens(long_paragraph),
    )

    parts = split_long_paragraph(
        long_paragraph,
        max_tokens=MAX_CHUNK_TOKEN,
    )

    for index, part in enumerate(parts):
        print(
            f"part {index}: "
            f"{count_tokens(part)} tokens"
        )
  # assert len(documents) == 3
  # assert len(chunks) == 12

  # assert all(chunk["text"] for chunk in chunks)
  # assert all(chunk["candidate_id"] == "candidate_001" for chunk in chunks)


if __name__ == "__main__":
  main()

