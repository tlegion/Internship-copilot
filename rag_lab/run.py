import json
from pathlib import Path

from chunking import chunk_documents
import tiktoken

DATA_PATH = Path(__file__).parent / "data" / "evidence.json"

encoding = tiktoken.get_encoding("cl100k_base")

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
    token_count = len(encoding.encode(chunk["text"]))

    print(
        f"{chunk['chunk_id']} "
        f"| source={chunk['source_id']} "
        f"| words={word_count} "
        f"| tokens={token_count}"
    )
    print(chunk["text"])
    print("-" * 80)
  # assert len(documents) == 3
  # assert len(chunks) == 12

  # assert all(chunk["text"] for chunk in chunks)
  # assert all(chunk["candidate_id"] == "candidate_001" for chunk in chunks)


if __name__ == "__main__":
  main()

