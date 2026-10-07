from pathlib import Path
import sys

# Make shared modules available when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from errors import HarnessError
from rag_lab.chunking import EMBEDDING_MODEL
from rag_lab.dense_search import dense_search
from rag_lab.indexing import sync_rag_index
from storages.database import load_indexed_chunks


def main():
    try:
        sync_result = sync_rag_index(candidate_id="candidate_001")
        print(
            f"documents={sync_result['documents']} chunks={sync_result['chunks']} "
            f"cached={sync_result['cached_embeddings']} new={sync_result['new_embeddings']}"
        )
        evidence_embed = load_indexed_chunks(
            EMBEDDING_MODEL, candidate_id="candidate_001"
        )
        query_text = "Do I have experience writing SQL queries and improving database performance?"
        search_results = dense_search(query_text, evidence_embed)
        print(f"Query: {query_text}")
        for rank, result in enumerate(search_results, start=1):
            print(
                f"rank={rank} score={result['score']:.4f} "
                f"chunk_id={result['chunk_id']} source_id={result['source_id']}\n"
                f"{result['text']}"
            )
    except HarnessError as e:
        print(f"[Runtime Error] {e}")
        return


if __name__ == "__main__":
    main()
