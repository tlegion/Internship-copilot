from pathlib import Path
import sys

# Make shared modules available when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from errors import HarnessError
from rag_lab.chunking import EMBEDDING_MODEL
from rag_lab.dense_search import dense_search
from rag_lab.indexing import sync_rag_index
from storages.database import load_indexed_chunks
from rag_lab.bm25_search import bm25_search
from rag_lab.hybrid_search import hybrid_search


def main():
    try:
        sync_result = sync_rag_index()
        query_text = (
            "Did I use a technique to keep chance effects aligned "
            "when comparing alternative server setups?"
        )

        results = hybrid_search(
            query_text,
            candidate_id="candidate_001",
            top_k=3,
        )
        print(f"\nQuery: {query_text}")
        print("\n=== Hybrid ===")

        for rank, result in enumerate(results, start=1):
            print(
                f"rank={rank} "
                f"rrf={result['rrf_score']:.6f} "
                f"dense_rank={result['dense_rank']} "
                f"bm25_rank={result['bm25_rank']} "
                f"chunk_id={result['chunk_id']} "
                f"source_id={result['source_id']}\n"
                f"{result['text']}"
            )
    except HarnessError as e:
        print(f"[Runtime Error] {e}")
        return


if __name__ == "__main__":
    main()
