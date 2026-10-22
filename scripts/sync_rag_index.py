from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_lab.indexing import sync_rag_index


if __name__ == "__main__":
    result = sync_rag_index()
    print(result)