from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_lab.job_indexing import sync_job_index

if __name__ == "__main__":
    print(sync_job_index())
