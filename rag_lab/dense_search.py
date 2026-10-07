from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from chunking import EMBEDDING_MODEL

ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")

client = OpenAI()



def embedding_texts(
        texts: list[str],
) -> list[list[float]]:
    try:
        response = client.embeddings.create(
            model = EMBEDDING_MODEL,
            input=texts,
        )

        return {
            "embedding" : [item.embedding for item in response.data],
            "success": True
        }
    except Exception as e:
        return {
            "success": False,
            "error_msg": str(e)
        }

# if __name__ == "__main__":
    # print(embedding_texts(input()))