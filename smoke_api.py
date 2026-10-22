"""Send one short Responses API request; run with .venv/bin/python smoke_api.py."""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise SystemExit("请在项目 .env 中配置 OPENAI_API_KEY，然后重新运行。")

    model = os.environ.get("OPENAI_MODEL") or "gpt-6-luna"
    # Disable automatic retries so this smoke check makes one request.
    with OpenAI(base_url="https://api.openai.com/v1", timeout=30.0, max_retries=0) as client:
        try:
            response = client.responses.create(
                model=model,
                input="请用中文简短回复：API 已接通。",
                reasoning={"effort": "none"},
                max_output_tokens=128,
                store=False,
            )
        except APIStatusError as error:
            # Print only status/code; never print credentials or request headers.
            raise SystemExit(f"API 请求失败：HTTP {error.status_code}，code={error.code}") from None
        except APIConnectionError:
            raise SystemExit("API 连接失败或超时，请检查网络。") from None

    print(f"model: {response.model}")
    print(f"status: {response.status}")
    print(f"reply: {response.output_text}")
    if response.usage:
        print(f"tokens: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
    if response.status != "completed" or not response.output_text.strip():
        raise SystemExit("请求已返回，但没有完整文本回复；首次请求尚未验收通过。")


if __name__ == "__main__":
    main()
