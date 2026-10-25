"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールは使わず、会話履歴を Markdown ファイルに保存して次回の質問に利用する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import json
import os
import re
import secrets
from pathlib import Path

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, TEMPERATURE, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "01"
COMPACTION_THRESHOLD = int(1_048_576 / 10.0)


def get_history_file() -> tuple[str, Path]:
    topic_hash = os.getenv("AIE_TOPIC_HASH", "").strip()
    if topic_hash and re.fullmatch(r"[0-9a-fA-F]+", topic_hash):
        history_file = HISTORY_DIR / f"{topic_hash}.md"
        if history_file.is_file():
            return topic_hash, history_file

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        topic_hash = secrets.token_hex(16)
        history_file = HISTORY_DIR / f"{topic_hash}.md"
        try:
            with history_file.open("x", encoding="utf-8") as file:
                file.write("# 会話履歴\n")
            return topic_hash, history_file
        except FileExistsError:
            continue


def compaction(client: genai.Client, history_file: Path) -> None:
    history = history_file.read_text(encoding="utf-8")
    response = client.models.generate_content(
        model=MODEL,
        contents=json.dumps(
            {
                "query": "会話履歴を、今後の質問への回答に必要な情報を保って簡潔に要約してください。",
                "history": history,
            },
            ensure_ascii=False,
        ),
        config=types.GenerateContentConfig(
            temperature=0,
            seed=42,
        ),
    )
    if not response.text:
        raise RuntimeError("会話履歴の要約を取得できませんでした。")
    history_file.write_text(f"# 会話要約\n\n{response.text}\n", encoding="utf-8")


def ask(question: str) -> str:
    topic_hash, history_file = get_history_file()
    history = history_file.read_text(encoding="utf-8")
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=json.dumps(
            {
                "query": "historyを踏まえ、questionに回答してください。",
                "question": question,
                "history": history,
            },
            ensure_ascii=False,
        ),
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            seed=42,
        ),
    )

    usage = response.usage_metadata
    if usage is None:
        print("[トークン数] 入力: -, 出力: -")
    else:
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        print(
            f"[トークン数] 入力: {input_tokens if input_tokens is not None else '-'}, "
            f"出力: {output_tokens if output_tokens is not None else '-'}"
        )

    if response.text is None:
        raise RuntimeError("モデルから回答テキストを取得できませんでした。")
    with history_file.open("a", encoding="utf-8") as file:
        file.write(f"\n## 質問\n\n{question}\n\n## 回答\n\n{response.text}\n")
    if len(history_file.read_text(encoding="utf-8")) > COMPACTION_THRESHOLD:
        compaction(client, history_file)

    print(f"[topic_hash] {topic_hash}")
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[temperature] {TEMPERATURE}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
