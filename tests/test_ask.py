import json
from types import SimpleNamespace

from app import ask


def test_ask_displays_token_counts_and_creates_random_history(monkeypatch, capsys, tmp_path):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
        text="回答",
    )
    generate_content_args = {}

    monkeypatch.setattr(ask, "HISTORY_DIR", tmp_path)
    monkeypatch.delenv("AIE_TOPIC_HASH", raising=False)

    def generate_content(**kwargs):
        generate_content_args.update(kwargs)
        return response

    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
    )

    assert ask.ask("質問") == "回答"
    output = capsys.readouterr().out
    assert "[トークン数] 入力: 12, 出力: 7" in output
    assert "[topic_hash] " in output
    assert generate_content_args["config"].temperature == ask.TEMPERATURE
    prompt = json.loads(generate_content_args["contents"])
    assert prompt["query"] == "historyを踏まえ、questionに回答してください。"
    assert prompt["question"] == "質問"
    assert prompt["history"] == "# 会話履歴\n"
    history_files = list(tmp_path.glob("*.md"))
    assert len(history_files) == 1
    assert "## 質問\n\n質問\n\n## 回答\n\n回答" in history_files[0].read_text()


def test_ask_uses_existing_history(monkeypatch, tmp_path):
    topic_hash = "abc123"
    history_file = tmp_path / f"{topic_hash}.md"
    history_file.write_text("# 会話履歴\n\n## 質問\n\n前の質問\n\n## 回答\n\n前の回答\n")
    monkeypatch.setattr(ask, "HISTORY_DIR", tmp_path)
    monkeypatch.setenv("AIE_TOPIC_HASH", topic_hash)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    generate_content_args = {}
    response = SimpleNamespace(
        usage_metadata=None,
        text="続きの回答",
    )

    def generate_content(**kwargs):
        generate_content_args.update(kwargs)
        return response

    monkeypatch.setattr(
        ask.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
    )

    assert ask.ask("次の質問") == "続きの回答"
    prompt = json.loads(generate_content_args["contents"])
    assert "前の質問" in prompt["history"]
    assert "前の回答" in prompt["history"]
    assert "## 質問\n\n次の質問\n\n## 回答\n\n続きの回答" in history_file.read_text()


def test_ask_compacts_history_after_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(ask, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(ask, "COMPACTION_THRESHOLD", 1)
    monkeypatch.delenv("AIE_TOPIC_HASH", raising=False)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    prompts = []
    responses = iter(
        [
            SimpleNamespace(
                usage_metadata=None,
                text="回答",
            ),
            SimpleNamespace(text="会話の要約"),
        ]
    )

    def generate_content(**kwargs):
        prompts.append(json.loads(kwargs["contents"]))
        return next(responses)

    monkeypatch.setattr(
        ask.genai,
        "Client",
        lambda **kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
    )

    assert ask.ask("質問") == "回答"
    assert len(prompts) == 2
    assert "質問" in prompts[1]["history"]
    assert "回答" in prompts[1]["history"]
    history_file = next(tmp_path.glob("*.md"))
    assert history_file.read_text() == "# 会話要約\n\n会話の要約\n"


def test_ask_creates_new_history_for_unknown_topic_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(ask, "HISTORY_DIR", tmp_path)
    monkeypatch.setenv("AIE_TOPIC_HASH", "unknown-topic")
    topic_hash, history_file = ask.get_history_file()

    assert topic_hash != "unknown-topic"
    assert history_file.name == f"{topic_hash}.md"
    assert history_file.is_file()