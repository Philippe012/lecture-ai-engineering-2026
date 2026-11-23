"""make_quiz と、承認が必要な save_learning_log のテスト（LLMもAPIキーも使わない）。"""

import json

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, run_with_approval


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    path = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LOG_DIR", path)
    return path


def _returns(result, name):
    return [p for m in result.all_messages() for p in m.parts
            if isinstance(p, ToolReturnPart) and p.tool_name == name]


def test_make_quiz_uses_terms_of_the_lecture():
    quiz = json.loads(tools.make_quiz(2))
    term = next(e for e in tools.load_glossary() if e["term"] == quiz["term"])
    assert term["lecture"] == 2
    assert quiz["answer"] == term["definition"]
    assert quiz["term"] in quiz["question"]
    assert "用語集にありません" in tools.make_quiz(99)


def test_save_learning_log_writes_markdown(log_dir):
    message = tools.save_learning_log("LoRA の復習", "元の重みは固定し、小さな行列だけを学習する。")
    files = list(log_dir.glob("*.md"))
    assert len(files) == 1 and files[0].name in message
    assert files[0].read_text(encoding="utf-8").startswith("# LoRA の復習")
    assert "両方を指定" in tools.save_learning_log("", "本文")


def test_save_stops_for_approval_without_writing(log_dir):
    with agent.override(model=TestModel(call_tools=["save_learning_log"])):
        result = agent.run_sync("今日の学びを保存して")
    assert isinstance(result.output, DeferredToolRequests)
    assert [c.tool_name for c in result.output.approvals] == ["save_learning_log"]
    assert not log_dir.exists()


def test_approved_save_writes_file(log_dir):
    model = TestModel(call_tools=["save_learning_log"])
    result = run_with_approval(agent, "今日の学びを保存して", model, answer=lambda _: "y")
    assert isinstance(result.output, str)
    assert len(list(log_dir.glob("*.md"))) == 1
    assert "保存しました" in _returns(result, "save_learning_log")[0].content


def test_denied_save_writes_nothing(log_dir):
    model = TestModel(call_tools=["save_learning_log"])
    result = run_with_approval(agent, "今日の学びを保存して", model, answer=lambda _: "n")
    assert isinstance(result.output, str)
    assert not log_dir.exists()
    assert "許可しませんでした" in str(_returns(result, "save_learning_log")[0].content)


def test_no_input_means_denied(log_dir):
    def closed(_):
        raise EOFError

    model = TestModel(call_tools=["save_learning_log"])
    result = run_with_approval(agent, "今日の学びを保存して", model, answer=closed)
    assert not log_dir.exists()
    assert isinstance(result.output, str)
