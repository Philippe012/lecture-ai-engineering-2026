"""ハイブリッド検索のテスト（埋め込みAPIを使わない）。"""

import json

from app import retrieval, tools
from app.config import HYBRID_MAX_PAGES, HYBRID_PER_SECTION

MEMORY_QUESTION = "第3回の例で、コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか。"


def test_tokenize_mixes_words_and_bigrams():
    assert retrieval.tokenize("LoRAと QLoRA") == ["lora", "qlora", "と"]
    assert retrieval.tokenize("長期メモリ") == ["長期", "期メ", "メモ", "モリ"]


def test_query_lectures_and_terms():
    assert retrieval.query_lectures(MEMORY_QUESTION) == {3}
    assert retrieval.query_lectures("第９回の内容") == set()
    assert "term:qlora" in retrieval.query_terms("LoRAとQLoRAの違い")


def test_lexical_only_finds_memory_examples():
    result = retrieval.search_hybrid(MEMORY_QUESTION, vector_hits=[])
    ids = [p["page_id"] for p in result["pages"]]
    assert {"lec03-main-p039", "lec03-main-p040"} <= set(ids)
    assert all(p["lecture"] == 3 for p in result["pages"] + result["more"])


def test_pages_are_unique_and_sections_are_spread():
    page = retrieval.read_page("lec03-main-p013")
    vector_hits = [{**page, "distance": 0.1}]
    result = retrieval.search_hybrid(MEMORY_QUESTION, vector_hits=vector_hits)
    ids = [p["page_id"] for p in result["pages"]] + [p["page_id"] for p in result["more"]]
    assert len(ids) == len(set(ids))
    assert "lec03-main-p013" in ids[:HYBRID_MAX_PAGES]
    sections = [p["path"].rsplit(" > ", 1)[0] for p in result["pages"]]
    assert max(sections.count(s) for s in sections) <= HYBRID_PER_SECTION


def test_read_page_tool_returns_error_for_unknown_id():
    assert "error" in json.loads(tools.read_page("lec09-main-p001"))
    assert json.loads(tools.read_page("lec03-main-p040"))["page"] == 40
