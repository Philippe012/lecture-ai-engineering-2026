"""基本RAG（ベクトル検索 上位RAG_TOP_K件）とハイブリッド検索で、根拠ページの取得を比べる。

使い方:
    uv run python -m scripts.compare_hybrid
"""

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import RAG_TOP_K
from app.retrieval import search, search_hybrid

QUESTIONS = [
    ("第3回の例で、コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか。",
     {"lec03-main-p013", "lec03-main-p039", "lec03-main-p040"}),
    ("第2回の説明で、LoRAは何を固定し何を学習しますか。QLoRAは何を加えますか。",
     {"lec02-main-p052"}),
    ("第3回で、エージェントのループを止める条件と、ツールの失敗時の振る舞いはどう説明されていますか。",
     set()),
    ("今回作ったRAGの実測の平均応答時間と正答率を教えてください。",
     set()),
]


def label(page_id: str, gold: set[str]) -> str:
    lecture, page = page_id.split("-main-p")
    return f"{lecture}p{int(page)}{'*' if page_id in gold else ''}"


def main() -> None:
    for question, gold in QUESTIONS:
        print(f"\n{'=' * 70}\n{question}\n根拠ページ: {sorted(gold) or '（指定なし）'}")

        started = perf_counter()
        basic = search(question, RAG_TOP_K)
        basic_time = perf_counter() - started
        basic_ids = [hit["page_id"] for hit in basic]
        basic_chars = sum(len(hit["content"]) for hit in basic)

        started = perf_counter()
        hybrid = search_hybrid(question, vector_hits=search(question, 10))
        hybrid_time = perf_counter() - started
        hybrid_ids = [p["page_id"] for p in hybrid["pages"]]
        more_ids = [p["page_id"] for p in hybrid["more"]]
        hybrid_chars = sum(len(p["content"]) for p in hybrid["pages"])

        def recall(ids):
            return f"{len(gold & set(ids))}/{len(gold)}" if gold else "-"

        print(f"[basic ] {recall(basic_ids)} 本文{basic_chars:,}字 {basic_time:.2f}s  "
              + " ".join(label(i, gold) for i in basic_ids))
        print(f"[hybrid] {recall(hybrid_ids)} 本文{hybrid_chars:,}字 {hybrid_time:.2f}s  "
              + " ".join(label(i, gold) for i in hybrid_ids)
              + f"  | more: {' '.join(label(i, gold) for i in more_ids)} (more込み {recall(hybrid_ids + more_ids)})")
        print(f"         回の指定={hybrid['lectures']} 用語={hybrid['query_terms']}")
        for p in hybrid["pages"]:
            print(f"         {label(p['page_id'], gold):>9} {p['found_by']} {p['path'][:50]}")


if __name__ == "__main__":
    main()
