"""第2回 宿題（発展）: 用語集を検索する仕組みで、学習と同じ評価セットに答える。

使い方:
    uv run python -m scripts.eval_glossary_search

LLMは使わず、質問の用語（評価セットの query）で app.tools.search_glossary を呼び、
1件目の用語が正解の用語か、用語集に無い質問では「見つかりませんでした」になるかを確かめる。
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools import load_glossary, search_glossary

EVAL = Path(__file__).resolve().parents[1] / "lectures" / "02" / "glossary_eval.json"
OUT = EVAL.with_name("search_result.json")


def main() -> None:
    items = json.loads(EVAL.read_text(encoding="utf-8"))["items"]
    terms = {t["id"]: t for t in load_glossary()}
    rows = []
    started = time.perf_counter()
    for item in items:
        raw = search_glossary(item["query"])
        hits = [] if "見つかりませんでした" in raw else json.loads(raw)
        top = hits[0]["term"] if hits else None
        if item["group"] == "not_in_glossary":
            ok = not hits
        else:
            ok = top == terms[item["term_id"]]["term"]
        rows.append({"group": item["group"], "question": item["question"], "query": item["query"],
                     "top_term": top, "hits": [h["term"] for h in hits], "ok": ok})
    seconds = time.perf_counter() - started

    summary = {}
    for group in ("in_glossary", "reworded", "not_in_glossary"):
        group_rows = [r for r in rows if r["group"] == group]
        summary[group] = f"{sum(r['ok'] for r in group_rows)}/{len(group_rows)}"
    summary["all"] = f"{sum(r['ok'] for r in rows)}/{len(rows)}"
    summary["seconds"] = round(seconds, 3)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for r in rows:
        print(f"{'○' if r['ok'] else '×'} [{r['group']}] {r['question']} -> {r['hits'] or '見つからない'}")


if __name__ == "__main__":
    main()
