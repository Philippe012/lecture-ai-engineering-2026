"""第5回: ページ単位のベクトル検索。"""

import argparse
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path

from google import genai
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_SOURCE, RAG_TOP_K, RAG_TREE, get_api_key,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS,
    HYBRID_CANDIDATE_K, HYBRID_MAX_PAGES, HYBRID_PER_SECTION, HYBRID_MORE, HYBRID_RRF_K,
)


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )

def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update({key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("埋め込みがゼロベクトルです。")
    return [value / norm for value in values]


def embed_text(text: str, task_type: str) -> list[float]:
    """Geminiで文章を埋め込む。入力上限を超えた場合は切り捨てず停止する。"""
    with genai.Client(api_key=get_api_key()) as client:
        count = client.models.count_tokens(model=EMBEDDING_MODEL, contents=text).total_tokens
        if count is None or count > EMBEDDING_MAX_TOKENS:
            raise ValueError(f"埋め込み入力が上限を超えるか計測不能です: {count} tokens。")
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
        )
    embedding = response.embeddings[0]
    if embedding.statistics and embedding.statistics.truncated:
        raise ValueError("埋め込みAPIが本文を切り捨てました。")
    return normalized_vector(embedding.values)


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("検索DBがありません。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def search(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """質問を埋め込み、近いページを最大top_k件返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with open_index(RAG_DB) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, document_id, lecture, document_title, section, title, page, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), top_k))
        return [dict(row) for row in rows]


def tokenize(text: str) -> list[str]:
    """英数字は単語、それ以外（日本語など）は2文字ずつに区切る。"""
    text = text.casefold()
    tokens = re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text)
    for run in re.findall(r"[^\x00-\x7f\s、。，．・：:；「」『』（）()【】［］\[\]／/！？!?〜~ー\-]+", text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class LexicalIndex:
    """ページ本文のBM25索引。埋め込みAPIを使わず、表記が一致する語で探す。"""

    def __init__(self, pages: list[dict], k1: float = 1.2, b: float = 0.75):
        self.pages = pages
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{p['title']}\n{p['content']}")) for p in pages]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average = sum(self.lengths) / max(len(self.docs), 1)
        frequency = Counter(token for doc in self.docs for token in doc)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        query_tokens = set(tokenize(query))
        scored = []
        for page, doc, length in zip(self.pages, self.docs, self.lengths):
            score = 0.0
            for token in query_tokens:
                tf = doc.get(token, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * length / self.average)
                    score += self.idf[token] * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scored.append((page, score))
        scored.sort(key=lambda item: (-item[1], item[0]["page_id"]))
        return scored[:top_k]


@lru_cache(maxsize=1)
def lexical_index() -> LexicalIndex:
    return LexicalIndex(load_pages(RAG_SOURCE))


@lru_cache(maxsize=1)
def load_terms() -> tuple[dict, dict]:
    """用語ノードと、ページIDごとの言及用語を返す。"""
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    terms = {n["id"]: n for n in graph["nodes"] if n["kind"] == "term"}
    mentions = defaultdict(set)
    for edge in graph["edges"]:
        if edge["relation"] == "mentions" and edge["target"] in terms:
            mentions[edge["source"]].add(edge["target"])
    return terms, mentions


def query_lectures(query: str) -> set[int]:
    """「第3回」「lecture 2」のような回の指定を取り出す。対象外の回は無視する。"""
    available = {int(doc_id[3:5]) for doc_id in RAG_DOCUMENT_IDS}
    found = {int(n) for n in re.findall(r"第\s*([0-9０-９]+)\s*回", query.translate(str.maketrans("０１２３４５６７８９", "0123456789")))}
    found |= {int(n) for n in re.findall(r"(?:lecture|session|lec)\s*0?([0-9]+)", query, re.IGNORECASE)}
    return found & available


def query_terms(query: str) -> set[str]:
    terms, _ = load_terms()
    text = query.casefold()
    return {tid for tid, node in terms.items()
            if any(len(alias) >= 2 and alias.casefold() in text for alias in node["aliases"] + [node["title"]])}


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[。！？])|\n", text.strip(), maxsplit=1)[0][:80]


def search_hybrid(query: str, vector_hits: list[dict] | None = None) -> dict:
    """ベクトル検索とBM25を順位で統合し、章ごとに偏らないよう本文を選び、残りは見出しで返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    pages = {p["page_id"]: p for p in load_pages(RAG_SOURCE)}
    if vector_hits is None:
        vector_hits = search(query, HYBRID_CANDIDATE_K)
    lexical_hits = lexical_index().search(query, HYBRID_CANDIDATE_K)
    lectures = query_lectures(query)
    anchors = query_terms(query)
    _, mentions = load_terms()

    scores, sources = defaultdict(float), defaultdict(list)
    for rank, hit in enumerate(vector_hits, 1):
        scores[hit["page_id"]] += 1 / (HYBRID_RRF_K + rank)
        sources[hit["page_id"]].append(f"vector#{rank}")
    for rank, (page, _) in enumerate(lexical_hits, 1):
        scores[page["page_id"]] += 1 / (HYBRID_RRF_K + rank)
        sources[page["page_id"]].append(f"bm25#{rank}")

    for page_id in list(scores):
        number = int(page_id.rsplit("-p", 1)[1])
        prefix = page_id.rsplit("-p", 1)[0]
        for neighbor in (f"{prefix}-p{number - 1:03d}", f"{prefix}-p{number + 1:03d}"):
            if neighbor in pages and neighbor not in scores and anchors & mentions[neighbor]:
                scores[neighbor] = scores[page_id] / 2
                sources[neighbor].append(f"neighbor:{page_id}")

    ranked = sorted((pid for pid in scores if pid in pages and (not lectures or pages[pid]["lecture"] in lectures)),
                    key=lambda pid: (-scores[pid], pid))

    selected, per_section = [], Counter()
    for page_id in ranked:
        key = (pages[page_id]["document_id"], pages[page_id]["section"])
        if per_section[key] < HYBRID_PER_SECTION:
            selected.append(page_id)
            per_section[key] += 1
        if len(selected) == HYBRID_MAX_PAGES:
            break
    for page_id in ranked:
        if len(selected) == HYBRID_MAX_PAGES:
            break
        if page_id not in selected:
            selected.append(page_id)

    def cite(page_id: str) -> dict:
        page = pages[page_id]
        return {"page_id": page_id, "lecture": page["lecture"], "page": page["page"],
                "path": f"{page['document_title']} > {page['section']} > {page['title']}",
                "found_by": sources[page_id]}

    rest = [pid for pid in ranked if pid not in selected][:HYBRID_MORE]
    return {
        "lectures": sorted(lectures),
        "query_terms": sorted(anchors),
        "pages": [{**cite(pid), "content": pages[pid]["content"]} for pid in selected],
        "more": [{**cite(pid), "first_sentence": first_sentence(pages[pid]["content"])} for pid in rest],
    }


def search_graph(query: str) -> dict:
    """ベクトル検索を入口に、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    seeds = [hit for hit in search(query) if hit["page_id"] in pages][:min(GRAPH_START_K, GRAPH_MAX_RESULTS)]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], "distance": seed["distance"], "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current], key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max((len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    return {"seeds": results, "expanded": expanded[:max(0, GRAPH_MAX_RESULTS - len(results))],
            "max_hops": GRAPH_HOPS}



def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    args = parser.parse_args()
    for i, result in enumerate(search(args.query), 1):
        print(f"\n{'=' * 60}\n検索結果 {i}\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        print(f"\n{result['content']}")
