# 第1回 解答

## 演習

- 講座に固有の質問（宿題の3レベル）へのモデルの回答（要約でよい）：
  `gemini-3.5-flash-lite` は「AIE2026の講座の資料にはアクセスできないので断定できない」と前置きしたうえで、一般論として3つの解釈を提示した：
  (1) 生成AI活用の習熟度（基礎・応用・発展）、(2) 思考の深さ（知識・応用・創造）、(3) 難易度別の選択式課題（初級・中級・上級）。
  最後に「講師やSlackで確認してください」と勧めていた。
- 用語集の正しい意味と比べて気づいたこと：
  正しくは「基礎（講義内容の確認）」「実装（自分でコードを書く）」「発展（調査や試行）」の3段階で、どれか1つに取り組めば完了、原則として採点しない（`data/glossary.json` の `homework-levels`）。
  モデルは「知らない」と認めた点は良かったが、そのあとに一般論でもっともらしい答えを3つも出した。(3) は名前が少し近いが、実際の3レベルは難易度ではなく「取り組み方の種類」の違いであり、「1つで完了」「採点しない」という重要な点はどれにも含まれていなかった。
  素のLLMは講座に固有の知識を持たないため、外部の情報（用語集の検索など）を与える必要があると分かった。
- 表示できたトークン数（入力 / 出力）：
  `app/ask.py` の TODO を改修し、`response.usage_metadata` から表示（`None` の場合も考慮）。
  - 「3レベル」の質問：入力 21 / 出力 651（`gemini-3.5-flash-lite`）
  - 「What is LoRA?」：入力 6 / 出力 675〜795（`gemini-3.5-flash-lite` / `gemini-3.1-flash-lite`）

## 宿題

- 取り組んだレベル：基礎
- やったこと：
  同じ質問「What is LoRA?」を、`AIE_MODEL` でモデルを変えて3回ずつ実行した（temperature は既定のまま）。
  - `gemini-3.5-flash-lite`（既定）
  - `gemini-3.1-flash-lite`
- 結果：

  | Model | Run 1 | Run 2 | Run 3 | 平均出力トークン |
  |---|---|---|---|---|
  | gemini-3.5-flash-lite | 739 | 675 | 695 | 約 703 |
  | gemini-3.1-flash-lite | 767 | 764 | 795 | 約 775 |

  入力トークンはすべて 6。
  - 同じモデル・同じ質問でも、毎回回答の文章が違った。ただし構成（課題 → 仕組み → 利点 → 用途）と「教科書に付箋を貼る」というたとえは毎回同じだった。
  - 細部は実行ごとに変わった（例：3.5-flash-lite では「Microsoft・2021年」に触れたのは1回目だけ、「学習パラメータ99%減・メモリ3倍減」という数字は2回目だけ）。
  - `gemini-3.1-flash-lite` は回答が長く（出力が約10%多い）、毎回「Summary」で締め、長さも安定していた（764〜795 vs 675〜739）。低ランク分解の具体的な説明（1000×1000 → 1000×r と r×1000）があったのは 3.1-flash-lite の3回目だけ。
- 気づき・疑問：
  - 一般的な用語なので大筋は正しいが、小さな誤りを自信のある口調で書いていた。
    - 3.1-flash-lite 1回目：「処理するデータ量が1000分の1になる」→ 実際に減るのは学習パラメータ数で、データ量ではない。
    - 3.1-flash-lite 2回目：LoRA ファイルが「数MB〜数GB」と書いており、一貫していない。
    - 3.5-flash-lite 3回目：「データは LoRA 層だけを通る」→ 実際は凍結した重みと LoRA 層の両方を通り、出力を足し合わせる。
    - 両モデルとも「catastrophic forgetting を完全に防ぐ」と言い切っており、言い過ぎ。
  - LLM の回答は実行ごとに揺れるので、確認せずに信用してはいけないと感じた。
  - 疑問：temperature を下げれば揺れや誤りは減るのか？（Google が temperature を非推奨にしたため、今回は試していない）
  - 繰り返し実験には上限が大きい `gemini-3.5-flash-lite` が向いている。

  **Extra: gemini-3.8-flash (stronger model, ~5 requests/minute, 10–20/day)**
3 questions × 2 runs, 13 s pause. (An earlier identical run was lost when the Codespace restarted, so this is my second run.)

| Question | Status | Time (s) | Output tokens | Output tokens/s |
|---|---|---|---|---|
| q1 run 1 | ok | 6.59 | 49 | 7.4 |
| q1 run 2 | 503 | 2.10 | - | - |
| q2 run 1 | 503 | 4.20 | - | - |
| q2 run 2 | ok | 18.89 | 790 | 41.8 |
| q3 run 1 | 503 | 5.03 | - | - |
| q3 run 2 | ok | 58.57 | 1116 | 19.1 |

Total: 6 requests in 160.4 s, 3 ok, 3 failed with 503 (model overloaded), 0 rate-limit errors (429).

- **Reliability:** 50% of requests failed with 503. This is server overload, not my rate limit
  (I stayed at ~2.2 requests/minute, under the ~5/minute limit). The 503 errors returned fast (2–5 s).
  In real use, the code needs automatic retry with backoff, or a fallback to a lighter model.
- **Speed:** successful runs were 7–12x slower than gemini-3.5-flash-lite on the same questions
  (q2: 18.9 s vs 3.0 s, q3: 58.6 s vs 4.8 s), at only 7–42 output tokens/s vs ~220–257.
  Part of this may be hidden "thinking" time. My code records only `candidates_token_count`,
  not `thoughts_token_count`, so the real number of generated tokens may be higher.
  Recording thinking tokens is a next step.
- **Quota:** these runs used 12 of the ~10–20 daily requests for this model. The stronger model
  is only practical for a few targeted tests. For repeated experiments, gemini-3.5-flash-lite
  was the clear winner: fastest, most stable, and no errors.