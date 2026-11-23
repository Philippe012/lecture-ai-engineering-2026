# 第3回 解答

## 演習

モデルはすべて `gemini-3.5-flash-lite`（既定モデル）。

- 第1回（素のLLM）とエージェントの回答の違い：
  - 質問「AIE2026の講座で使う「lecNN-start タグ」とは何ですか？」
  - 第1回の `app.ask`（ツールなし）は、「講義資料の中で講義の開始地点を示す目印」「Notebookの目次生成に使う」などと、**もっともらしいが誤った説明**を断定した（ハルシネーション）。講座に固有の言葉なので、モデルの知識には答えがない。
  - エージェントは `search_glossary` で用語集を引き、「各回の演習開始時点のコードに付けたGitタグ（例：lec03-start）。前の回が終わっていなくても、このタグからブランチを作れば演習を始められる」と**正しく**答え、関係する回（第1回）も示した。
- `--trace` で見えたツール呼び出し（ツール名と引数）：
  - lecNN-start：`search_glossary({'query': 'lecNN-start タグ'})` の1回
  - 「LoRAとQLoRAの違いは？」：`search_glossary({'query': 'LoRA'})` と `search_glossary({'query': 'QLoRA'})` の2回。docstring の「query には用語を1つだけ入れる」に従い、2つの用語を分けて検索した。2回の呼び出しは1回のモデル応答の中で同時に出ていた
- 1つの質問での LLM 呼び出し回数（`requests`）：
  - lecNN-start：requests=2（ツール呼び出し1回）、入力1,379／出力111 tokens、推定 $0.00069
  - LoRAとQLoRA：requests=2（ツール呼び出し2回）、入力1,792／出力183 tokens、推定 $0.00100
  - 1回目の応答でツール呼び出しを返し、2回目の応答で結果を読んで回答する、というループが見えた。ツールを2回呼んでも、同じ応答で並べて呼べば requests は増えなかった
- 追加したツールと、docstring に書いた説明：
  - `make_quiz(lecture: int)`（`app/tools.py`）。指定した回の用語から1つを選び、問題・答え（用語集の定義）を返す
  - docstring：「指定した回（1始まりの整数）の用語から1つを選び、確認問題と答えを返す。『第2回の内容から確認問題を出して』のように、復習用の問題を求められたときに使う。答えは用語集の定義なので、利用者が考える前に答えを見せないように出題する。」
  - 「第2回の内容から確認問題を1問出して」→ `make_quiz({'lecture': 2})` が1回呼ばれ、「QLoRA」が選ばれた。モデルは答えを書かずに問題とヒント（量子化とLoRAの組み合わせ、メモリの利点）だけを出した。requests=2、入力1,370／出力102 tokens
  - `uv run pytest -q`：すべて通過

## 宿題

- 取り組んだレベル：**実装**

- やったこと：
  - `app/tools.py` に `save_learning_log(title, content)` を作成。`learning_log/` に `日時_タイトル.md` として保存する（`learning_log/` は `.gitignore` 済みで、個人の記録はコミットされない）
  - `app/agent.py` で `agent.tool_plain(tools.save_learning_log, requires_approval=True)` として登録し、`output_type=[str, DeferredToolRequests]` にした
  - CLI に `run_with_approval` と `ask_approval` を追加。実行が承認待ち（`DeferredToolRequests`）で止まると、ツール名と引数（保存するタイトルと本文）を表示して `この内容で実行しますか？ [y/N]` と聞く。`y` なら `DeferredToolResults.approvals` に `True`、それ以外なら `ToolDenied("利用者が実行を許可しませんでした。保存はしていません。")` を入れ、`message_history` と一緒に渡して続きを実行する。入力が無い（EOF）場合も拒否として扱い、既定を「保存しない」にした
  - `INSTRUCTIONS` に「学習の記録を保存するよう頼まれたときだけ save_learning_log を使う」を追加
  - TestModel のテスト `tests/test_learning_log.py`（6件）：make_quiz が指定回の用語を返す／保存の中身／承認待ちで止まりファイルが作られない／承認すると保存される／拒否すると保存されない／入力が無いと保存されない。既存の `test_agent.py` は、TestModel が全ツールを呼ぶと保存が承認待ちになるため、「承認待ちの呼び出しを除いて、呼び出しと結果の数が一致する」に直した。`uv run pytest -q` で17件すべて通過

- 結果：質問「今日学んだLoRAとQLoRAの違いを、学習ログに保存してください。」を同じ条件で2回実行した

  | | 承認（y） | 拒否（n） |
  |---|---|---|
  | ツール呼び出し | search_glossary(LoRA)、search_glossary(QLoRA)、save_learning_log | 同じ3つ（save は拒否） |
  | save の結果 | `保存しました: learning_log/20261123-095109_LoRAとQLoRAの違い.md` | `利用者が実行を許可しませんでした。保存はしていません。` |
  | ファイル | 1件作成（用語集の定義に沿ったLoRA・QLoRAのまとめ） | 作成されない |
  | 最終回答 | 「学習ログとして保存しました」 | 「承認されなかったため保存はキャンセルされました。もう一度希望する場合は知らせてください」 |
  | requests / tool_calls | 3 / 3 | 3 / 2 |
  | 入力／出力 tokens・推定費用 | 3,205 / 231・$0.00154 | 3,174 / 239・$0.00155 |

  承認したときの `--trace`（抜粋）：
  ```
  [承認待ち] save_learning_log({"title": "LoRAとQLoRAの違い", "content": "## LoRAとQLoRAの違い\n\n- **LoRA**\n  - 事前学習済みの重みは固定し、…"})
  この内容で実行しますか？ [y/N]: [承認] 実行します
  [tool call]   search_glossary({'query': 'LoRA'})
  [tool call]   search_glossary({'query': 'QLoRA'})
  [tool call]   save_learning_log({'title': 'LoRAとQLoRAの違い', 'content': '## LoRAとQLoRAの違い\n\n- **LoRA**\n …'})
  [tool return] save_learning_log -> 保存しました: learning_log/20261123-095109_LoRAとQLoRAの違い.md…
  [usage] RunUsage(... input_tokens=3205, output_tokens=231, requests=3, tool_calls=3)
  ```
  拒否したときは `[tool return] save_learning_log -> 利用者が実行を許可しませんでした。保存はしていません。` となり、`learning_log/` に新しいファイルはできなかった。

- 気づき・疑問：
  - 承認の確認では、**モデルが書いた本文がそのまま見える**。何が書き込まれるかを見てから判断できるのが、取り消しにくい操作の前に確認を挟む意味だと分かった。
  - 拒否しても実行はエラーにならず、拒否の理由がツールの結果としてモデルに戻り、モデルが「保存はキャンセルされた」と利用者に説明した。`ToolDenied` のメッセージがそのまま回答の材料になるので、何をしなかったかを明確に書いておくのが大事。
  - 保存を頼んだだけでも、モデルは先に用語集を2回検索してから本文を作った（`INSTRUCTIONS` の「必ず search_glossary で確認する」が効いている）。根拠のある記録になる一方、requests が3回、入力トークンが約3,200に増えた。承認待ちで一度止まり、承認後に続きを実行するので、そのぶん呼び出しも増える。
  - 承認時と拒否時で tool_calls が 3 と 2 と違った。拒否した呼び出しは実行されたツールとして数えられないようだ。
  - モデルが本文の先頭にも見出し（`## LoRAとQLoRAの違い`）を入れたため、ツールが付ける `# タイトル` と見出しが重複した。ツール側で本文先頭の重複見出しを除くか、docstring に「content に見出しは入れない」と書くと良さそう。
  - 疑問：承認待ちの状態をファイルなどに保存しておけば、CLIを閉じた後で承認して続きを実行することもできるのか。Web画面で使う場合はどう実装するのか。
  - 実装とテストの作成にはAIツールを利用した。上の結果は、自分の環境で実行したログから転記した。
