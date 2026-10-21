"""Lecture 1: ask a Gemini model questions from the command line.

Modes
-----
1. Single question (exercise / Basic level):
       uv run python -m app.ask "What is LoRA?"
       uv run python -m app.ask              # asks the default course-specific question

2. Chat with conversation history (Implementation level):
       uv run python -m app.ask --chat
   The full history is sent on every turn, so the model can answer follow-up
   questions. Commands inside the chat: /history, /reset, /exit

3. Benchmark response time and token counts (Advanced level):
       uv run python -m app.ask --bench
       uv run python -m app.ask --bench --models gemini-3.5-flash-lite gemini-3.1-flash-lite \
           --runs 3 --sleep 4.5 --csv lectures/01/bench_results.csv

The default model comes from the AIE_MODEL environment variable (see app/config.py).
The API key is read only from the GEMINI_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# Course-specific question. The correct answer is in data/glossary.json (homework-levels).
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# Benchmark defaults: questions that need a short, medium and long answer.
DEFAULT_BENCH_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
DEFAULT_BENCH_QUESTIONS = [
    "What is LoRA? Answer in one sentence.",
    "What is LoRA?",
    "Explain step by step how a Transformer language model generates text.",
]

# The free tier for gemini-3.5-flash-lite allows about 15 requests per minute.
# Waiting 4.5 s between requests keeps us at about 13 requests per minute.
DEFAULT_SLEEP_SECONDS = 4.5


# Shared helpers


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


def make_client() -> genai.Client:
    return genai.Client(api_key=get_api_key())


def read_usage(response: types.GenerateContentResponse) -> Usage:
    """Read token counts from response.usage_metadata (it can be missing)."""
    usage = response.usage_metadata
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count,
        total_tokens=usage.total_token_count,
    )


def format_usage(usage: Usage) -> str:
    return f"[tokens] input={usage.input_tokens}, output={usage.output_tokens}"


# Mode 1: single question (no tools, no history)


def ask(question: str, model: str = MODEL) -> str:
    """Ask one question with no tools and no history (the "raw" LLM)."""
    client = make_client()
    response = client.models.generate_content(model=model, contents=question)
    print(format_usage(read_usage(response)))
    return response.text or ""


# Mode 2: chat with conversation history (Implementation level)


def user_message(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def model_message(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


def print_history(history: list[types.Content]) -> None:
    if not history:
        print("[history is empty]\n")
        return
    for i, message in enumerate(history, start=1):
        text = "".join(part.text or "" for part in (message.parts or []))
        preview = text.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"  {i:>2}. {message.role:<5} {preview}")
    print()


def chat(model: str = MODEL) -> None:
    """Interactive chat that keeps the conversation history.

    The Gemini API is stateless: it does not remember earlier requests.
    To support follow-up questions, we keep every user and model message in
    `history` and send the whole list on each turn. (client.chats does the
    same thing internally; we do it by hand to make the mechanism visible.)
    Because the history grows, the input token count grows on every turn.
    """
    client = make_client()
    history: list[types.Content] = []
    turn = 0

    print(f"[model] {model}")
    print("Chat mode. Commands: /history, /reset, /exit\n")

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question in ("/exit", "/quit"):
            break
        if question == "/reset":
            history.clear()
            turn = 0
            print("[history cleared]\n")
            continue
        if question == "/history":
            print_history(history)
            continue

        history.append(user_message(question))
        start = time.perf_counter()
        try:
            response = client.models.generate_content(model=model, contents=history)
        except errors.APIError as e:
            # Remove the unanswered question so the history stays user/model/user/model.
            history.pop()
            print(api_error_message(e.code, e.message) + "\n")
            continue
        seconds = time.perf_counter() - start

        answer = response.text or ""
        history.append(model_message(answer))
        turn += 1

        print(f"\nmodel> {answer}\n")
        print(
            f"[turn {turn}] {format_usage(read_usage(response))}, "
            f"time={seconds:.2f}s, messages in history={len(history)}\n"
        )


# Mode 3: benchmark response time and tokens (Advanced level)


@dataclass
class BenchResult:
    model: str
    question_id: str
    question: str
    run: int
    status: str  # "ok", or the API error code such as "429" or "503"
    seconds: float
    input_tokens: int | None
    output_tokens: int | None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def tokens_per_second(self) -> float | None:
        if not self.ok or not self.output_tokens or self.seconds <= 0:
            return None
        return self.output_tokens / self.seconds


def run_once(
    client: genai.Client, model: str, question_id: str, question: str, run: int
) -> BenchResult:
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
    except errors.APIError as e:
        seconds = time.perf_counter() - start
        return BenchResult(model, question_id, question, run, str(e.code), seconds, None, None)
    seconds = time.perf_counter() - start
    usage = read_usage(response)
    return BenchResult(
        model, question_id, question, run, "ok", seconds, usage.input_tokens, usage.output_tokens
    )


def mean_or_none(values: list[float | int | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_summary(
    results: list[BenchResult], questions: list[tuple[str, str]], wall_seconds: float
) -> None:
    print("\nQuestions:")
    for question_id, question in questions:
        print(f"  {question_id}: {question}")

    # Table 1: per model and question (Markdown, ready to paste into answer.md)
    print("\n### Per model and question\n")
    print(
        "| Model | Question | OK / runs | Avg time (s) | Min-max time (s) "
        "| Avg input tokens | Avg output tokens | Avg output tokens/s |"
    )
    print("|---|---|---|---|---|---|---|---|")
    groups: dict[tuple[str, str], list[BenchResult]] = {}
    for r in results:
        groups.setdefault((r.model, r.question_id), []).append(r)
    for (model, question_id), group in groups.items():
        ok = [r for r in group if r.ok]
        times = [r.seconds for r in ok]
        time_range = f"{min(times):.2f}-{max(times):.2f}" if times else "-"
        print(
            f"| {model} | {question_id} | {len(ok)} / {len(group)} "
            f"| {fmt(mean_or_none(times))} | {time_range} "
            f"| {fmt(mean_or_none([r.input_tokens for r in ok]), 0)} "
            f"| {fmt(mean_or_none([r.output_tokens for r in ok]), 0)} "
            f"| {fmt(mean_or_none([r.tokens_per_second for r in ok]), 1)} |"
        )

    # Table 2: per model overall
    print("\n### Per model\n")
    print("| Model | Requests | OK | Errors | Avg time (s) | Avg output tokens/s |")
    print("|---|---|---|---|---|---|")
    for model in dict.fromkeys(r.model for r in results):
        group = [r for r in results if r.model == model]
        ok = [r for r in group if r.ok]
        error_counts = Counter(r.status for r in group if not r.ok)
        error_text = ", ".join(f"{code} x{n}" for code, n in error_counts.items()) or "0"
        print(
            f"| {model} | {len(group)} | {len(ok)} | {error_text} "
            f"| {fmt(mean_or_none([r.seconds for r in ok]))} "
            f"| {fmt(mean_or_none([r.tokens_per_second for r in ok]), 1)} |"
        )

    # Rate-limit information
    rate_limited = sum(1 for r in results if r.status == "429")
    overloaded = sum(1 for r in results if r.status == "503")
    per_minute = len(results) / (wall_seconds / 60) if wall_seconds > 0 else 0
    print(
        f"\nTotal: {len(results)} requests in {wall_seconds:.1f}s "
        f"(~{per_minute:.1f} requests/minute)"
    )
    print(f"Rate-limit errors (429): {rate_limited}, model overloaded (503): {overloaded}")


def write_csv(results: list[BenchResult], path: str) -> None:
    fields = [
        "model", "question_id", "question", "run", "status",
        "seconds", "input_tokens", "output_tokens", "tokens_per_second",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["seconds"] = round(r.seconds, 3)
            tps = r.tokens_per_second
            row["tokens_per_second"] = round(tps, 1) if tps is not None else ""
            writer.writerow(row)


def bench(
    models: list[str],
    questions: list[str],
    runs: int,
    sleep_seconds: float,
    csv_path: str | None,
) -> None:
    client = make_client()
    labeled = [(f"q{i}", q) for i, q in enumerate(questions, start=1)]
    total = len(models) * len(labeled) * runs

    print(
        f"[bench] {len(models)} model(s) x {len(labeled)} question(s) x {runs} run(s) "
        f"= {total} requests"
    )
    if sleep_seconds > 0:
        print(
            f"[bench] waiting {sleep_seconds:.1f}s between requests "
            f"(at most ~{60 / sleep_seconds:.0f} requests/minute)"
        )
    print()

    results: list[BenchResult] = []
    count = 0
    bench_start = time.perf_counter()
    for model in models:
        for question_id, question in labeled:
            for run in range(1, runs + 1):
                count += 1
                r = run_once(client, model, question_id, question, run)
                results.append(r)
                print(
                    f"[{count}/{total}] {model} | {question_id} | run {run} | {r.status} "
                    f"| {r.seconds:.2f}s | input={r.input_tokens} output={r.output_tokens}"
                )
                if count < total and sleep_seconds > 0:
                    time.sleep(sleep_seconds)
    wall_seconds = time.perf_counter() - bench_start

    print_summary(results, labeled, wall_seconds)
    if csv_path:
        write_csv(results, csv_path)
        print(f"\n[bench] saved raw results to {csv_path}")


# Command line


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a Gemini model questions (Lecture 1)")
    parser.add_argument(
        "question", nargs="?", default=DEFAULT_QUESTION, help="question for single mode"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--chat", action="store_true", help="interactive chat that keeps conversation history"
    )
    mode.add_argument(
        "--bench", action="store_true",
        help="measure response time and tokens across models and questions",
    )
    parser.add_argument(
        "--model", default=MODEL, help=f"model for single and chat mode (default: {MODEL})"
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_BENCH_MODELS, help="models to compare (bench)"
    )
    parser.add_argument(
        "--questions", nargs="+", default=DEFAULT_BENCH_QUESTIONS, help="questions to use (bench)"
    )
    parser.add_argument("--runs", type=int, default=3, help="runs per model and question (bench)")
    parser.add_argument(
        "--sleep", type=float, default=DEFAULT_SLEEP_SECONDS,
        help="seconds to wait between requests, to respect rate limits (bench)",
    )
    parser.add_argument("--csv", default=None, help="save raw results to this CSV file (bench)")
    args = parser.parse_args()

    if args.chat:
        chat(args.model)
        return
    if args.bench:
        bench(args.models, args.questions, args.runs, args.sleep, args.csv)
        return

    print(f"[model] {args.model}")
    print(f"[question] {args.question}\n")
    try:
        print(ask(args.question, args.model))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()