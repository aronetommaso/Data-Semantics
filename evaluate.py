"""
evaluate.py — Evaluation of the ACLED Router pipeline
====================================================================
For each question in evaluation_dataset.json, runs the full pipeline
and measures:
  1. Router accuracy   — did the router pick the right pipeline?
  2. Answer quality    — LLM-as-judge (relevance + faithfulness, 0-10)
                         or exact match for questions with a known answer

Usage:
    python evaluate.py
    python evaluate.py --dataset path/to/custom_dataset.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
import re
import time
from dotenv import load_dotenv
from google import genai

from router import load_sparql_module, route_question
from graphrag_terminal import GraphRAGAnswerer, GraphRAGConfig

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "evaluation_dataset.json"
DEFAULT_RESULTS = SCRIPT_DIR / "evaluation_results.json"
JUDGE_MODEL = "gemini-2.5-flash-lite"
MAX_RETRIES = 6
BASE_RETRY_DELAY = 20.0
DEFAULT_QUESTION_RETRIES = 12
TRANSIENT_ERROR_PATTERNS = (
    "429",
    "503",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "high demand",
    "temporarily unavailable",
)

env_path = SCRIPT_DIR / ".env"
if not env_path.exists():
    env_path = SCRIPT_DIR.parent / ".env"
load_dotenv(dotenv_path=env_path)

JUDGE_SYSTEM = """
You are an expert evaluator for a Q&A system about ACLED Middle East conflict data.
Given a question and an answer, rate the answer on two dimensions from 0 to 10:
  - relevance:    does the answer actually address the question asked?
  - faithfulness: is the answer grounded in real data, without hallucinations?

Return ONLY a JSON object with exactly these keys:
{"relevance": <int 0-10>, "faithfulness": <int 0-10>, "rationale": "<one sentence>"}
"""

# ──────────────────────────────────────────────────────────────────────────────
# 2. LLM-AS-JUDGE
# ──────────────────────────────────────────────────────────────────────────────

def require_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    return api_key


def setup_gemini():
    return genai.Client(api_key=require_gemini_api_key())


def generate_with_backoff(client, prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            )
            if not response.text:
                raise RuntimeError("Gemini returned an empty response")
            return response.text
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait_seconds = BASE_RETRY_DELAY * (2 ** attempt)
            if is_transient_error(exc):
                wait_seconds = max(wait_seconds, 70.0)
            print(
                f"  Judge error, waiting {wait_seconds:.0f}s... "
                f"(attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait_seconds)
    raise RuntimeError("Max retries exceeded")


def _parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def llm_judge(client, question: str, answer: str) -> dict:
    prompt = (
        f"{JUDGE_SYSTEM.strip()}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}"
    )
    return _parse_json_object(generate_with_backoff(client, prompt))


def is_transient_error(exc: Exception) -> bool:
    message = str(exc)
    return any(pattern in message for pattern in TRANSIENT_ERROR_PATTERNS)


def retry_delay(attempt: int) -> float:
    return min(30.0 * (2 ** max(attempt - 1, 0)), 300.0)


def _normalize_number(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"(?<=\d),(?=\d{3}\b)", "", normalized)
    normalized = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", normalized)
    return normalized


def _is_numeric_expected(value: str) -> bool:
    return re.fullmatch(r"\d+(?:[.,]\d+)?", value.strip()) is not None


def exact_match(answer: str, expected_value: str) -> bool:
    expected_text = str(expected_value).strip()
    expected_normalized = _normalize_number(expected_text)
    normalized_answer = _normalize_number(answer)

    if _is_numeric_expected(expected_text):
        expected_number = expected_normalized.replace(",", ".")
        answer_numbers = re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", normalized_answer)
        return expected_number in answer_numbers

    expected = re.escape(re.sub(r"\s+", " ", expected_normalized))
    normalized_answer = re.sub(r"\s+", " ", normalized_answer)
    return re.search(rf"(?<!\w){expected}(?!\w)", normalized_answer) is not None


def write_results(results_path: Path, results: list[dict]) -> None:
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_existing_results(results_path: Path) -> dict[str, dict]:
    if not results_path.exists():
        return {}
    with results_path.open("r", encoding="utf-8") as f:
        existing = json.load(f)
    if not isinstance(existing, list):
        return {}
    return {
        str(item["id"]): item
        for item in existing
        if isinstance(item, dict) and "id" in item
    }


def ordered_results(dataset: list[dict], results_by_id: dict[str, dict]) -> list[dict]:
    ordered = []
    seen = set()
    for item in dataset:
        qid = str(item["id"])
        if qid in results_by_id:
            ordered.append(results_by_id[qid])
            seen.add(qid)
    ordered.extend(result for qid, result in results_by_id.items() if qid not in seen)
    return ordered


# ──────────────────────────────────────────────────────────────────────────────
# 3. EVALUATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    dataset_path: Path,
    results_path: Path,
    sleep_seconds: float,
    graphrag_llm_provider: str,
    max_question_retries: int,
    retry_until_success: bool,
    only_route: str,
) -> None:
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("Loading pipelines...")
    judge_client = setup_gemini()
    sparql_module = load_sparql_module()
    graphrag_answerer = GraphRAGAnswerer(
        GraphRAGConfig(llm_provider=graphrag_llm_provider)
    )

    results_by_id = load_existing_results(results_path) if only_route != "all" else {}
    router_correct = 0
    router_total = 0
    answer_scores = []

    print("\n" + "=" * 70)
    print("  ACLED Pipeline Evaluation")
    print("=" * 70)

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        expected_route = item.get("expected_route")
        answer_type = item.get("answer_type", "llm_judge")

        if only_route != "all" and expected_route != only_route:
            continue

        print(f"\n[{qid}] {question}")

        payload = None
        attempt = 0
        while retry_until_success or attempt < max_question_retries:
            attempt += 1
            try:
                payload = route_question(question, graphrag_answerer, sparql_module)
            except Exception as exc:
                if is_transient_error(exc) and (retry_until_success or attempt < max_question_retries):
                    wait_seconds = retry_delay(attempt)
                    max_label = "∞" if retry_until_success else str(max_question_retries)
                    print(
                        f"  Transient provider error, waiting {wait_seconds:.0f}s... "
                        f"(attempt {attempt}/{max_label})"
                    )
                    time.sleep(wait_seconds)
                    continue
                print(f"  ERROR: {exc}")
                results_by_id[qid] = {
                    "id": qid,
                    "question": question,
                    "expected_route": expected_route,
                    "error": str(exc),
                }
                write_results(results_path, ordered_results(dataset, results_by_id))
                break

            # GraphRAG catches 429 internally and returns it as a status field
            if "rate_limit" in payload.get("status", ""):
                if retry_until_success or attempt < max_question_retries:
                    wait_seconds = retry_delay(attempt)
                    max_label = "∞" if retry_until_success else str(max_question_retries)
                    print(
                        f"  Rate limited in pipeline, waiting {wait_seconds:.0f}s... "
                        f"(attempt {attempt}/{max_label})"
                    )
                    time.sleep(wait_seconds)
                    continue
            break
        if payload is None:
            continue

        actual_route = payload["route"]
        answer = payload["answer"]

        # ── Router check ──────────────────────────────────────────────────────
        route_correct = None
        if expected_route:
            route_correct = actual_route == expected_route
            router_total += 1
            if route_correct:
                router_correct += 1
            status = "✓" if route_correct else f"✗ (expected {expected_route})"
            print(f"  Route:  {actual_route}  {status}")
        else:
            print(f"  Route:  {actual_route}")

        # ── Answer quality ────────────────────────────────────────────────────
        score = {}
        score_skipped = False
        if "rate_limit" in payload.get("status", ""):
            score = {"skipped": True, "reason": payload.get("status")}
            score_skipped = True
            print(f"  Score:  skipped ({payload.get('status')})")
        elif answer_type == "exact":
            expected_value = item["expected_value"]
            match = exact_match(answer, expected_value)
            score = {"exact_match": match, "expected": expected_value}
            answer_scores.append(1.0 if match else 0.0)
            print(f"  Exact:  {'✓' if match else '✗'}  (expected: {expected_value})")
        else:
            try:
                judged = llm_judge(judge_client, question, answer)
            except Exception as exc:
                score = {"skipped": True, "reason": f"judge_error: {exc}"}
                score_skipped = True
                print(f"  Score:  skipped (judge error: {exc})")
            else:
                score = judged
                relevance = judged.get("relevance", 0)
                faithfulness = judged.get("faithfulness", 0)
                avg = (relevance + faithfulness) / 2
                answer_scores.append(avg / 10)
                print(f"  Relevance: {relevance}/10  |  Faithfulness: {faithfulness}/10")
                print(f"  Rationale: {judged.get('rationale', '')}")

        results_by_id[qid] = {
            "id": qid,
            "question": question,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_correct,
            "answer": answer,
            "score": score,
            "score_skipped": score_skipped,
        }
        write_results(results_path, ordered_results(dataset, results_by_id))

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    if router_total:
        pct = 100 * router_correct / router_total
        print(f"  Router accuracy : {router_correct}/{router_total}  ({pct:.0f}%)")
    if answer_scores:
        avg_quality = sum(answer_scores) / len(answer_scores)
        print(f"  Answer quality  : {avg_quality:.2f} / 1.00  (avg across {len(answer_scores)} questions)")
    print("=" * 70)

    # ── Save results ──────────────────────────────────────────────────────────
    write_results(results_path, ordered_results(dataset, results_by_id))
    print(f"\nDetailed results saved to {results_path.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the ACLED Router pipeline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to evaluation dataset JSON (default: evaluation_dataset.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path for evaluation results JSON (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--sleep-between-questions",
        type=float,
        default=5.0,
        help="Seconds to wait between questions (default: 5)",
    )
    parser.add_argument(
        "--graphrag-llm-provider",
        choices=("groq", "gemini"),
        default="groq",
        help="LLM provider used inside GraphRAG scoring/answering (default: groq)",
    )
    parser.add_argument(
        "--max-question-retries",
        type=int,
        default=DEFAULT_QUESTION_RETRIES,
        help="Maximum retries for transient provider errors per question (default: 12)",
    )
    parser.add_argument(
        "--retry-until-success",
        action="store_true",
        help="Retry transient provider errors forever. Use only when you are okay waiting indefinitely.",
    )
    parser.add_argument(
        "--only-route",
        choices=("all", "sparql", "graphrag"),
        default="all",
        help="Evaluate only one expected route and preserve existing results for the others (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        sys.exit(1)
    if args.max_question_retries < 1 and not args.retry_until_success:
        print("--max-question-retries must be at least 1 unless --retry-until-success is set")
        sys.exit(1)
    evaluate(
        args.dataset,
        args.output,
        args.sleep_between_questions,
        args.graphrag_llm_provider,
        args.max_question_retries,
        args.retry_until_success,
        args.only_route,
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
