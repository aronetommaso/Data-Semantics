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
JUDGE_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 6
BASE_RETRY_DELAY = 20.0

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
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
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


def _normalize_number(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"(?<=\d),(?=\d{3}\b)", "", normalized)
    normalized = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", normalized)
    return normalized

def exact_match(answer: str, expected_value: str) -> bool:
    expected = re.escape(_normalize_number(str(expected_value)).strip())
    normalized_answer = _normalize_number(answer)
    return re.search(rf"(?<![\w.]){expected}(?![\w.])", normalized_answer) is not None


def write_results(results_path: Path, results: list[dict]) -> None:
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# 3. EVALUATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    dataset_path: Path,
    results_path: Path,
    sleep_seconds: float,
    graphrag_llm_provider: str,
) -> None:
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("Loading pipelines...")
    judge_client = setup_gemini()
    sparql_module = load_sparql_module()
    graphrag_answerer = GraphRAGAnswerer(
        GraphRAGConfig(llm_provider=graphrag_llm_provider)
    )

    results = []
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

        print(f"\n[{qid}] {question}")

        payload = None
        for attempt in range(3):
            try:
                payload = route_question(question, graphrag_answerer, sparql_module)
            except Exception as exc:
                if "429" in str(exc) and attempt < 2:
                    print(f"  Rate limited, waiting 30s... (attempt {attempt + 1}/3)")
                    time.sleep(30)
                    continue
                print(f"  ERROR: {exc}")
                results.append({"id": qid, "question": question, "error": str(exc)})
                write_results(results_path, results)
                break

            # GraphRAG catches 429 internally and returns it as a status field
            if "rate_limit" in payload.get("status", ""):
                if attempt < 2:
                    print(f"  Rate limited in pipeline, waiting 30s... (attempt {attempt + 1}/3)")
                    time.sleep(30)
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

        results.append({
            "id": qid,
            "question": question,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_correct,
            "answer": answer,
            "score": score,
            "score_skipped": score_skipped,
        })
        write_results(results_path, results)

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
    write_results(results_path, results)
    print(f"\nDetailed results saved to {results_path.name}")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        sys.exit(1)
    evaluate(
        args.dataset,
        args.output,
        args.sleep_between_questions,
        args.graphrag_llm_provider,
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
