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
import sys
from pathlib import Path
import re
import time
import requests
from dotenv import load_dotenv

from router import GROQ_API_KEY, GROQ_API_URL, _load_sparql_module, route_question
from graphrag_terminal import GraphRAGAnswerer, GraphRAGConfig

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "evaluation_dataset.json"
DEFAULT_RESULTS = SCRIPT_DIR / "evaluation_results.json"
JUDGE_MODEL = "llama-3.3-70b-versatile"

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

def llm_judge(question: str, answer: str) -> dict:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])


def _normalize_number(text: str) -> str:
    return re.sub(r"[,\.]", "", text.lower())

def exact_match(answer: str, expected_value: str) -> bool:
    return _normalize_number(expected_value) in _normalize_number(answer)


# ──────────────────────────────────────────────────────────────────────────────
# 3. EVALUATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(dataset_path: Path) -> None:
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("Loading pipelines...")
    sparql_module = _load_sparql_module()
    graphrag_answerer = GraphRAGAnswerer(GraphRAGConfig())

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
        if answer_type == "exact":
            expected_value = item["expected_value"]
            match = exact_match(answer, expected_value)
            score = {"exact_match": match, "expected": expected_value}
            answer_scores.append(1.0 if match else 0.0)
            print(f"  Exact:  {'✓' if match else '✗'}  (expected: {expected_value})")
        else:
            judged = llm_judge(question, answer)
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
        })

        time.sleep(5)

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
    with DEFAULT_RESULTS.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {DEFAULT_RESULTS.name}")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        sys.exit(1)
    evaluate(args.dataset)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
