"""
Router — ACLED GraphRAG pipeline
====================================================================
Receives a natural language question, decides which pipeline to use
(SPARQL for specific/factual questions, GraphRAG for general/analytical
questions), calls it, and returns a unified payload.

Usage:
    python router.py                    # interactive mode
    python router.py --query "..."      # single question and exit
    python router.py --json             # print full JSON payload
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from graphrag_terminal import GraphRAGAnswerer, GraphRAGConfig

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
env_path = SCRIPT_DIR / ".env"
if not env_path.exists():
    env_path = SCRIPT_DIR.parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
ROUTER_MODEL = "llama-3.1-8b-instant"

# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD SPARQL MODULE (filename has parentheses, cannot use normal import)
# ──────────────────────────────────────────────────────────────────────────────

def _load_sparql_module():
    path = SCRIPT_DIR / "SPARQL_Generator_cle(daNLaSPARQL).py"
    spec = importlib.util.spec_from_file_location("sparql_generator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ──────────────────────────────────────────────────────────────────────────────
# 3. ROUTING LOGIC
# ──────────────────────────────────────────────────────────────────────────────

ROUTING_SYSTEM = """
You are a query router for the ACLED conflict data pipeline.
Given a user question, decide which pipeline should answer it.

Return ONLY a JSON object with one key: "route", with value "sparql" or "graphrag".

Use "sparql" for:
- Specific numerical questions (counts, sums, averages)
- Questions about specific actors, countries, or dates
- Questions requiring precise filtering ("in 2022", "in Yemen", "how many")
- Factual lookups ("which country had the most", "total fatalities of X")

Use "graphrag" for:
- General pattern or trend analysis
- Questions about relationships between groups of actors
- Broad thematic questions ("describe the conflict", "what are the main dynamics")
- Questions that require synthesizing across many events without a precise number
"""


def decide_route(question: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": ROUTING_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    route = parsed.get("route", "sparql").lower()
    return route if route in {"sparql", "graphrag"} else "sparql"


# ──────────────────────────────────────────────────────────────────────────────
# 4. MAIN ROUTING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def route_question(
    question: str,
    graphrag_answerer: GraphRAGAnswerer,
    sparql_module,
) -> dict:
    """Decide which pipeline to use and return a unified payload."""
    route = decide_route(question)
    print(f"[Router] → {route.upper()}")
    if route == "graphrag":
        return graphrag_answerer.answer(question)
    return sparql_module.answer_question(question)


# ──────────────────────────────────────────────────────────────────────────────
# 5. TERMINAL INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

def run_interactive(
    graphrag_answerer: GraphRAGAnswerer,
    sparql_module,
    as_json: bool,
) -> None:
    print("=" * 70)
    print("  ACLED Router — SPARQL + GraphRAG")
    print("=" * 70)
    print("Type a question in English (or 'exit' to quit)")
    print("-" * 70)

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        try:
            payload = route_question(question, graphrag_answerer, sparql_module)
            if as_json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print(f"\nAnswer:\n{payload['answer']}\n")
        except Exception as exc:
            print(f"\nError: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACLED Router — SPARQL + GraphRAG")
    parser.add_argument("--query", help="Run a single question and exit.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sparql_module = _load_sparql_module()
    graphrag_answerer = GraphRAGAnswerer(GraphRAGConfig())

    if args.query:
        payload = route_question(args.query, graphrag_answerer, sparql_module)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(payload["answer"])
        return

    run_interactive(graphrag_answerer, sparql_module, as_json=args.json)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
