"""
SPARQL Generator — "Pointed" mode of the ACLED GraphRAG pipeline
====================================================================
Transforms a natural language question (in English) into a SPARQL query,
executes it on GraphDB, and returns a natural language answer.

Internal pipeline:
  1. LLM generates the SPARQL query from question + ontology schema
  2. Executes the query on GraphDB via HTTP POST
  3. LLM formats the raw results into a natural language answer

Dependencies:
    pip install requests python-dotenv
"""

import os
import re
import requests
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
env_path = SCRIPT_DIR / ".env"
if not env_path.exists():
    env_path = SCRIPT_DIR.parent / ".env"
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GRAPHDB_REPO_URL = os.getenv("GRAPHDB_REPO_URL", "http://localhost:7200/repositories/acled-kg")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


# ──────────────────────────────────────────────────────────────────────────────
# 2. ONTOLOGY SCHEMA
# ──────────────────────────────────────────────────────────────────────────────

ONTOLOGY_SCHEMA = """
ACLED ONTOLOGY — Knowledge Graph in Turtle/SPARQL format

Namespaces:
  PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
  PREFIX res:  <http://data-semantics-2526.org/acled/resource/>
  PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
  PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

Main classes:
  conf:ConflictEvent  → every ACLED event
  conf:Actor          → abstract class (8 subclasses)
  conf:Country        → countries
  conf:Source         → news sources

Subclasses of conf:Actor:
  conf:StateForces, conf:RebelGroup, conf:PoliticalMilitia,
  conf:IdentityMilitia, conf:Rioters, conf:Protesters,
  conf:Civilian, conf:ExternalOther

Properties on ConflictEvent:
  conf:eventDate     (xsd:date)
  conf:disorderType  (xsd:string)
  conf:eventType     (xsd:string)
  conf:subEventType  (xsd:string)
  conf:fatalities    (xsd:integer)
  conf:notes         (xsd:string)
  conf:locationName  (xsd:string)
  conf:latitude      (xsd:decimal)
  conf:longitude     (xsd:decimal)
  conf:hasActor1     → conf:Actor
  conf:hasActor2     → conf:Actor
  conf:locatedIn     → conf:Country
  conf:reportedBy    → conf:Source

Properties on Actor:
  conf:actorName     (xsd:string)

Properties on Country:
  conf:countryName   (xsd:string)
  conf:isoCode       (xsd:integer)

Properties on Source:
  conf:sourceName    (xsd:string)

IMPORTANT DISTINCTIONS:
  - conf:locatedIn → the COUNTRY WHERE THE EVENT HAPPENED (geographic location)
  - conf:hasActor1/hasActor2 → WHO WAS INVOLVED in the event (can be from any country)
  Example: an Israeli army attack in Palestine has locatedIn=Palestine, hasActor1=Israeli Forces

Countries WHERE EVENTS ARE LOCATED (use these exact names with conf:countryName):
  Palestine, Syria, Bahrain, Yemen, Iran, Turkey, Iraq, Israel,
  Lebanon, Jordan, Saudi Arabia, Qatar, Kuwait,
  United Arab Emirates, Oman

Actor names follow a pattern like:
  "Military Forces of Israel (2022-)"
  "Military Forces of Turkey (2016-)"
  "Hamas Movement"
  Use REGEX or CONTAINS to match actor names across time periods.

Date range of data: 2015-01-03 to 2023-12-31

QUERY STYLE RULES:
  - ALWAYS use SELECT instead of ASK. Instead of asking "is X present?",
    count how many events involve X (COUNT). This gives more useful information.
  - When searching for an actor that may have multiple time-period variants
    (e.g. "Military Forces of Israel (2009-2021)" and "Military Forces of Israel (2022-)"),
    use FILTER(CONTAINS(LCASE(?actorName), "keyword")) to match all variants.
  - NEVER use "?actor a conf:Actor" to find actors — OWL reasoning is disabled,
    so conf:Actor as a class returns no results. Always reach actors via events:
      { ?event conf:hasActor1 ?actor } UNION { ?event conf:hasActor2 ?actor }
  - ALWAYS add ORDER BY and LIMIT 20 when the query returns a list of entities
    (actors, countries, event types, etc.). Never return unbounded lists.
  - ALWAYS scope COUNT to a specific pattern, never COUNT(*) on joined patterns.
  - For event categories such as "Battles", "Protests", or
    "Explosions/Remote violence", match literal values with conf:eventType.
    Do not treat event types as RDF classes or resources.
  - For journalistic source questions, use conf:reportedBy ?source and
    ?source conf:sourceName ?sourceName. Count DISTINCT ?sourceName when the
    question asks for unique journalistic sources.
  - For actor subclasses such as State Forces or Rebel Group, use RDF type
    triples with the subclass, e.g. ?actor a conf:StateForces. Do not search
    for the words "state forces" in actor names.
  - When both actor positions are relevant, wrap the UNION inside braces:
      { ?event conf:hasActor1 ?actor } UNION { ?event conf:hasActor2 ?actor }
    Then put shared filters/triples after the UNION.

"""


# ──────────────────────────────────────────────────────────────────────────────
# 3. FEW-SHOT EXAMPLES
# ──────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
EXAMPLE 1
Question: "How many events are there in total in the Knowledge Graph?"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT (COUNT(?event) AS ?total)
WHERE { ?event a conf:ConflictEvent . }

EXAMPLE 2
Question: "Total fatalities in Yemen in 2022?"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT (SUM(?fat) AS ?totalFatalities)
WHERE {
  ?event a conf:ConflictEvent ;
         conf:fatalities ?fat ;
         conf:eventDate ?date ;
         conf:locatedIn ?country .
  ?country conf:countryName "Yemen" .
  FILTER(?date >= "2022-01-01"^^xsd:date && ?date <= "2022-12-31"^^xsd:date)
}

EXAMPLE 3
Question: "Top 5 actors by number of events"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT ?actorName (COUNT(?event) AS ?eventCount)
WHERE {
  ?event a conf:ConflictEvent .
  { ?event conf:hasActor1 ?actor } UNION { ?event conf:hasActor2 ?actor }
  ?actor conf:actorName ?actorName .
}
GROUP BY ?actorName
ORDER BY DESC(?eventCount)
LIMIT 5

EXAMPLE 4
Question: "Which countries had more than 1000 events?"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT ?countryName (COUNT(?event) AS ?eventCount)
WHERE {
  ?event a conf:ConflictEvent ;
         conf:locatedIn ?country .
  ?country conf:countryName ?countryName .
}
GROUP BY ?countryName
HAVING (COUNT(?event) > 1000)
ORDER BY DESC(?eventCount)

EXAMPLE 5
Question: "Is Israel included in the events?"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
EXAMPLE 5
Question: "Is Israel included in the events?"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT (SUM(?c) AS ?total) WHERE {
  {
    SELECT (COUNT(DISTINCT ?ev) AS ?c) WHERE {
      ?ev a conf:ConflictEvent ;
          conf:locatedIn ?country .
      ?country conf:countryName "Israel" .
    }
  }
  UNION
  {
    SELECT (COUNT(DISTINCT ?ev) AS ?c) WHERE {
      ?ev a conf:ConflictEvent .
      { ?ev conf:hasActor1 ?actor } UNION { ?ev conf:hasActor2 ?actor }
      ?actor conf:actorName ?actorName .
      FILTER(CONTAINS(LCASE(?actorName), "israel"))
    }
  }
}

EXAMPLE 6
Question: "How many events involved Israeli forces?"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT (COUNT(DISTINCT ?event) AS ?eventCount)
WHERE {
  ?event a conf:ConflictEvent .
  { ?event conf:hasActor1 ?actor } UNION { ?event conf:hasActor2 ?actor }
  ?actor conf:actorName ?actorName .
  FILTER(CONTAINS(LCASE(?actorName), "israel"))
}

EXAMPLE 7
Question: "How many events are categorized as Battles?"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT (COUNT(DISTINCT ?event) AS ?eventCount)
WHERE {
  ?event a conf:ConflictEvent ;
         conf:eventType "Battles" .
}

EXAMPLE 8
Question: "Which country experienced the highest number of protests?"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT ?countryName (COUNT(DISTINCT ?event) AS ?eventCount)
WHERE {
  ?event a conf:ConflictEvent ;
         conf:eventType "Protests" ;
         conf:locatedIn ?country .
  ?country conf:countryName ?countryName .
}
GROUP BY ?countryName
ORDER BY DESC(?eventCount)
LIMIT 1

EXAMPLE 9
Question: "What is the primary event type involving State Forces?"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT ?eventType (COUNT(DISTINCT ?event) AS ?eventCount)
WHERE {
  ?event a conf:ConflictEvent ;
         conf:eventType ?eventType .
  { ?event conf:hasActor1 ?actor } UNION { ?event conf:hasActor2 ?actor }
  ?actor a conf:StateForces .
}
GROUP BY ?eventType
ORDER BY DESC(?eventCount)
LIMIT 1

EXAMPLE 10
Question: "Identify the most active rebel group in the dataset based on total event frequency."
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT ?actorName (COUNT(DISTINCT ?event) AS ?eventCount)
WHERE {
  ?event a conf:ConflictEvent .
  { ?event conf:hasActor1 ?actor } UNION { ?event conf:hasActor2 ?actor }
  ?actor a conf:RebelGroup ;
         conf:actorName ?actorName .
}
GROUP BY ?actorName
ORDER BY DESC(?eventCount)
LIMIT 1

EXAMPLE 11
Question: "How many unique journalistic sources are mapped in the graph?"
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT (COUNT(DISTINCT ?sourceName) AS ?sourceCount)
WHERE {
  ?event a conf:ConflictEvent ;
         conf:reportedBy ?source .
  ?source conf:sourceName ?sourceName .
}

EXAMPLE 12
Question: "List the most cited journalistic source for events with strictly more than 50 fatalities."
Query:
PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
SELECT ?sourceName (COUNT(DISTINCT ?event) AS ?eventCount)
WHERE {
  ?event a conf:ConflictEvent ;
         conf:fatalities ?fatalities ;
         conf:reportedBy ?source .
  ?source conf:sourceName ?sourceName .
  FILTER(?fatalities > 50)
}
GROUP BY ?sourceName
ORDER BY DESC(?eventCount)
LIMIT 1
"""


# ──────────────────────────────────────────────────────────────────────────────
# 4. FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def require_groq_api_key() -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env")
    return GROQ_API_KEY


def post_with_retry(url: str, **kwargs) -> requests.Response:
    timeout = kwargs.pop("timeout", REQUEST_TIMEOUT)
    response = None
    for attempt in range(MAX_RETRIES):
        response = requests.post(url, timeout=timeout, **kwargs)
        if response.status_code not in RETRY_STATUS_CODES or attempt == MAX_RETRIES - 1:
            return response

        retry_after = response.headers.get("Retry-After")
        wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 * (attempt + 1)
        time.sleep(min(wait_seconds, 30.0))

    if response is None:
        raise RuntimeError("HTTP request was not executed")
    return response


def call_groq(prompt: str, system_message: str = "") -> str:
    headers = {
        "Authorization": f"Bearer {require_groq_api_key()}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
    }
    response = post_with_retry(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def extract_sparql(text: str) -> str:
    match = re.search(r"```(?:sparql)?\s*(.*?)```", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = text.strip()

    if re.match(r"(?is)^\s*(PREFIX\s+\w+:\s*<[^>]+>\s*)*(SELECT|CONSTRUCT|DESCRIBE|ASK)\b", candidate):
        return candidate
    raise ValueError("No SPARQL query found in model response")


def generate_sparql_query(question: str) -> str:
    system_msg = (
        "You are an expert in SPARQL and Knowledge Graphs. Your task is to "
        "translate natural language questions into valid SPARQL queries "
        "for the ACLED Knowledge Graph. Always use SELECT, never ASK.\n\n"
        "PERFORMANCE RULES (mandatory):\n"
        "- NEVER combine two independent COUNT patterns in the same WHERE with OPTIONAL. "
        "Use UNION of subqueries instead.\n"
        "- Always scope COUNT(DISTINCT ...) to a single coherent pattern.\n"
        "- NEVER use correlated OPTIONALs for independent aggregations."
    )
    prompt = f"""
{ONTOLOGY_SCHEMA}

{FEW_SHOT_EXAMPLES}

Now translate this question into a valid SPARQL query.
Respond ONLY with the SPARQL query, enclosed between ```sparql and ```.
Do not add explanations.

Question: "{question}"
"""
    response = call_groq(prompt, system_msg)
    return extract_sparql(response)


def execute_sparql(query: str) -> dict:
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/sparql-query",
    }
    response = post_with_retry(GRAPHDB_REPO_URL, data=query, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"GraphDB error ({response.status_code}): {response.text}")
    return response.json()


def format_results_as_text(results: dict) -> str:
    # Handle ASK queries
    if "boolean" in results:
        return f"Answer: {'Yes' if results['boolean'] else 'No'}"
    # Handle SELECT queries
    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return "No results found."
    lines = []
    for i, row in enumerate(bindings, 1):
        row_parts = []
        for var, val in row.items():
            row_parts.append(f"{var}={val['value']}")
        lines.append(f"Row {i}: " + " | ".join(row_parts))
    return "\n".join(lines)


MAX_RESULTS_CHARS = 6000

def explain_results(question: str, query: str, results_text: str) -> str:
    if len(results_text) > MAX_RESULTS_CHARS:
        results_text = results_text[:MAX_RESULTS_CHARS] + "\n[... truncated for length ...]"
    prompt = f"""
The user asked this question: "{question}"

To answer it, the following SPARQL query was executed:
{query}

The results obtained from the Knowledge Graph are:
{results_text}

Generate a clear and concise answer in English based on the results.
Do not invent information that is not in the data. If the data is numerical,
present it in a readable way (e.g. use thousand separators).
"""
    return call_groq(prompt)


def answer_question(question: str, verbose: bool = False) -> dict:
    """Pipeline: question → SPARQL → raw results → NL answer."""
    if verbose:
        print(f"\nQuestion: {question}\n")

    # Step 1: Generate SPARQL
    if verbose:
        print("Generating SPARQL query...")
    query = generate_sparql_query(question)
    if verbose:
        print(f"\nGenerated query:\n{query}\n")

    # Step 2: Execute on GraphDB
    if verbose:
        print("Executing on GraphDB...")
    raw_results = execute_sparql(query)
    results_text = format_results_as_text(raw_results)
    if verbose:
        print(f"\nRaw results:\n{results_text}\n")

    # Step 3: Format as natural language
    if verbose:
        print("Formatting final answer...")
    final_answer = explain_results(question, query, results_text)
    if verbose:
        print(f"\nAnswer:\n{final_answer}\n")

    return {
        "route": "sparql",
        "query": question,
        "answer": final_answer,
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "query_generator": MODEL_NAME,
            "answer_generator": MODEL_NAME,
        },
        "retrieval": {
            "sparql_query": query,
            "results_text": results_text,
        },
        "context": {
            "raw_results": raw_results,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN — Interactive mode
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  SPARQL Generator — ACLED Knowledge Graph Q&A")
    print("=" * 70)
    print(f"GraphDB: {GRAPHDB_REPO_URL}")
    print(f"Model:   {MODEL_NAME}")
    print()
    print("Type a question in English (or 'exit' to quit)")
    print("-" * 70)

    while True:
        try:
            question = input("\n> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break
            answer_question(question, verbose=True)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
