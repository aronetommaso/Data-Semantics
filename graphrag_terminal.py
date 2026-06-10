"""
Terminal GraphRAG runner for ACLED community reports.

This script follows the Microsoft GraphRAG global-search idea at a small scale:
community reports are first scored for relevance by a lightweight LLM, then the
highest-scoring reports are passed as context to a stronger answer LLM.

Dependencies:
    pip install requests python-dotenv google-genai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from groq_key_rotation import (
    current_groq_api_key,
    current_groq_key_name,
    groq_api_key_count,
    require_groq_api_key,
    rotate_groq_api_key,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMMUNITIES_PATH = SCRIPT_DIR / "leiden_graphrag_communities.json"
DEFAULT_REPORTS_DIR = SCRIPT_DIR / "trio" / "final_community_reports"
DEFAULT_RETRIEVAL_LOG_PATH = SCRIPT_DIR / "graphrag_retrieval_validation.jsonl"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_SCORER_MODEL = "llama-3.1-8b-instant"
DEFAULT_ANSWER_MODEL = "llama-3.3-70b-versatile"
DEFAULT_LLM_PROVIDER = "groq"
DEFAULT_GEMINI_SCORER_MODEL = "gemini-2.5-flash-lite"
DEFAULT_GEMINI_ANSWER_MODEL = "gemini-3.5-flash"


@dataclass(frozen=True)
class CommunityReport:
    """Compact textual representation of one GraphRAG community report.

    Attributes:
        report_id: Stable community/report identifier.
        title: Human-readable report title.
        text: Text passed to the LLM scorer and answer model.
        metadata: Structured fields useful for validation and future routing.
    """

    report_id: str
    title: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalScore:
    """LLM relevance score for one community report.

    Attributes:
        report_id: Stable community/report identifier.
        score: Relevance score from 0 to 10.
        rationale: Short reason returned by the scoring LLM.
        matched_evidence: Short evidence cues found in the report.
    """

    report_id: str
    score: float
    rationale: str
    matched_evidence: list[str]


@dataclass(frozen=True)
class GraphRAGConfig:
    """Runtime configuration for the terminal GraphRAG pipeline.

    Attributes:
        reports_dir: Directory containing final markdown community reports.
        communities_path: Optional path to the precomputed GraphRAG communities JSON.
        scorer_model: Lightweight model used for report relevance scoring.
        answer_model: Stronger model used for final answer synthesis.
        score_threshold: Minimum relevance score for selecting reports.
        max_context_reports: Maximum number of selected reports in the answer context.
        max_scorer_reports: Maximum number of lexical candidates sent to the scorer LLM.
        scorer_report_chars: Maximum report characters sent to the scorer LLM.
        answer_report_chars: Maximum report characters sent to the answer LLM.
        request_timeout: HTTP timeout in seconds for Groq calls.
        retrieval_log_path: Optional JSONL path used to validate retrieval decisions.
        llm_provider: LLM provider for GraphRAG scoring and answer generation.
    """

    reports_dir: Path = DEFAULT_REPORTS_DIR
    communities_path: Path | None = DEFAULT_COMMUNITIES_PATH
    llm_provider: str = DEFAULT_LLM_PROVIDER
    scorer_model: str = DEFAULT_SCORER_MODEL
    answer_model: str = DEFAULT_ANSWER_MODEL
    score_threshold: float = 7.0
    max_context_reports: int = 4
    max_scorer_reports: int = 5
    scorer_report_chars: int = 6000
    answer_report_chars: int = 12000
    request_timeout: int = 90
    retrieval_log_path: Path | None = DEFAULT_RETRIEVAL_LOG_PATH

    def __post_init__(self) -> None:
        provider = self.llm_provider.lower()
        if provider not in {"groq", "gemini"}:
            raise ValueError("llm_provider must be 'groq' or 'gemini'")
        object.__setattr__(self, "llm_provider", provider)
        if provider == "gemini" and self.scorer_model == DEFAULT_SCORER_MODEL:
            object.__setattr__(self, "scorer_model", DEFAULT_GEMINI_SCORER_MODEL)
        if provider == "gemini" and self.answer_model == DEFAULT_ANSWER_MODEL:
            object.__setattr__(self, "answer_model", DEFAULT_GEMINI_ANSWER_MODEL)


class GroqClient:
    """Small OpenAI-compatible client for Groq chat completions."""

    def __init__(self, api_key: str | None = None, api_url: str = GROQ_API_URL, timeout: int = 90) -> None:
        """Initialize the Groq client.

        Args:
            api_key: Groq API key.
            api_url: Groq OpenAI-compatible chat completions endpoint.
            timeout: Request timeout in seconds.
        """

        self.api_key = api_key
        self.api_url = api_url
        self.timeout = timeout

    def chat(
        self,
        model: str,
        system_message: str,
        user_message: str,
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Call a Groq chat model and return the assistant content.

        Args:
            model: Groq model name.
            system_message: System instruction.
            user_message: User prompt.
            temperature: Sampling temperature.
            response_format: Optional OpenAI-compatible response format.

        Returns:
            Assistant message content.

        Raises:
            requests.HTTPError: If Groq returns a non-success status code.
        """

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        response = None
        for key_attempt in range(max(groq_api_key_count(), 1)):
            api_key = self.api_key or current_groq_api_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            for attempt in range(3):
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code != 429:
                    break

                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 * (attempt + 1)
                time.sleep(min(wait_seconds, 8.0))

            if response is None or response.status_code != 429:
                break
            if key_attempt < groq_api_key_count() - 1:
                previous_key = current_groq_key_name()
                self.api_key = rotate_groq_api_key()
                print(f"[Groq] rate limit on {previous_key}; switching to {current_groq_key_name()}")

        if response is None:
            raise RuntimeError("Groq request was not executed")
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class GeminiClient:
    """Small client adapter exposing the same chat interface for Gemini."""

    def __init__(self, api_key: str, timeout: int = 90) -> None:
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key.
            timeout: Reserved for interface compatibility.
        """

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Install google-genai to use --llm-provider gemini") from exc

        self.client = genai.Client(api_key=api_key)
        self.timeout = timeout

    def chat(
        self,
        model: str,
        system_message: str,
        user_message: str,
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Call a Gemini model and return the text content."""

        config: dict[str, Any] = {"temperature": temperature}
        if response_format and response_format.get("type") == "json_object":
            config["response_mime_type"] = "application/json"

        response = self.client.models.generate_content(
            model=model,
            contents=f"{system_message.strip()}\n\n{user_message}",
            config=config,
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response")
        return response.text


def load_environment(provider: str = DEFAULT_LLM_PROVIDER) -> str:
    """Load `.env` and return the selected provider API key.

    Returns:
        API key from `GROQ_API_KEY` or `GEMINI_API_KEY`.

    Raises:
        ValueError: If the API key is missing.
    """

    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        env_path = SCRIPT_DIR.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    provider = provider.lower()
    if provider == "groq":
        return require_groq_api_key()

    env_key = "GEMINI_API_KEY"
    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(f"{env_key} not found in .env")
    return api_key


def make_llm_client(provider: str, timeout: int):
    """Build the configured GraphRAG LLM client."""

    api_key = load_environment(provider)
    if provider == "gemini":
        return GeminiClient(api_key, timeout=timeout)
    return GroqClient(api_key, timeout=timeout)


def is_rate_limit_error(exc: Exception) -> bool:
    """Return whether an LLM exception looks like a provider rate limit."""

    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code == 429
    message = str(exc)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def top_items(values: dict[str, int] | None, limit: int = 8) -> list[tuple[str, int]]:
    """Return the largest items from a count dictionary.

    Args:
        values: Count dictionary.
        limit: Maximum number of items.

    Returns:
        Sorted `(name, count)` pairs.
    """

    if not values:
        return []
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]


def format_counts(values: dict[str, int] | None, limit: int = 8) -> str:
    """Format a count dictionary as compact text.

    Args:
        values: Count dictionary.
        limit: Maximum number of items to include.

    Returns:
        Comma-separated count text.
    """

    items = top_items(values, limit=limit)
    return ", ".join(f"{name} ({count})" for name, count in items) if items else "None"


def clean_markdown_report(text: str) -> str:
    """Normalize a markdown report for LLM consumption.

    Args:
        text: Raw markdown report text.

    Returns:
        Markdown report text without an outer fenced code block.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:markdown)?\s*", "", stripped, count=1)
        stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    return stripped.strip()


def report_id_from_path(path: Path) -> str:
    """Parse the GraphRAG community id from a markdown report filename.

    Args:
        path: Markdown report path.

    Returns:
        Parsed report id.

    Raises:
        ValueError: If the filename does not contain a GraphRAG community id.
    """

    match = re.search(r"(graphrag_community_\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse report id from filename: {path.name}")
    return match.group(1)


def extract_title(markdown_text: str, fallback: str) -> str:
    """Extract the first markdown heading as the report title.

    Args:
        markdown_text: Clean markdown report text.
        fallback: Fallback title.

    Returns:
        Report title.
    """

    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_community_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load optional structured community metadata.

    Args:
        path: Optional path to `leiden_graphrag_communities.json`.

    Returns:
        Dictionary keyed by report id.
    """

    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if isinstance(value, dict)}


def metadata_from_raw(raw: dict[str, Any] | None, markdown_path: Path) -> dict[str, Any]:
    """Create compact validation metadata for one report.

    Args:
        raw: Optional structured community metadata.
        markdown_path: Markdown report path.

    Returns:
        Metadata dictionary.
    """

    report_text = markdown_path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {
        "report_file": str(markdown_path),
        "report_size_chars": len(report_text),
        "report_size_bytes": markdown_path.stat().st_size,
    }
    if not raw:
        return metadata
    metadata.update(
        {
            "event_count": raw.get("event_count"),
            "actor_count": raw.get("actor_count"),
            "countries": top_items(raw.get("countries"), 10),
            "event_types": top_items(raw.get("event_types"), 8),
            "sub_event_types": top_items(raw.get("sub_event_types"), 10),
            "first_date": raw.get("first_date"),
            "last_date": raw.get("last_date"),
            "original_community_ids": raw.get("original_community_ids", []),
        }
    )
    return metadata


def load_community_reports(reports_dir: Path, communities_path: Path | None = None) -> list[CommunityReport]:
    """Load final GraphRAG community reports from markdown files.

    Args:
        reports_dir: Directory containing `report_graphrag_community_*.md` files.
        communities_path: Optional structured metadata JSON path.

    Returns:
        List of markdown report objects sorted by report id.

    Raises:
        FileNotFoundError: If the report directory or markdown reports are missing.
    """

    if not reports_dir.exists():
        raise FileNotFoundError(f"Markdown reports directory not found: {reports_dir}")

    paths = sorted(reports_dir.glob("report_graphrag_community_*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown community reports found in: {reports_dir}")

    metadata_by_id = load_community_metadata(communities_path)
    reports = []
    for path in paths:
        report_id = report_id_from_path(path)
        markdown_text = clean_markdown_report(path.read_text(encoding="utf-8"))
        reports.append(
            CommunityReport(
                report_id=report_id,
                title=extract_title(markdown_text, f"Community report {report_id}"),
                text=f"Report ID: {report_id}\nReport file: {path.name}\n\n{markdown_text}",
                metadata=metadata_from_raw(metadata_by_id.get(report_id), path),
            )
        )
    return reports


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response.

    Args:
        text: Raw model response.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If no JSON object can be parsed.
    """

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {text[:200]}")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object in model response")
    return value


def normalize_score(value: Any) -> float:
    """Normalize a model score to the 0-10 range.

    Args:
        value: Raw score value.

    Returns:
        Clamped float score.
    """

    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(10.0, score))


def tokenize(text: str) -> set[str]:
    """Extract normalized keyword tokens from text.

    Args:
        text: Input text.

    Returns:
        Set of meaningful lowercase tokens.
    """

    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "which",
        "what",
        "where",
        "when",
        "main",
        "events",
        "event",
        "are",
        "were",
        "was",
        "in",
        "on",
        "of",
        "to",
        "from",
        "by",
        "a",
        "an",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if token not in stopwords
    }


def local_relevance_score(report: CommunityReport, query: str) -> RetrievalScore:
    """Score one report with a cheap lexical heuristic before LLM scoring.

    Args:
        report: Community report.
        query: User query.

    Returns:
        Retrieval score marked as local prefilter evidence.
    """

    query_tokens = tokenize(query)
    text_lower = report.text.lower()
    matched = sorted(token for token in query_tokens if token in text_lower)
    score = min(10.0, len(matched) * 2.5)

    metadata = report.metadata
    country_names = [name.lower() for name, _count in metadata.get("countries", [])]
    metadata_matches = []
    for token in query_tokens:
        matched_countries = [country for country in country_names if token in country]
        if matched_countries:
            score = max(score, 7.0)
            metadata_matches.extend(f"country metadata: {country}" for country in matched_countries)

    return RetrievalScore(
        report_id=report.report_id,
        score=score,
        rationale="Local lexical prefilter score before LLM report scoring.",
        matched_evidence=(matched + metadata_matches)[:8],
    )


def truncate_text(text: str, max_chars: int) -> str:
    """Trim long report text while preserving the beginning and ending.

    Args:
        text: Original text.
        max_chars: Maximum characters to keep.

    Returns:
        Truncated text.
    """

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars].rstrip()
        + "\n\n[... report truncated for token budget ...]\n\n"
        + text[-tail_chars:].lstrip()
    )


def report_for_budget(report: CommunityReport, max_chars: int) -> CommunityReport:
    """Return a copy of a report with text trimmed to a character budget.

    Args:
        report: Source report.
        max_chars: Maximum text characters.

    Returns:
        Report with truncated text.
    """

    return CommunityReport(
        report_id=report.report_id,
        title=report.title,
        text=truncate_text(report.text, max_chars),
        metadata=report.metadata,
    )


def query_years(query: str) -> set[int]:
    """Extract explicit years from the user query.

    Args:
        query: User query.

    Returns:
        Set of years.
    """

    return {int(match) for match in re.findall(r"\b(19\d{2}|20\d{2})\b", query)}


def available_year_range(reports: list[CommunityReport]) -> tuple[int | None, int | None]:
    """Find the available year range from report metadata.

    Args:
        reports: Loaded community reports.

    Returns:
        `(first_year, last_year)` if available.
    """

    years = []
    for report in reports:
        for key in ("first_date", "last_date"):
            value = report.metadata.get(key)
            if isinstance(value, str) and re.match(r"\d{4}-", value):
                years.append(int(value[:4]))
    if not years:
        return None, None
    return min(years), max(years)


def is_outside_available_years(query: str, reports: list[CommunityReport]) -> bool:
    """Check whether explicit query years fall outside report coverage.

    Args:
        query: User query.
        reports: Loaded community reports.

    Returns:
        True if all explicit query years are outside the report range.
    """

    years = query_years(query)
    first_year, last_year = available_year_range(reports)
    if not years or first_year is None or last_year is None:
        return False
    return all(year < first_year or year > last_year for year in years)


def score_report(
    client: GroqClient,
    report: CommunityReport,
    query: str,
    model: str,
    max_report_chars: int,
) -> RetrievalScore:
    """Score one report for relevance to the user query.

    Args:
        client: Groq client.
        report: Community report to score.
        query: User query.
        model: Lightweight Groq model used for scoring.
        max_report_chars: Maximum report characters sent to the scorer.

    Returns:
        Retrieval score.
    """

    report = report_for_budget(report, max_chars=max_report_chars)
    system_message = (
        "You are the retrieval scorer in a GraphRAG pipeline. "
        "Score whether the community report contains useful context for answering the user query. "
        "Return only JSON with keys: score, rationale, matched_evidence. "
        "The score must be a number from 0 to 10. matched_evidence must be a short list of strings."
    )
    user_message = (
        f"User query:\n{query}\n\n"
        f"Community report:\n{report.text}\n\n"
        "Return JSON only."
    )

    content = client.chat(
        model=model,
        system_message=system_message,
        user_message=user_message,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    parsed = extract_json_object(content)
    evidence = parsed.get("matched_evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    return RetrievalScore(
        report_id=report.report_id,
        score=normalize_score(parsed.get("score")),
        rationale=str(parsed.get("rationale", "")).strip(),
        matched_evidence=[str(item).strip() for item in evidence if str(item).strip()],
    )


def select_reports(
    reports: list[CommunityReport],
    scores: list[RetrievalScore],
    score_threshold: float,
    max_context_reports: int,
) -> tuple[list[CommunityReport], list[RetrievalScore]]:
    """Select high-scoring reports using a Microsoft-style relevance filter.

    Args:
        reports: All reports.
        scores: Retrieval scores for all reports.
        score_threshold: Minimum score to include a report.
        max_context_reports: Maximum number of reports to include.

    Returns:
        Selected reports and their scores, sorted by descending score.
    """

    report_by_id = {report.report_id: report for report in reports}
    ranked_scores = sorted(scores, key=lambda item: item.score, reverse=True)
    selected_scores = [score for score in ranked_scores if score.score >= score_threshold]
    if not selected_scores and ranked_scores:
        selected_scores = [ranked_scores[0]]
    selected_scores = selected_scores[:max_context_reports]
    selected_reports = [report_by_id[score.report_id] for score in selected_scores if score.report_id in report_by_id]
    return selected_reports, selected_scores


def build_answer_context(
    selected_reports: list[CommunityReport],
    selected_scores: list[RetrievalScore],
    max_report_chars: int,
) -> str:
    """Build final answer context from selected reports.

    Args:
        selected_reports: Reports selected for context.
        selected_scores: Scores for selected reports.
        max_report_chars: Maximum report characters per selected report.

    Returns:
        Context string for answer synthesis.
    """

    score_by_id = {score.report_id: score for score in selected_scores}
    blocks = []
    for index, report in enumerate(selected_reports, start=1):
        report = report_for_budget(report, max_report_chars)
        score = score_by_id[report.report_id]
        evidence = "; ".join(score.matched_evidence) if score.matched_evidence else "No specific evidence returned"
        blocks.append(
            f"[Selected report {index}]\n"
            f"Relevance score: {score.score:.1f}/10\n"
            f"Retrieval rationale: {score.rationale}\n"
            f"Matched evidence: {evidence}\n"
            f"{report.text}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_from_reports(
    client: GroqClient,
    query: str,
    selected_reports: list[CommunityReport],
    selected_scores: list[RetrievalScore],
    model: str,
    max_report_chars: int,
) -> str:
    """Generate the final answer from selected GraphRAG reports.

    Args:
        client: Groq client.
        query: User query.
        selected_reports: Reports selected for answer context.
        selected_scores: Retrieval scores for selected reports.
        model: Groq model used for answer synthesis.
        max_report_chars: Maximum report characters per selected report.

    Returns:
        Natural-language answer in English.
    """

    if not selected_reports:
        return "I could not retrieve a relevant community report for this question."

    system_message = (
        "You are the answer generator in an ACLED GraphRAG pipeline. "
        "Answer in English using only the selected community reports. "
        "Be concise, mention uncertainty when the reports are too aggregated, "
        "and cite report ids in square brackets such as [graphrag_community_001]."
    )
    user_message = (
        f"User query:\n{query}\n\n"
        f"Selected GraphRAG report context:\n"
        f"{build_answer_context(selected_reports, selected_scores, max_report_chars)}\n\n"
        "Write the answer now."
    )
    return client.chat(
        model=model,
        system_message=system_message,
        user_message=user_message,
        temperature=0.2,
    ).strip()


def append_retrieval_log(path: Path, payload: dict[str, Any]) -> None:
    """Append one retrieval validation record to a JSONL file.

    Args:
        path: Output JSONL path.
        payload: Router-compatible answer payload.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ranked_report_info(
    scores: list[RetrievalScore],
    reports: list[CommunityReport],
) -> list[dict[str, Any]]:
    """Build compact report information ordered by score.

    Args:
        scores: Retrieval scores.
        reports: Loaded community reports.

    Returns:
        Ranked report info dictionaries for UI display and validation logs.
    """

    report_by_id = {report.report_id: report for report in reports}
    ranked = []
    for score in sorted(scores, key=lambda item: item.score, reverse=True):
        report = report_by_id.get(score.report_id)
        if not report:
            continue
        ranked.append(
            {
                "report_id": report.report_id,
                "title": report.title,
                "score": score.score,
                "metadata": report.metadata,
            }
        )
    return ranked


class GraphRAGAnswerer:
    """Router-ready GraphRAG component."""

    def __init__(self, config: GraphRAGConfig) -> None:
        """Initialize the answerer and load reports.

        Args:
            config: Runtime GraphRAG configuration.
        """

        self.config = config
        self.client = make_llm_client(config.llm_provider, timeout=config.request_timeout)
        self.reports = load_community_reports(config.reports_dir, config.communities_path)

    def answer(self, query: str, log_retrieval: bool = True) -> dict[str, Any]:
        """Answer a user query with GraphRAG.

        Args:
            query: User query.
            log_retrieval: Whether to append a validation JSONL row.

        Returns:
            Router-compatible payload with answer and retrieval diagnostics.
        """

        local_scores = [local_relevance_score(report, query) for report in self.reports]

        if is_outside_available_years(query, self.reports):
            first_year, last_year = available_year_range(self.reports)
            payload = {
                "route": "graphrag",
                "query": query,
                "answer": (
                    f"The available GraphRAG community reports cover {first_year}-{last_year}, "
                    "so they do not contain evidence for the explicit year requested. "
                    "Use Query Answering only if the knowledge graph contains that year; otherwise this is out of scope."
                ),
                "status": "no_temporal_coverage",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "models": {
                    "retrieval_scorer": "local_temporal_coverage_check",
                    "answer_generator": "none",
                },
                "retrieval": {
                    "strategy": "temporal_coverage_check_before_llm_scoring",
                    "score_threshold": self.config.score_threshold,
                    "max_context_reports": self.config.max_context_reports,
                    "max_scorer_reports": 0,
                    "total_reports_scored": 0,
                    "selected_report_ids": [],
                    "scores": [asdict(score) for score in sorted(local_scores, key=lambda item: item.score, reverse=True)],
                    "ranked_report_info": ranked_report_info(local_scores, self.reports),
                },
                "context": {"selected_reports": []},
            }
            if log_retrieval and self.config.retrieval_log_path:
                append_retrieval_log(self.config.retrieval_log_path, payload)
            return payload

        report_by_id = {report.report_id: report for report in self.reports}
        local_ranked = sorted(local_scores, key=lambda item: item.score, reverse=True)
        candidate_scores = local_ranked[: self.config.max_scorer_reports]
        scores_by_id = {score.report_id: score for score in local_scores}
        llm_scores_by_id = {}
        llm_scored_ids = []
        rate_limited = False

        for candidate in candidate_scores:
            report = report_by_id[candidate.report_id]
            try:
                llm_score = score_report(
                    self.client,
                    report,
                    query,
                    self.config.scorer_model,
                    self.config.scorer_report_chars,
                )
                scores_by_id[llm_score.report_id] = llm_score
                llm_scores_by_id[llm_score.report_id] = llm_score
                llm_scored_ids.append(llm_score.report_id)
            except Exception as exc:
                if is_rate_limit_error(exc):
                    rate_limited = True
                    scores_by_id[candidate.report_id] = RetrievalScore(
                        report_id=candidate.report_id,
                        score=candidate.score,
                        rationale=(
                            f"{self.config.llm_provider} rate limit reached during LLM scoring; "
                            "using local prefilter score."
                        ),
                        matched_evidence=candidate.matched_evidence,
                    )
                    break
                raise

        scores = list(scores_by_id.values())
        selection_scores = list(llm_scores_by_id.values()) if llm_scores_by_id else scores
        selected_reports, selected_scores = select_reports(
            self.reports,
            selection_scores,
            self.config.score_threshold,
            self.config.max_context_reports,
        )
        answer_status = "rate_limited_retrieval_fallback" if rate_limited else "ok"
        try:
            answer = answer_from_reports(
                self.client,
                query,
                selected_reports,
                selected_scores,
                self.config.answer_model,
                self.config.answer_report_chars,
            )
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise
            answer_status = "rate_limited_answer_generation"
            selected_ids = ", ".join(score.report_id for score in selected_scores) or "none"
            answer = (
                f"{self.config.llm_provider} rate limit was reached during answer generation. "
                f"Retrieval still selected these reports for validation: {selected_ids}. "
                "Retry later or lower --max-scorer-reports, --scorer-report-chars, or --answer-report-chars."
            )

        payload = {
            "route": "graphrag",
            "query": query,
            "answer": answer,
            "status": answer_status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "models": {
                "llm_provider": self.config.llm_provider,
                "retrieval_scorer": self.config.scorer_model,
                "answer_generator": self.config.answer_model,
            },
            "retrieval": {
                "strategy": "local_prefilter_then_llm_score_top_community_reports",
                "score_threshold": self.config.score_threshold,
                "max_context_reports": self.config.max_context_reports,
                "max_scorer_reports": self.config.max_scorer_reports,
                "llm_scored_report_ids": llm_scored_ids,
                "rate_limited": rate_limited,
                "total_reports_scored": len(scores),
                "selected_report_ids": [score.report_id for score in selected_scores],
                "scores": [asdict(score) for score in sorted(scores, key=lambda item: item.score, reverse=True)],
                "ranked_report_info": ranked_report_info(scores, self.reports),
            },
            "context": {
                "selected_reports": [
                    {
                        "report_id": report.report_id,
                        "title": report.title,
                        "metadata": report.metadata,
                    }
                    for report in selected_reports
                ]
            },
        }
        if log_retrieval and self.config.retrieval_log_path:
            append_retrieval_log(self.config.retrieval_log_path, payload)
        return payload


def print_retrieval_table(payload: dict[str, Any]) -> None:
    """Print a compact retrieval validation view.

    Args:
        payload: Router-compatible GraphRAG payload.
    """

    scores = payload["retrieval"]["scores"]
    selected_ids = set(payload["retrieval"]["selected_report_ids"])
    selected_reports = payload["context"]["selected_reports"]
    selected_by_id = {report["report_id"]: report for report in selected_reports}
    ranked_info_by_id = {
        report["report_id"]: report
        for report in payload["retrieval"].get("ranked_report_info", [])
    }

    print("\nRetrieval ranking: top 5 reports")
    print("-" * 80)
    for rank, item in enumerate(scores[:5], start=1):
        selected = "*" if item["report_id"] in selected_ids else " "
        report = ranked_info_by_id.get(item["report_id"], {})
        metadata = report.get("metadata", {})
        title = report.get("title", "Unknown report title")
        rationale = item["rationale"].replace("\n", " ")[:72]
        evidence = ", ".join(item.get("matched_evidence", [])[:3]) or "no explicit evidence listed"
        countries = ", ".join(name for name, _count in metadata.get("countries", [])[:3]) or "unknown"
        dates = f"{metadata.get('first_date', 'unknown')} to {metadata.get('last_date', 'unknown')}"
        print(f"{rank:>2}. {selected} {item['report_id']:<28} score={item['score']:>4.1f}")
        print(f"    title: {title}")
        print(f"    period/countries: {dates} | {countries}")
        print(f"    why: {rationale}")
        print(f"    evidence: {evidence[:120]}")

    print("-" * 80)
    print("* = selected for answer context")

    print("\nReports used as answer context")
    print("-" * 80)
    if not selected_reports:
        print("No reports were selected for answer context.")
    for report in selected_reports:
        metadata = report.get("metadata", {})
        score = next((item for item in scores if item["report_id"] == report["report_id"]), None)
        score_text = f"{score['score']:.1f}" if score else "n/a"
        countries = ", ".join(name for name, _count in metadata.get("countries", [])[:5]) or "unknown"
        event_types = ", ".join(name for name, _count in metadata.get("event_types", [])[:4]) or "unknown"
        dates = f"{metadata.get('first_date', 'unknown')} to {metadata.get('last_date', 'unknown')}"
        print(f"- {report['report_id']} | score={score_text} | {report['title']}")
        print(f"  period: {dates}")
        print(f"  countries: {countries}")
        print(f"  event types: {event_types}")
    print("-" * 80)
    print()


def run_interactive(answerer: GraphRAGAnswerer, as_json: bool, show_retrieval: bool) -> None:
    """Run a terminal chat loop.

    Args:
        answerer: GraphRAG answerer.
        as_json: Whether to print full JSON payloads.
        show_retrieval: Whether to print the retrieval validation table.
    """

    print("ACLED GraphRAG terminal UI")
    print("Type a question in English. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            query = input("GraphRAG> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            break

        try:
            payload = answerer.answer(query)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        if as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            if show_retrieval:
                print_retrieval_table(payload)
            print("Answer")
            print("-" * 80)
            print(payload["answer"])
            print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Run the ACLED GraphRAG terminal pipeline.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--communities-path", type=Path, default=DEFAULT_COMMUNITIES_PATH)
    parser.add_argument("--llm-provider", choices=("groq", "gemini"), default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--scorer-model", default=DEFAULT_SCORER_MODEL)
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--score-threshold", type=float, default=7.0)
    parser.add_argument("--max-context-reports", type=int, default=4)
    parser.add_argument("--max-scorer-reports", type=int, default=5)
    parser.add_argument("--scorer-report-chars", type=int, default=6000)
    parser.add_argument("--answer-report-chars", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retrieval-log-path", type=Path, default=DEFAULT_RETRIEVAL_LOG_PATH)
    parser.add_argument("--no-retrieval-log", action="store_true")
    parser.add_argument("--show-retrieval", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print router-compatible JSON output.")
    parser.add_argument("--query", help="Run one query and exit.")
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point."""

    args = parse_args()
    log_path = None if args.no_retrieval_log else args.retrieval_log_path
    config = GraphRAGConfig(
        reports_dir=args.reports_dir,
        communities_path=args.communities_path,
        llm_provider=args.llm_provider,
        scorer_model=args.scorer_model,
        answer_model=args.answer_model,
        score_threshold=args.score_threshold,
        max_context_reports=args.max_context_reports,
        max_scorer_reports=args.max_scorer_reports,
        scorer_report_chars=args.scorer_report_chars,
        answer_report_chars=args.answer_report_chars,
        request_timeout=args.timeout,
        retrieval_log_path=log_path,
    )
    answerer = GraphRAGAnswerer(config)

    if args.query:
        payload = answerer.answer(args.query)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            if args.show_retrieval:
                print_retrieval_table(payload)
            print(payload["answer"])
        return

    run_interactive(answerer, as_json=args.json, show_retrieval=args.show_retrieval)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
