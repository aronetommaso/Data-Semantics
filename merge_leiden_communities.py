from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import requests

try:
    from cdlib import algorithms
except ImportError:  # pragma: no cover - optional dependency fallback
    algorithms = None


LOGGER = logging.getLogger(__name__)

CONF_PREFIX = "http://data-semantics-2526.org/acled/ontology#"
ACTOR_RESOURCE_PREFIX = "http://data-semantics-2526.org/acled/resource/actor/"


@dataclass(frozen=True)
class MergeConfig:
    """Configuration for building higher-level GraphRAG communities."""

    input_path: Path = Path("leiden_communities.json")
    output_path: Path = Path("leiden_communities_merged.json")
    report_path: Path = Path("leiden_merge_report.json")
    graphdb_url: str = "http://localhost:7200/repositories/MiddleEastConflict"
    preserve_top_n: int = 30
    min_events_per_report: int = 20
    min_actors_per_report: int = 5
    similarity_threshold: float = 0.18
    meta_leiden_threshold: float = 0.28
    max_top_values: int = 12
    country_weight: float = 0.35
    event_type_weight: float = 0.25
    sub_event_type_weight: float = 0.20
    time_overlap_weight: float = 0.15
    source_weight: float = 0.05
    request_timeout_seconds: int = 60


@dataclass
class CommunityStats:
    """GraphDB enrichment data for one Leiden community."""

    community_id: str
    actors: list[str]
    event_count: int = 0
    actor_count: int = 0
    countries: dict[str, int] = field(default_factory=dict)
    event_types: dict[str, int] = field(default_factory=dict)
    sub_event_types: dict[str, int] = field(default_factory=dict)
    top_sources: dict[str, int] = field(default_factory=dict)
    first_date: str | None = None
    last_date: str | None = None
    event_ids: list[str] = field(default_factory=list)
    query_error: str | None = None

    @property
    def is_empty(self) -> bool:
        """Return True when no events were found for this community."""

        return self.event_count == 0


@dataclass
class FinalCommunity:
    """A GraphRAG-ready community after higher-level merging."""

    community_id: str
    original_community_ids: list[str]
    actors: list[str]
    event_count: int
    actor_count: int
    countries: dict[str, int]
    event_types: dict[str, int]
    sub_event_types: dict[str, int]
    top_sources: dict[str, int]
    first_date: str | None
    last_date: str | None
    merge_reason: str
    preserved: bool = False


def load_communities(path: Path) -> dict[str, list[str]]:
    """Load Leiden communities from a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {str(key): list(value) for key, value in data.items()}


def save_json(path: Path, data: Any) -> None:
    """Save JSON data with stable indentation and UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def numeric_community_id(community_id: str) -> int:
    """Extract the numeric suffix from IDs such as community_12."""

    try:
        return int(community_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def actor_uri(actor_slug: str) -> str:
    """Build a full actor URI from the slug stored in Leiden JSON."""

    return f"{ACTOR_RESOURCE_PREFIX}{actor_slug}"


def sparql_headers() -> dict[str, str]:
    """Return standard headers for GraphDB SPARQL queries."""

    return {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/sparql-query",
    }


def run_sparql_query(query: str, config: MergeConfig) -> list[dict[str, Any]]:
    """Run a SPARQL query against GraphDB and return result bindings."""

    response = requests.post(
        config.graphdb_url,
        data=query,
        headers=sparql_headers(),
        timeout=config.request_timeout_seconds,
    )
    response.raise_for_status()
    return response.json().get("results", {}).get("bindings", [])


def build_stats_query(actor_slugs: list[str]) -> str:
    """Build a readable SPARQL query that enriches one community."""

    values_clause = " ".join(f"<{actor_uri(actor)}>" for actor in actor_slugs)
    return f"""
PREFIX conf: <{CONF_PREFIX}>

SELECT ?event ?date ?countryName ?eventType ?subEventType ?sourceName WHERE {{
  VALUES ?actorUri {{ {values_clause} }}

  ?event a conf:ConflictEvent .
  {{
    ?event conf:hasActor1 ?actorUri .
  }}
  UNION
  {{
    ?event conf:hasActor2 ?actorUri .
  }}

  OPTIONAL {{ ?event conf:eventDate ?date . }}
  OPTIONAL {{
    ?event conf:locatedIn ?country .
    OPTIONAL {{ ?country conf:countryName ?countryName . }}
  }}
  OPTIONAL {{ ?event conf:eventType ?eventType . }}
  OPTIONAL {{ ?event conf:subEventType ?subEventType . }}
  OPTIONAL {{
    ?event conf:reportedBy ?source .
    ?source conf:sourceName ?sourceName .
  }}
}}
"""


def binding_value(row: dict[str, Any], key: str) -> str | None:
    """Extract a string value from a SPARQL binding row."""

    value = row.get(key, {}).get("value")
    return str(value) if value is not None else None


def trim_counter(counter: Counter[str], max_values: int) -> dict[str, int]:
    """Return the most common values from a counter as a plain dict."""

    return dict(counter.most_common(max_values))


def fetch_community_stats(
    community_id: str,
    actors: list[str],
    config: MergeConfig,
) -> CommunityStats:
    """Fetch event and metadata statistics for one Leiden community."""

    stats = CommunityStats(
        community_id=community_id,
        actors=actors,
        actor_count=len(actors),
    )
    if not actors:
        return stats

    try:
        bindings = run_sparql_query(build_stats_query(actors), config)
    except requests.RequestException as exc:
        LOGGER.warning("GraphDB query failed for %s: %s", community_id, exc)
        stats.query_error = str(exc)
        return stats

    events: set[str] = set()
    dates: list[str] = []
    countries: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    sub_event_types: Counter[str] = Counter()
    sources: Counter[str] = Counter()

    for row in bindings:
        event = binding_value(row, "event")
        if event:
            events.add(event)

        event_date = binding_value(row, "date")
        if event_date:
            dates.append(event_date[:10])

        for key, counter in (
            ("countryName", countries),
            ("eventType", event_types),
            ("subEventType", sub_event_types),
            ("sourceName", sources),
        ):
            value = binding_value(row, key)
            if value:
                counter[value] += 1

    unique_dates = sorted(set(dates))
    stats.event_count = len(events)
    stats.event_ids = sorted(events)
    stats.first_date = unique_dates[0] if unique_dates else None
    stats.last_date = unique_dates[-1] if unique_dates else None
    stats.countries = trim_counter(countries, config.max_top_values)
    stats.event_types = trim_counter(event_types, config.max_top_values)
    stats.sub_event_types = trim_counter(sub_event_types, config.max_top_values)
    stats.top_sources = trim_counter(sources, config.max_top_values)
    return stats


def fetch_all_stats(
    communities: dict[str, list[str]],
    config: MergeConfig,
) -> dict[str, CommunityStats]:
    """Fetch GraphDB statistics for every community."""

    stats_by_id: dict[str, CommunityStats] = {}
    for index, (community_id, actors) in enumerate(communities.items(), start=1):
        LOGGER.info("Fetching stats for %s (%s/%s)", community_id, index, len(communities))
        stats_by_id[community_id] = fetch_community_stats(community_id, actors, config)
    return stats_by_id


def is_small_community(stats: CommunityStats, config: MergeConfig) -> bool:
    """Return True when a community is too small to deserve its own LLM report."""

    return (
        stats.event_count < config.min_events_per_report
        or stats.actor_count < config.min_actors_per_report
    )


def parse_date(value: str | None) -> date | None:
    """Parse an ISO date string, returning None for missing or invalid values."""

    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def weighted_jaccard(left: dict[str, int], right: dict[str, int]) -> float:
    """Compute weighted Jaccard similarity between two counted feature maps."""

    keys = set(left) | set(right)
    if not keys:
        return 0.0
    intersection = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
    union = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
    return intersection / union if union else 0.0


def time_overlap_similarity(left: CommunityStats, right: CommunityStats) -> float:
    """Estimate similarity between two community date ranges."""

    left_start = parse_date(left.first_date)
    left_end = parse_date(left.last_date)
    right_start = parse_date(right.first_date)
    right_end = parse_date(right.last_date)
    if not all((left_start, left_end, right_start, right_end)):
        return 0.0

    assert left_start is not None
    assert left_end is not None
    assert right_start is not None
    assert right_end is not None

    overlap_days = (min(left_end, right_end) - max(left_start, right_start)).days + 1
    if overlap_days <= 0:
        return 0.0

    span_days = (max(left_end, right_end) - min(left_start, right_start)).days + 1
    return overlap_days / span_days if span_days else 0.0


def community_similarity(
    left: CommunityStats,
    right: CommunityStats,
    config: MergeConfig,
) -> tuple[float, dict[str, float]]:
    """Compute an interpretable similarity score between two communities."""

    components = {
        "countries": weighted_jaccard(left.countries, right.countries),
        "event_types": weighted_jaccard(left.event_types, right.event_types),
        "sub_event_types": weighted_jaccard(left.sub_event_types, right.sub_event_types),
        "time_overlap": time_overlap_similarity(left, right),
        "sources": weighted_jaccard(left.top_sources, right.top_sources),
    }
    score = (
        config.country_weight * components["countries"]
        + config.event_type_weight * components["event_types"]
        + config.sub_event_type_weight * components["sub_event_types"]
        + config.time_overlap_weight * components["time_overlap"]
        + config.source_weight * components["sources"]
    )
    return score, components


def build_meta_graph(
    stats_by_id: dict[str, CommunityStats],
    preserved_ids: set[str],
    config: MergeConfig,
) -> nx.Graph:
    """Build a community-level graph weighted by metadata similarity."""

    graph = nx.Graph()
    community_ids = list(stats_by_id)
    for community_id, stats in stats_by_id.items():
        graph.add_node(
            community_id,
            event_count=stats.event_count,
            actor_count=stats.actor_count,
            preserved=community_id in preserved_ids,
        )

    for left_index, left_id in enumerate(community_ids):
        for right_id in community_ids[left_index + 1 :]:
            score, components = community_similarity(
                stats_by_id[left_id],
                stats_by_id[right_id],
                config,
            )
            if score >= config.similarity_threshold:
                graph.add_edge(left_id, right_id, weight=score, **components)
    return graph


def top_component_reason(components: dict[str, float]) -> str:
    """Return a compact human-readable reason from similarity components."""

    useful = [(name, score) for name, score in components.items() if score > 0]
    if not useful:
        return "fallback: no strong metadata overlap"
    name, score = max(useful, key=lambda item: item[1])
    return f"metadata similarity via {name} ({score:.2f})"


def best_target_for_small_community(
    small_id: str,
    target_ids: Iterable[str],
    stats_by_id: dict[str, CommunityStats],
    config: MergeConfig,
) -> tuple[str | None, float, dict[str, float]]:
    """Find the most similar non-preserved merge target for a small community."""

    best_id: str | None = None
    best_score = -1.0
    best_components: dict[str, float] = {}
    for target_id in target_ids:
        if target_id == small_id:
            continue
        score, components = community_similarity(
            stats_by_id[small_id],
            stats_by_id[target_id],
            config,
        )
        if score > best_score:
            best_id = target_id
            best_score = score
            best_components = components
    return best_id, best_score, best_components


def aggregate_stats(
    final_id: str,
    original_ids: list[str],
    stats_by_id: dict[str, CommunityStats],
    merge_reason: str,
    preserved: bool = False,
) -> FinalCommunity:
    """Aggregate original community statistics into one final community."""

    actors: set[str] = set()
    event_ids: set[str] = set()
    countries: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    sub_event_types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    dates: list[str] = []

    for community_id in original_ids:
        stats = stats_by_id[community_id]
        actors.update(stats.actors)
        event_ids.update(stats.event_ids)
        countries.update(stats.countries)
        event_types.update(stats.event_types)
        sub_event_types.update(stats.sub_event_types)
        sources.update(stats.top_sources)
        if stats.first_date:
            dates.append(stats.first_date)
        if stats.last_date:
            dates.append(stats.last_date)

    ordered_dates = sorted(set(dates))
    return FinalCommunity(
        community_id=final_id,
        original_community_ids=original_ids,
        actors=sorted(actors),
        event_count=len(event_ids),
        actor_count=len(actors),
        countries=dict(countries.most_common()),
        event_types=dict(event_types.most_common()),
        sub_event_types=dict(sub_event_types.most_common()),
        top_sources=dict(sources.most_common()),
        first_date=ordered_dates[0] if ordered_dates else None,
        last_date=ordered_dates[-1] if ordered_dates else None,
        merge_reason=merge_reason,
        preserved=preserved,
    )


def run_meta_leiden(
    meta_graph: nx.Graph,
    candidate_ids: set[str],
    config: MergeConfig,
) -> list[list[str]]:
    """Run Leiden on the meta-graph candidates, with greedy fallback."""

    subgraph = meta_graph.subgraph(candidate_ids).copy()
    weak_edges = [
        (left, right)
        for left, right, data in subgraph.edges(data=True)
        if float(data.get("weight", 0.0)) < config.meta_leiden_threshold
    ]
    subgraph.remove_edges_from(weak_edges)

    if subgraph.number_of_nodes() == 0:
        return []
    if subgraph.number_of_edges() == 0:
        return [[node] for node in subgraph.nodes]

    if algorithms is not None:
        try:
            partition = algorithms.leiden(subgraph, weights="weight")
            return [list(community) for community in partition.communities]
        except Exception as exc:  # pragma: no cover - depends on optional backend
            LOGGER.warning("Meta-Leiden failed, using greedy modularity: %s", exc)

    communities = nx.algorithms.community.greedy_modularity_communities(
        subgraph,
        weight="weight",
    )
    return [list(community) for community in communities]


def merge_small_communities(
    stats_by_id: dict[str, CommunityStats],
    preserved_ids: set[str],
    meta_graph: nx.Graph,
    config: MergeConfig,
) -> tuple[dict[str, FinalCommunity], list[dict[str, Any]]]:
    """Merge non-preserved small communities into higher-level communities."""

    final: dict[str, FinalCommunity] = {}
    merge_records: list[dict[str, Any]] = []

    remaining_ids = [community_id for community_id in stats_by_id if community_id not in preserved_ids]
    large_ids = {
        community_id
        for community_id in remaining_ids
        if not is_small_community(stats_by_id[community_id], config)
    }
    small_ids = set(remaining_ids) - large_ids

    assignments: dict[str, list[str]] = {community_id: [community_id] for community_id in large_ids}
    unassigned_small: set[str] = set()

    for small_id in sorted(small_ids):
        target_id, score, components = best_target_for_small_community(
            small_id,
            large_ids,
            stats_by_id,
            config,
        )
        if target_id and score >= config.similarity_threshold:
            assignments[target_id].append(small_id)
            merge_records.append(
                {
                    "source": small_id,
                    "target": target_id,
                    "score": round(score, 4),
                    "reason": top_component_reason(components),
                    "strategy": "merge_small_into_larger",
                }
            )
        else:
            unassigned_small.add(small_id)

    for group_index, group_ids in enumerate(
        run_meta_leiden(meta_graph, unassigned_small, config),
        start=1,
    ):
        final_id = f"meta_small_{group_index}"
        assignments[final_id] = sorted(group_ids)
        merge_records.append(
            {
                "source": sorted(group_ids),
                "target": final_id,
                "score": None,
                "reason": "meta-graph Leiden/greedy grouping among small communities",
                "strategy": "meta_graph_clustering",
            }
        )

    for community_id in preserved_ids:
        final[community_id] = aggregate_stats(
            final_id=community_id,
            original_ids=[community_id],
            stats_by_id=stats_by_id,
            merge_reason="preserved: already reported",
            preserved=True,
        )

    for final_id, original_ids in sorted(assignments.items()):
        final[final_id] = aggregate_stats(
            final_id=final_id,
            original_ids=sorted(original_ids),
            stats_by_id=stats_by_id,
            merge_reason="higher-level merged community"
            if len(original_ids) > 1
            else "kept as sufficiently large remaining community",
        )

    return final, merge_records


def build_diagnostics(
    original_count: int,
    final_communities: dict[str, FinalCommunity],
    preserved_ids: set[str],
    stats_by_id: dict[str, CommunityStats],
    merge_records: list[dict[str, Any]],
    config: MergeConfig,
) -> dict[str, Any]:
    """Build a diagnostics report for the merge process."""

    final_count = len(final_communities)
    small_count = sum(
        1
        for community_id, stats in stats_by_id.items()
        if community_id not in preserved_ids and is_small_community(stats, config)
    )
    avoided_reports = max(original_count - final_count, 0)
    return {
        "original_number_of_communities": original_count,
        "final_number_of_communities": final_count,
        "number_of_preserved_communities": len(preserved_ids),
        "number_of_small_remaining_communities": small_count,
        "number_of_merged_small_communities": len(
            {
                source
                for record in merge_records
                for source in (
                    record["source"]
                    if isinstance(record["source"], list)
                    else [record["source"]]
                )
            }
        ),
        "reports_avoided": avoided_reports,
        "token_saving_estimate": {
            "reports_avoided": avoided_reports,
            "formula": "reports_avoided * average_tokens_per_report",
            "example_at_3000_tokens_per_report": avoided_reports * 3000,
        },
        "thresholds": {
            "preserve_top_n": config.preserve_top_n,
            "min_events_per_report": config.min_events_per_report,
            "min_actors_per_report": config.min_actors_per_report,
            "similarity_threshold": config.similarity_threshold,
            "meta_leiden_threshold": config.meta_leiden_threshold,
        },
        "query_errors": {
            community_id: stats.query_error
            for community_id, stats in stats_by_id.items()
            if stats.query_error
        },
        "merge_reasons": merge_records,
        "final_community_sizes": {
            community_id: {
                "actor_count": community.actor_count,
                "event_count": community.event_count,
                "original_community_count": len(community.original_community_ids),
                "preserved": community.preserved,
            }
            for community_id, community in final_communities.items()
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the merge script."""

    parser = argparse.ArgumentParser(
        description="Merge tiny Leiden communities into GraphRAG-ready higher-level communities."
    )
    parser.add_argument("--input", type=Path, default=MergeConfig.input_path)
    parser.add_argument("--output", type=Path, default=MergeConfig.output_path)
    parser.add_argument("--report", type=Path, default=MergeConfig.report_path)
    parser.add_argument("--graphdb-url", default=MergeConfig.graphdb_url)
    parser.add_argument("--preserve-top-n", type=int, default=MergeConfig.preserve_top_n)
    parser.add_argument("--min-events-per-report", type=int, default=MergeConfig.min_events_per_report)
    parser.add_argument("--min-actors-per-report", type=int, default=MergeConfig.min_actors_per_report)
    parser.add_argument("--similarity-threshold", type=float, default=MergeConfig.similarity_threshold)
    parser.add_argument("--meta-leiden-threshold", type=float, default=MergeConfig.meta_leiden_threshold)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> MergeConfig:
    """Create a MergeConfig instance from CLI arguments."""

    return MergeConfig(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        graphdb_url=args.graphdb_url,
        preserve_top_n=args.preserve_top_n,
        min_events_per_report=args.min_events_per_report,
        min_actors_per_report=args.min_actors_per_report,
        similarity_threshold=args.similarity_threshold,
        meta_leiden_threshold=args.meta_leiden_threshold,
    )


def main() -> None:
    """CLI entrypoint for Leiden community post-processing."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    config = config_from_args(args)

    communities = load_communities(config.input_path)
    LOGGER.info("Loaded %s original Leiden communities", len(communities))
    stats_by_id = fetch_all_stats(communities, config)

    sorted_ids = sorted(
        stats_by_id,
        key=lambda community_id: (
            -stats_by_id[community_id].event_count,
            -stats_by_id[community_id].actor_count,
            numeric_community_id(community_id),
        ),
    )
    preserved_ids = set(sorted_ids[: config.preserve_top_n])
    LOGGER.info(
        "Preserving %s already reported communities, ordered by event count",
        len(preserved_ids),
    )

    meta_graph = build_meta_graph(stats_by_id, preserved_ids, config)
    final_communities, merge_records = merge_small_communities(
        stats_by_id,
        preserved_ids,
        meta_graph,
        config,
    )
    diagnostics = build_diagnostics(
        original_count=len(communities),
        final_communities=final_communities,
        preserved_ids=preserved_ids,
        stats_by_id=stats_by_id,
        merge_records=merge_records,
        config=config,
    )

    save_json(
        config.output_path,
        {
            community_id: asdict(community)
            for community_id, community in final_communities.items()
        },
    )
    save_json(config.report_path, diagnostics)

    print(f"Original communities: {len(communities)}")
    print(f"Final communities: {len(final_communities)}")
    print(f"Preserved communities: {len(preserved_ids)}")
    print(f"Merged output: {config.output_path.resolve()}")
    print(f"Diagnostics report: {config.report_path.resolve()}")


if __name__ == "__main__":
    main()
