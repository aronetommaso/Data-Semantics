from __future__ import annotations

import argparse
import logging
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import networkx as nx
from networkx.algorithms.community.quality import modularity

from merge_leiden_communities import (
    ACTOR_RESOURCE_PREFIX,
    CommunityStats,
    MergeConfig,
    aggregate_stats,
    fetch_all_stats,
    is_small_community,
    load_communities,
    numeric_community_id,
    run_meta_leiden,
    save_json,
    time_overlap_similarity,
    top_component_reason,
    weighted_jaccard,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphRAGCommunityConfig(MergeConfig):
    """Configuration for building reportable second-level GraphRAG communities."""

    output_path: Path = Path("leiden_graphrag_communities.json")
    report_ready_path: Path = Path("leiden_graphrag_report_input.json")
    diagnostics_path: Path = Path("leiden_graphrag_community_report.json")
    graph_path: Path = Path("acled_leiden_network.gexf")
    target_report_count: int = 40
    seed_min_events: int = 20
    seed_min_actors: int = 5
    min_events_per_report: int = 20
    min_actors_per_report: int = 5
    max_events_per_report: int = 800
    attach_threshold: float = 0.20
    force_merge_threshold: float = 0.08
    residual_similarity_threshold: float = 0.24
    country_weight: float = 0.30
    event_type_weight: float = 0.22
    sub_event_type_weight: float = 0.18
    time_overlap_weight: float = 0.12
    source_weight: float = 0.04
    actor_graph_weight: float = 0.14


@dataclass
class SimilarityResult:
    """Interpretable similarity score between two Leiden base communities."""

    score: float
    components: dict[str, float]


def load_actor_graph(path: Path) -> nx.Graph | None:
    """Load the actor co-occurrence graph when available."""

    if not path.exists():
        LOGGER.warning("Actor graph not found: %s", path)
        return None
    return nx.read_gexf(path)


def actor_nodes_for_stats(graph: nx.Graph | None, stats: CommunityStats) -> set[str]:
    """Map actor slugs from a community to actor graph node URIs."""

    if graph is None:
        return set()
    return {
        f"{ACTOR_RESOURCE_PREFIX}{actor}"
        for actor in stats.actors
        if f"{ACTOR_RESOURCE_PREFIX}{actor}" in graph
    }


def edge_weight(graph: nx.Graph, left: str, right: str) -> float:
    """Return a numeric graph edge weight, defaulting to one."""

    return float(graph[left][right].get("weight", 1.0))


def weighted_volume(graph: nx.Graph, nodes: set[str]) -> float:
    """Compute weighted volume of a node set."""

    return sum(edge_weight(graph, node, neighbor) for node in nodes for neighbor in graph.neighbors(node))


def weighted_cut(graph: nx.Graph, nodes: set[str]) -> float:
    """Compute weighted cut between a node set and its complement."""

    return sum(
        edge_weight(graph, node, neighbor)
        for node in nodes
        for neighbor in graph.neighbors(node)
        if neighbor not in nodes
    )


def graph_conductance(graph: nx.Graph, nodes: set[str]) -> float:
    """Compute weighted conductance for a community in the actor graph."""

    if not nodes or len(nodes) == graph.number_of_nodes():
        return 0.0
    rest = set(graph.nodes()) - nodes
    denominator = min(weighted_volume(graph, nodes), weighted_volume(graph, rest))
    return weighted_cut(graph, nodes) / denominator if denominator else 0.0


def cross_actor_proximity(
    graph: nx.Graph | None,
    left_nodes: set[str],
    right_nodes: set[str],
) -> float:
    """Estimate actor-network proximity using weighted cross-community actor edges."""

    if graph is None or not left_nodes or not right_nodes:
        return 0.0

    smaller, larger = (left_nodes, right_nodes) if len(left_nodes) <= len(right_nodes) else (right_nodes, left_nodes)
    cross_weight = 0.0
    for node in smaller:
        for neighbor in graph.neighbors(node):
            if neighbor in larger:
                cross_weight += edge_weight(graph, node, neighbor)

    denominator = min(weighted_volume(graph, left_nodes), weighted_volume(graph, right_nodes))
    return cross_weight / denominator if denominator else 0.0


def community_similarity(
    left: CommunityStats,
    right: CommunityStats,
    graph: nx.Graph | None,
    actor_node_sets: dict[str, set[str]],
    config: GraphRAGCommunityConfig,
) -> SimilarityResult:
    """Compute metadata and actor-graph similarity between two base communities."""

    components = {
        "countries": weighted_jaccard(left.countries, right.countries),
        "event_types": weighted_jaccard(left.event_types, right.event_types),
        "sub_event_types": weighted_jaccard(left.sub_event_types, right.sub_event_types),
        "time_overlap": time_overlap_similarity(left, right),
        "sources": weighted_jaccard(left.top_sources, right.top_sources),
        "actor_graph": cross_actor_proximity(
            graph,
            actor_node_sets.get(left.community_id, set()),
            actor_node_sets.get(right.community_id, set()),
        ),
    }
    score = (
        config.country_weight * components["countries"]
        + config.event_type_weight * components["event_types"]
        + config.sub_event_type_weight * components["sub_event_types"]
        + config.time_overlap_weight * components["time_overlap"]
        + config.source_weight * components["sources"]
        + config.actor_graph_weight * components["actor_graph"]
    )
    return SimilarityResult(score=score, components=components)


def build_enhanced_meta_graph(
    stats_by_id: dict[str, CommunityStats],
    graph: nx.Graph | None,
    actor_node_sets: dict[str, set[str]],
    config: GraphRAGCommunityConfig,
) -> nx.Graph:
    """Build a community-level graph using metadata and actor-network similarity."""

    meta_graph = nx.Graph()
    community_ids = list(stats_by_id)
    for community_id, stats in stats_by_id.items():
        meta_graph.add_node(
            community_id,
            event_count=stats.event_count,
            actor_count=stats.actor_count,
        )

    for left_index, left_id in enumerate(community_ids):
        for right_id in community_ids[left_index + 1 :]:
            similarity = community_similarity(
                stats_by_id[left_id],
                stats_by_id[right_id],
                graph,
                actor_node_sets,
                config,
            )
            if similarity.score >= config.force_merge_threshold:
                meta_graph.add_edge(
                    left_id,
                    right_id,
                    weight=similarity.score,
                    **similarity.components,
                )
    return meta_graph


def select_seed_communities(
    stats_by_id: dict[str, CommunityStats],
    config: GraphRAGCommunityConfig,
) -> set[str]:
    """Select large enough Leiden communities as reportable seed targets."""

    candidate_ids = {
        community_id
        for community_id, stats in stats_by_id.items()
        if stats.event_count >= config.seed_min_events
        or stats.actor_count >= config.seed_min_actors
    }
    if candidate_ids:
        ordered_candidates = [
            community_id
            for community_id in sorted_by_event_count(stats_by_id)
            if community_id in candidate_ids
        ]
        seed_count = min(config.target_report_count, len(ordered_candidates))
        return set(ordered_candidates[:seed_count])

    fallback_count = max(1, min(config.target_report_count, len(stats_by_id)))
    LOGGER.warning("No seeds matched thresholds; falling back to top %s communities", fallback_count)
    return set(sorted_by_event_count(stats_by_id)[:fallback_count])


def sorted_by_event_count(stats_by_id: dict[str, CommunityStats]) -> list[str]:
    """Order community IDs by event count, actor count, and numeric suffix."""

    return sorted(
        stats_by_id,
        key=lambda community_id: (
            -stats_by_id[community_id].event_count,
            -stats_by_id[community_id].actor_count,
            numeric_community_id(community_id),
        ),
    )


def best_target(
    source_id: str,
    target_ids: Iterable[str],
    stats_by_id: dict[str, CommunityStats],
    graph: nx.Graph | None,
    actor_node_sets: dict[str, set[str]],
    config: GraphRAGCommunityConfig,
) -> tuple[str | None, SimilarityResult]:
    """Find the best seed target for a source community."""

    best_id: str | None = None
    best_similarity = SimilarityResult(score=-1.0, components={})
    for target_id in target_ids:
        if target_id == source_id:
            continue
        similarity = community_similarity(
            stats_by_id[source_id],
            stats_by_id[target_id],
            graph,
            actor_node_sets,
            config,
        )
        if similarity.score > best_similarity.score:
            best_id = target_id
            best_similarity = similarity
    return best_id, best_similarity


def group_residual_communities(
    residual_ids: set[str],
    meta_graph: nx.Graph,
    config: GraphRAGCommunityConfig,
) -> dict[str, list[str]]:
    """Cluster residual communities that are not similar enough to any seed."""

    if not residual_ids:
        return {}

    residual_graph = meta_graph.subgraph(residual_ids).copy()
    weak_edges = [
        (left, right)
        for left, right, data in residual_graph.edges(data=True)
        if float(data.get("weight", 0.0)) < config.residual_similarity_threshold
    ]
    residual_graph.remove_edges_from(weak_edges)

    grouped: dict[str, list[str]] = {}
    for index, group_ids in enumerate(
        run_meta_leiden(residual_graph, set(residual_graph.nodes), config),
        start=1,
    ):
        grouped[f"graphrag_residual_{index}"] = sorted(group_ids)
    return grouped


def aggregate_event_count(original_ids: list[str], stats_by_id: dict[str, CommunityStats]) -> int:
    """Count distinct events across a group of original Leiden communities."""

    event_ids: set[str] = set()
    for community_id in original_ids:
        event_ids.update(stats_by_id[community_id].event_ids)
    return len(event_ids)


def merge_tiny_final_groups(
    assignments: dict[str, list[str]],
    seed_ids: set[str],
    stats_by_id: dict[str, CommunityStats],
    graph: nx.Graph | None,
    actor_node_sets: dict[str, set[str]],
    config: GraphRAGCommunityConfig,
) -> list[dict[str, Any]]:
    """Force final groups below report thresholds into the closest seed when possible."""

    records: list[dict[str, Any]] = []
    tiny_final_ids = [
        final_id
        for final_id, original_ids in assignments.items()
        if final_id not in seed_ids
        and (
            aggregate_event_count(original_ids, stats_by_id) < config.min_events_per_report
            or len({actor for cid in original_ids for actor in stats_by_id[cid].actors})
            < config.min_actors_per_report
        )
    ]

    for final_id in tiny_final_ids:
        original_ids = assignments.get(final_id, [])
        if not original_ids:
            continue

        source_stats = pseudo_stats(final_id, original_ids, stats_by_id)
        augmented_stats = dict(stats_by_id)
        augmented_stats[final_id] = source_stats
        target_id, similarity = best_target(
            final_id,
            seed_ids,
            augmented_stats,
            graph,
            actor_node_sets,
            config,
        )
        if target_id and similarity.score >= config.force_merge_threshold:
            assignments[target_id].extend(original_ids)
            del assignments[final_id]
            records.append(
                {
                    "source": final_id,
                    "target": target_id,
                    "score": round(similarity.score, 4),
                    "reason": top_component_reason(similarity.components),
                    "strategy": "force_tiny_final_group_into_seed",
                }
            )
    return records


def pseudo_stats(
    community_id: str,
    original_ids: list[str],
    stats_by_id: dict[str, CommunityStats],
) -> CommunityStats:
    """Create aggregate stats for a temporary final group."""

    actors: set[str] = set()
    event_ids: set[str] = set()
    countries: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    sub_event_types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    dates: list[str] = []

    for original_id in original_ids:
        stats = stats_by_id[original_id]
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
    return CommunityStats(
        community_id=community_id,
        actors=sorted(actors),
        event_count=len(event_ids),
        actor_count=len(actors),
        countries=dict(countries),
        event_types=dict(event_types),
        sub_event_types=dict(sub_event_types),
        top_sources=dict(sources),
        first_date=ordered_dates[0] if ordered_dates else None,
        last_date=ordered_dates[-1] if ordered_dates else None,
        event_ids=sorted(event_ids),
    )


def build_graphrag_assignments(
    stats_by_id: dict[str, CommunityStats],
    seed_ids: set[str],
    meta_graph: nx.Graph,
    graph: nx.Graph | None,
    actor_node_sets: dict[str, set[str]],
    config: GraphRAGCommunityConfig,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Attach non-seed communities to seeds and cluster remaining residuals."""

    assignments: dict[str, list[str]] = {seed_id: [seed_id] for seed_id in seed_ids}
    residual_ids: set[str] = set()
    records: list[dict[str, Any]] = []

    for community_id in sorted(set(stats_by_id) - seed_ids):
        target_id, similarity = best_target(
            community_id,
            seed_ids,
            stats_by_id,
            graph,
            actor_node_sets,
            config,
        )
        if target_id and similarity.score >= config.attach_threshold:
            assignments[target_id].append(community_id)
            records.append(
                {
                    "source": community_id,
                    "target": target_id,
                    "score": round(similarity.score, 4),
                    "reason": top_component_reason(similarity.components),
                    "strategy": "attach_non_seed_to_seed",
                }
            )
        else:
            residual_ids.add(community_id)

    residual_groups = group_residual_communities(residual_ids, meta_graph, config)
    for final_id, original_ids in residual_groups.items():
        assignments[final_id] = original_ids
        records.append(
            {
                "source": original_ids,
                "target": final_id,
                "score": None,
                "reason": "residual meta-graph clustering",
                "strategy": "cluster_unassigned_residuals",
            }
        )

    records.extend(
        merge_tiny_final_groups(
            assignments,
            seed_ids,
            stats_by_id,
            graph,
            actor_node_sets,
            config,
        )
    )
    return assignments, records


def entropy(counter: dict[str, int]) -> float:
    """Compute Shannon entropy over a counter dictionary."""

    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log2(value / total) for value in counter.values() if value > 0)


def build_final_communities(
    assignments: dict[str, list[str]],
    seed_ids: set[str],
    stats_by_id: dict[str, CommunityStats],
) -> dict[str, dict[str, Any]]:
    """Aggregate assignments into final GraphRAG community payloads."""

    final: dict[str, dict[str, Any]] = {}
    for index, (final_id, original_ids) in enumerate(
        sorted(assignments.items(), key=lambda item: (-aggregate_event_count(item[1], stats_by_id), item[0])),
        start=1,
    ):
        output_id = f"graphrag_community_{index:03d}"
        community = aggregate_stats(
            final_id=output_id,
            original_ids=sorted(set(original_ids)),
            stats_by_id=stats_by_id,
            merge_reason="seed-centered GraphRAG community"
            if final_id in seed_ids
            else "residual GraphRAG meta-community",
            preserved=False,
        )
        payload = asdict(community)
        payload["seed_community_id"] = final_id if final_id in seed_ids else None
        payload["source_group_id"] = final_id
        payload["report_required"] = True
        payload["quality"] = {
            "country_entropy": entropy(payload["countries"]),
            "event_type_entropy": entropy(payload["event_types"]),
            "sub_event_type_entropy": entropy(payload["sub_event_types"]),
        }
        final[output_id] = payload
    return final


def report_ready_payload(final_communities: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Create the old-format community input expected by 03_build_report.py."""

    return {
        community_id: data["actors"]
        for community_id, data in final_communities.items()
        if data.get("report_required", True)
    }


def compute_partition_metrics(
    final_communities: dict[str, dict[str, Any]],
    original_communities: dict[str, list[str]],
    graph: nx.Graph | None,
) -> dict[str, Any]:
    """Compute modularity and conductance metrics for original and final partitions."""

    if graph is None:
        return {"error": "Actor graph not available"}

    def nodes_for_actors(actors: list[str]) -> set[str]:
        return {
            f"{ACTOR_RESOURCE_PREFIX}{actor}"
            for actor in actors
            if f"{ACTOR_RESOURCE_PREFIX}{actor}" in graph
        }

    final_partition = [nodes_for_actors(data["actors"]) for data in final_communities.values()]
    final_partition = [nodes for nodes in final_partition if nodes]
    final_covered = set().union(*final_partition) if final_partition else set()
    final_full_partition = final_partition + [{node} for node in set(graph.nodes()) - final_covered]
    final_conductances = [graph_conductance(graph, nodes) for nodes in final_partition]

    original_partition = [nodes_for_actors(actors) for actors in original_communities.values()]
    original_partition = [nodes for nodes in original_partition if nodes]
    original_covered = set().union(*original_partition) if original_partition else set()
    original_full_partition = original_partition + [{node} for node in set(graph.nodes()) - original_covered]
    original_conductances = [graph_conductance(graph, nodes) for nodes in original_partition]

    final_by_id = {
        community_id: graph_conductance(graph, nodes_for_actors(data["actors"]))
        for community_id, data in final_communities.items()
    }
    return {
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "original_partition": {
            "communities": len(original_partition),
            "modularity": modularity(graph, original_full_partition, weight="weight"),
            "average_conductance": mean(original_conductances) if original_conductances else None,
            "min_conductance": min(original_conductances) if original_conductances else None,
            "max_conductance": max(original_conductances) if original_conductances else None,
        },
        "final_partition": {
            "communities": len(final_partition),
            "modularity": modularity(graph, final_full_partition, weight="weight"),
            "average_conductance": mean(final_conductances) if final_conductances else None,
            "min_conductance": min(final_conductances) if final_conductances else None,
            "max_conductance": max(final_conductances) if final_conductances else None,
        },
        "final_conductance_by_community": {
            community_id: round(value, 6) for community_id, value in final_by_id.items()
        },
    }


def top_values(values: dict[str, int], limit: int = 5) -> dict[str, int]:
    """Return the top values from a frequency dictionary."""

    return dict(Counter(values).most_common(limit))


def build_diagnostics(
    original_count: int,
    seed_ids: set[str],
    final_communities: dict[str, dict[str, Any]],
    stats_by_id: dict[str, CommunityStats],
    merge_records: list[dict[str, Any]],
    graph_metrics: dict[str, Any],
    config: GraphRAGCommunityConfig,
) -> dict[str, Any]:
    """Build a diagnostics report for the full GraphRAG community rebuild."""

    final_summary = {
        community_id: {
            "original_community_count": len(data["original_community_ids"]),
            "actor_count": data["actor_count"],
            "event_count": data["event_count"],
            "first_date": data["first_date"],
            "last_date": data["last_date"],
            "seed_community_id": data["seed_community_id"],
            "top_countries": top_values(data["countries"]),
            "top_event_types": top_values(data["event_types"]),
            "top_sub_event_types": top_values(data["sub_event_types"]),
            "quality": data["quality"],
        }
        for community_id, data in final_communities.items()
    }
    return {
        "strategy": "full GraphRAG second-level rebuild from Leiden base communities",
        "original_number_of_leiden_communities": original_count,
        "seed_communities": len(seed_ids),
        "final_report_communities": len(final_communities),
        "target_report_count": config.target_report_count,
        "reports_avoided": max(original_count - len(final_communities), 0),
        "thresholds": {
            "seed_min_events": config.seed_min_events,
            "seed_min_actors": config.seed_min_actors,
            "min_events_per_report": config.min_events_per_report,
            "min_actors_per_report": config.min_actors_per_report,
            "max_events_per_report": config.max_events_per_report,
            "attach_threshold": config.attach_threshold,
            "force_merge_threshold": config.force_merge_threshold,
            "residual_similarity_threshold": config.residual_similarity_threshold,
        },
        "similarity_weights": {
            "country_weight": config.country_weight,
            "event_type_weight": config.event_type_weight,
            "sub_event_type_weight": config.sub_event_type_weight,
            "time_overlap_weight": config.time_overlap_weight,
            "source_weight": config.source_weight,
            "actor_graph_weight": config.actor_graph_weight,
        },
        "query_errors": {
            community_id: stats.query_error
            for community_id, stats in stats_by_id.items()
            if stats.query_error
        },
        "final_communities": final_summary,
        "merge_records": merge_records,
        "graph_metrics": graph_metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full GraphRAG rebuild."""

    parser = argparse.ArgumentParser(
        description="Build reportable GraphRAG communities from fine Leiden communities."
    )
    parser.add_argument("--input", type=Path, default=GraphRAGCommunityConfig.input_path)
    parser.add_argument("--output", type=Path, default=GraphRAGCommunityConfig.output_path)
    parser.add_argument("--report-ready", type=Path, default=GraphRAGCommunityConfig.report_ready_path)
    parser.add_argument("--diagnostics", type=Path, default=GraphRAGCommunityConfig.diagnostics_path)
    parser.add_argument("--graph", type=Path, default=GraphRAGCommunityConfig.graph_path)
    parser.add_argument("--graphdb-url", default=GraphRAGCommunityConfig.graphdb_url)
    parser.add_argument("--target-report-count", type=int, default=GraphRAGCommunityConfig.target_report_count)
    parser.add_argument("--seed-min-events", type=int, default=GraphRAGCommunityConfig.seed_min_events)
    parser.add_argument("--seed-min-actors", type=int, default=GraphRAGCommunityConfig.seed_min_actors)
    parser.add_argument("--min-events-per-report", type=int, default=GraphRAGCommunityConfig.min_events_per_report)
    parser.add_argument("--min-actors-per-report", type=int, default=GraphRAGCommunityConfig.min_actors_per_report)
    parser.add_argument("--max-events-per-report", type=int, default=GraphRAGCommunityConfig.max_events_per_report)
    parser.add_argument("--attach-threshold", type=float, default=GraphRAGCommunityConfig.attach_threshold)
    parser.add_argument("--force-merge-threshold", type=float, default=GraphRAGCommunityConfig.force_merge_threshold)
    parser.add_argument("--residual-threshold", type=float, default=GraphRAGCommunityConfig.residual_similarity_threshold)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> GraphRAGCommunityConfig:
    """Create configuration from parsed CLI arguments."""

    return GraphRAGCommunityConfig(
        input_path=args.input,
        output_path=args.output,
        report_ready_path=args.report_ready,
        diagnostics_path=args.diagnostics,
        graph_path=args.graph,
        graphdb_url=args.graphdb_url,
        target_report_count=args.target_report_count,
        seed_min_events=args.seed_min_events,
        seed_min_actors=args.seed_min_actors,
        min_events_per_report=args.min_events_per_report,
        min_actors_per_report=args.min_actors_per_report,
        max_events_per_report=args.max_events_per_report,
        attach_threshold=args.attach_threshold,
        force_merge_threshold=args.force_merge_threshold,
        residual_similarity_threshold=args.residual_threshold,
    )


def main() -> None:
    """Run the full GraphRAG community rebuild workflow."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    config = config_from_args(args)

    original_communities = load_communities(config.input_path)
    LOGGER.info("Loaded %s fine Leiden communities", len(original_communities))
    stats_by_id = fetch_all_stats(original_communities, config)
    actor_graph = load_actor_graph(config.graph_path)
    actor_node_sets = {
        community_id: actor_nodes_for_stats(actor_graph, stats)
        for community_id, stats in stats_by_id.items()
    }

    seed_ids = select_seed_communities(stats_by_id, config)
    LOGGER.info("Selected %s seed communities", len(seed_ids))
    meta_graph = build_enhanced_meta_graph(stats_by_id, actor_graph, actor_node_sets, config)
    assignments, merge_records = build_graphrag_assignments(
        stats_by_id,
        seed_ids,
        meta_graph,
        actor_graph,
        actor_node_sets,
        config,
    )
    final_communities = build_final_communities(assignments, seed_ids, stats_by_id)
    report_ready = report_ready_payload(final_communities)
    graph_metrics = compute_partition_metrics(final_communities, original_communities, actor_graph)
    diagnostics = build_diagnostics(
        original_count=len(original_communities),
        seed_ids=seed_ids,
        final_communities=final_communities,
        stats_by_id=stats_by_id,
        merge_records=merge_records,
        graph_metrics=graph_metrics,
        config=config,
    )

    save_json(config.output_path, final_communities)
    save_json(config.report_ready_path, report_ready)
    save_json(config.diagnostics_path, diagnostics)

    print(f"Original Leiden communities: {len(original_communities)}")
    print(f"Seed communities: {len(seed_ids)}")
    print(f"Final report communities: {len(final_communities)}")
    print(f"Target report count: {config.target_report_count}")
    print(f"Reports avoided: {max(len(original_communities) - len(final_communities), 0)}")
    print(f"GraphRAG communities: {config.output_path.resolve()}")
    print(f"Report-ready input: {config.report_ready_path.resolve()}")
    print(f"Diagnostics report: {config.diagnostics_path.resolve()}")


if __name__ == "__main__":
    main()
