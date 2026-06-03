from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import networkx as nx
from networkx.algorithms.community.quality import modularity

from merge_leiden_communities import (
    ACTOR_RESOURCE_PREFIX,
    CommunityStats,
    MergeConfig,
    aggregate_stats,
    best_target_for_small_community,
    build_meta_graph,
    fetch_all_stats,
    is_small_community,
    load_communities,
    numeric_community_id,
    run_meta_leiden,
    save_json,
    top_component_reason,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttachMergeConfig(MergeConfig):
    """Configuration for attaching small communities to preserved report targets."""

    output_path: Path = Path("leiden_communities_attached_merged.json")
    diagnostics_path: Path = Path("leiden_attach_merge_report.json")
    report_ready_path: Path = Path("leiden_attached_new_reports_input.json")
    graph_path: Path = Path("acled_leiden_network.gexf")
    attach_to_preserved_threshold: float = 0.18


def sorted_by_event_count(stats_by_id: dict[str, CommunityStats]) -> list[str]:
    """Return community IDs ordered by event count, actor count, then numeric ID."""

    return sorted(
        stats_by_id,
        key=lambda community_id: (
            -stats_by_id[community_id].event_count,
            -stats_by_id[community_id].actor_count,
            numeric_community_id(community_id),
        ),
    )


def merge_with_preserved_targets(
    stats_by_id: dict[str, CommunityStats],
    preserved_ids: set[str],
    meta_graph: nx.Graph,
    config: AttachMergeConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
    """Attach small communities to preserved reports when similar, otherwise create new report groups."""

    remaining_ids = [community_id for community_id in stats_by_id if community_id not in preserved_ids]
    large_remaining_ids = {
        community_id
        for community_id in remaining_ids
        if not is_small_community(stats_by_id[community_id], config)
    }
    small_ids = set(remaining_ids) - large_remaining_ids

    assignments: dict[str, list[str]] = {
        community_id: [community_id] for community_id in preserved_ids | large_remaining_ids
    }
    report_required_ids: set[str] = set(large_remaining_ids)
    unassigned_small: set[str] = set()
    merge_records: list[dict[str, Any]] = []

    candidate_targets = preserved_ids | large_remaining_ids
    for small_id in sorted(small_ids):
        target_id, score, components = best_target_for_small_community(
            small_id,
            candidate_targets,
            stats_by_id,
            config,
        )
        if target_id and score >= config.attach_to_preserved_threshold:
            assignments[target_id].append(small_id)
            merge_records.append(
                {
                    "source": small_id,
                    "target": target_id,
                    "target_report_status": "already_reported"
                    if target_id in preserved_ids
                    else "new_report_required",
                    "score": round(score, 4),
                    "reason": top_component_reason(components),
                    "strategy": "attach_small_to_best_target_including_preserved",
                }
            )
        else:
            unassigned_small.add(small_id)

    for group_index, group_ids in enumerate(
        run_meta_leiden(meta_graph, unassigned_small, config),
        start=1,
    ):
        final_id = f"attached_meta_small_{group_index}"
        assignments[final_id] = sorted(group_ids)
        report_required_ids.add(final_id)
        merge_records.append(
            {
                "source": sorted(group_ids),
                "target": final_id,
                "target_report_status": "new_report_required",
                "score": None,
                "reason": "meta-graph Leiden/greedy grouping after preserved-target attachment",
                "strategy": "residual_meta_graph_clustering",
            }
        )

    final_communities: dict[str, dict[str, Any]] = {}
    for final_id, original_ids in sorted(assignments.items()):
        preserved_report = final_id in preserved_ids
        community = aggregate_stats(
            final_id=final_id,
            original_ids=sorted(original_ids),
            stats_by_id=stats_by_id,
            merge_reason="already reported target with attached small communities"
            if preserved_report and len(original_ids) > 1
            else "preserved: already reported"
            if preserved_report
            else "new higher-level report community",
            preserved=preserved_report,
        )
        payload = asdict(community)
        payload["report_required"] = final_id in report_required_ids
        payload["attached_to_existing_report"] = preserved_report and len(original_ids) > 1
        payload["newly_attached_original_ids"] = [
            community_id for community_id in sorted(original_ids) if community_id != final_id
        ]
        final_communities[final_id] = payload

    report_ready = {
        final_id: data["actors"]
        for final_id, data in final_communities.items()
        if data["report_required"]
    }
    return final_communities, report_ready, merge_records


def edge_weight(graph: nx.Graph, left: str, right: str) -> float:
    """Return a numeric edge weight, defaulting to one."""

    return float(graph[left][right].get("weight", 1.0))


def weighted_volume(graph: nx.Graph, nodes: set[str]) -> float:
    """Compute weighted graph volume for a node set."""

    return sum(edge_weight(graph, node, neighbor) for node in nodes for neighbor in graph.neighbors(node))


def weighted_cut(graph: nx.Graph, nodes: set[str]) -> float:
    """Compute weighted cut size between a node set and the rest of the graph."""

    return sum(
        edge_weight(graph, node, neighbor)
        for node in nodes
        for neighbor in graph.neighbors(node)
        if neighbor not in nodes
    )


def graph_conductance(graph: nx.Graph, nodes: set[str]) -> float:
    """Compute conductance for a node set inside a weighted graph."""

    if not nodes or len(nodes) == graph.number_of_nodes():
        return 0.0
    rest = set(graph.nodes()) - nodes
    denominator = min(weighted_volume(graph, nodes), weighted_volume(graph, rest))
    return weighted_cut(graph, nodes) / denominator if denominator else 0.0


def actor_nodes_for_community(graph: nx.Graph, actors: list[str]) -> set[str]:
    """Map actor slugs from JSON back to graph node URIs."""

    return {
        f"{ACTOR_RESOURCE_PREFIX}{actor}"
        for actor in actors
        if f"{ACTOR_RESOURCE_PREFIX}{actor}" in graph
    }


def compute_graph_metrics(
    final_communities: dict[str, dict[str, Any]],
    original_communities: dict[str, list[str]],
    config: AttachMergeConfig,
) -> dict[str, Any]:
    """Compute modularity and conductance diagnostics for original and final partitions."""

    if not config.graph_path.exists():
        return {"error": f"Graph file not found: {config.graph_path}"}

    graph = nx.read_gexf(config.graph_path)
    final_partition = [
        actor_nodes_for_community(graph, data["actors"])
        for data in final_communities.values()
    ]
    final_partition = [nodes for nodes in final_partition if nodes]
    covered = set().union(*final_partition) if final_partition else set()
    final_full_partition = final_partition + [{node} for node in set(graph.nodes()) - covered]
    final_conductances = [graph_conductance(graph, nodes) for nodes in final_partition]

    original_partition = [
        actor_nodes_for_community(graph, actors)
        for actors in original_communities.values()
    ]
    original_partition = [nodes for nodes in original_partition if nodes]
    original_covered = set().union(*original_partition) if original_partition else set()
    original_full_partition = original_partition + [
        {node} for node in set(graph.nodes()) - original_covered
    ]
    original_conductances = [graph_conductance(graph, nodes) for nodes in original_partition]

    new_report_conductance = {
        community_id: graph_conductance(graph, actor_nodes_for_community(graph, data["actors"]))
        for community_id, data in final_communities.items()
        if data["report_required"]
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
        "new_report_community_conductance": {
            community_id: round(value, 6)
            for community_id, value in new_report_conductance.items()
        },
        "average_new_report_conductance": mean(new_report_conductance.values())
        if new_report_conductance
        else None,
    }


def compact_counter(values: dict[str, int], limit: int = 5) -> dict[str, int]:
    """Return the top values from a counter-like dictionary."""

    return dict(Counter(values).most_common(limit))


def build_diagnostics(
    original_count: int,
    final_communities: dict[str, dict[str, Any]],
    report_ready: dict[str, list[str]],
    preserved_ids: set[str],
    stats_by_id: dict[str, CommunityStats],
    merge_records: list[dict[str, Any]],
    graph_metrics: dict[str, Any],
    config: AttachMergeConfig,
) -> dict[str, Any]:
    """Build a diagnostics report for preserved-target attachment."""

    attached_to_preserved = [
        record
        for record in merge_records
        if record.get("target_report_status") == "already_reported"
    ]
    new_report_diagnostics = {
        community_id: {
            "original_community_count": len(data["original_community_ids"]),
            "actor_count": data["actor_count"],
            "event_count": data["event_count"],
            "first_date": data["first_date"],
            "last_date": data["last_date"],
            "top_countries": compact_counter(data["countries"]),
            "top_event_types": compact_counter(data["event_types"]),
            "top_sub_event_types": compact_counter(data["sub_event_types"]),
        }
        for community_id, data in final_communities.items()
        if data["report_required"]
    }

    reports_avoided = original_count - len(preserved_ids) - len(report_ready)
    return {
        "strategy": "small communities may attach to already reported top communities",
        "original_number_of_communities": original_count,
        "final_number_of_analysis_communities": len(final_communities),
        "already_reported_preserved_communities": len(preserved_ids),
        "new_reports_required": len(report_ready),
        "reports_avoided_after_first_30": reports_avoided,
        "small_communities_attached_to_existing_reports": len(attached_to_preserved),
        "small_remaining_communities": sum(
            1
            for community_id, stats in stats_by_id.items()
            if community_id not in preserved_ids and is_small_community(stats, config)
        ),
        "thresholds": {
            "preserve_top_n": config.preserve_top_n,
            "min_events_per_report": config.min_events_per_report,
            "min_actors_per_report": config.min_actors_per_report,
            "similarity_threshold": config.similarity_threshold,
            "attach_to_preserved_threshold": config.attach_to_preserved_threshold,
            "meta_leiden_threshold": config.meta_leiden_threshold,
        },
        "query_errors": {
            community_id: stats.query_error
            for community_id, stats in stats_by_id.items()
            if stats.query_error
        },
        "new_report_communities": new_report_diagnostics,
        "merge_reasons": merge_records,
        "graph_metrics": graph_metrics,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the preserved-target merge variant."""

    parser = argparse.ArgumentParser(
        description="Attach small Leiden communities to top preserved reports when similar."
    )
    parser.add_argument("--input", type=Path, default=AttachMergeConfig.input_path)
    parser.add_argument("--output", type=Path, default=AttachMergeConfig.output_path)
    parser.add_argument("--diagnostics", type=Path, default=AttachMergeConfig.diagnostics_path)
    parser.add_argument("--report-ready", type=Path, default=AttachMergeConfig.report_ready_path)
    parser.add_argument("--graph", type=Path, default=AttachMergeConfig.graph_path)
    parser.add_argument("--graphdb-url", default=AttachMergeConfig.graphdb_url)
    parser.add_argument("--preserve-top-n", type=int, default=AttachMergeConfig.preserve_top_n)
    parser.add_argument("--min-events-per-report", type=int, default=AttachMergeConfig.min_events_per_report)
    parser.add_argument("--min-actors-per-report", type=int, default=AttachMergeConfig.min_actors_per_report)
    parser.add_argument("--similarity-threshold", type=float, default=AttachMergeConfig.similarity_threshold)
    parser.add_argument("--attach-threshold", type=float, default=AttachMergeConfig.attach_to_preserved_threshold)
    parser.add_argument("--meta-leiden-threshold", type=float, default=AttachMergeConfig.meta_leiden_threshold)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AttachMergeConfig:
    """Create an AttachMergeConfig from parsed CLI arguments."""

    return AttachMergeConfig(
        input_path=args.input,
        output_path=args.output,
        diagnostics_path=args.diagnostics,
        report_ready_path=args.report_ready,
        graph_path=args.graph,
        graphdb_url=args.graphdb_url,
        preserve_top_n=args.preserve_top_n,
        min_events_per_report=args.min_events_per_report,
        min_actors_per_report=args.min_actors_per_report,
        similarity_threshold=args.similarity_threshold,
        attach_to_preserved_threshold=args.attach_threshold,
        meta_leiden_threshold=args.meta_leiden_threshold,
    )


def main() -> None:
    """Run the preserved-target attachment workflow."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    config = config_from_args(args)

    original_communities = load_communities(config.input_path)
    LOGGER.info("Loaded %s original Leiden communities", len(original_communities))
    stats_by_id = fetch_all_stats(original_communities, config)
    preserved_ids = set(sorted_by_event_count(stats_by_id)[: config.preserve_top_n])
    LOGGER.info("Using %s already reported communities as attachable targets", len(preserved_ids))

    meta_graph = build_meta_graph(stats_by_id, preserved_ids, config)
    final_communities, report_ready, merge_records = merge_with_preserved_targets(
        stats_by_id,
        preserved_ids,
        meta_graph,
        config,
    )
    graph_metrics = compute_graph_metrics(final_communities, original_communities, config)
    diagnostics = build_diagnostics(
        original_count=len(original_communities),
        final_communities=final_communities,
        report_ready=report_ready,
        preserved_ids=preserved_ids,
        stats_by_id=stats_by_id,
        merge_records=merge_records,
        graph_metrics=graph_metrics,
        config=config,
    )

    save_json(config.output_path, final_communities)
    save_json(config.report_ready_path, report_ready)
    save_json(config.diagnostics_path, diagnostics)

    print(f"Original communities: {len(original_communities)}")
    print(f"Already reported preserved communities: {len(preserved_ids)}")
    print(f"Final analysis communities: {len(final_communities)}")
    print(f"New reports required: {len(report_ready)}")
    print(f"Merged output: {config.output_path.resolve()}")
    print(f"Report-ready input: {config.report_ready_path.resolve()}")
    print(f"Diagnostics report: {config.diagnostics_path.resolve()}")


if __name__ == "__main__":
    main()
