"""
Analyze Leiden GraphRAG communities across ACLED event variables.

The input `leiden_graphrag_report_input.json` maps GraphRAG community ids to
actor slugs. This script joins those actors to `events.csv` and `actors.csv`,
then writes community-level summaries, long-form variable counts, actor
coverage tables, and a few diagnostic plots.

Usage:
    python analyze_leiden_communities.py
    python analyze_leiden_communities.py --top-n 15 --save-event-assignments
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMMUNITIES = SCRIPT_DIR / "leiden_graphrag_report_input.json"
DEFAULT_EVENTS = SCRIPT_DIR / "events.csv"
DEFAULT_ACTORS = SCRIPT_DIR / "actors.csv"
DEFAULT_DIAGNOSTICS = SCRIPT_DIR / "leiden_graphrag_community_report.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "leiden_community_analysis"

CATEGORICAL_COLUMNS = [
    "country",
    "iso",
    "disorder_type",
    "event_type",
    "sub_event_type",
    "inter1",
    "inter2",
    "source_scale",
    "location",
]

NUMERIC_COLUMNS = [
    "fatalities",
    "latitude",
    "longitude",
]

PROFILE_COLUMNS = [
    "country",
    "disorder_type",
    "event_type",
    "sub_event_type",
]


def slug_from_uri(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).rstrip("/").split("/")[-1]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_diagnostics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = load_json(path)
    return data.get("final_communities", {})


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def build_actor_community_maps(
    communities: dict[str, list[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    community_to_actors = {
        community_id: set(actors)
        for community_id, actors in communities.items()
    }
    actor_to_communities: dict[str, set[str]] = defaultdict(set)
    for community_id, actors in community_to_actors.items():
        for actor in actors:
            actor_to_communities[actor].add(community_id)
    return community_to_actors, actor_to_communities


def assign_events_to_communities(
    events: pd.DataFrame,
    actor_to_communities: dict[str, set[str]],
) -> pd.DataFrame:
    rows = []
    for row in events.itertuples(index=False):
        event = row._asdict()
        actor1_slug = event.get("actor1_slug", "")
        actor2_slug = event.get("actor2_slug", "")
        matched = set(actor_to_communities.get(actor1_slug, set()))
        matched.update(actor_to_communities.get(actor2_slug, set()))
        for community_id in sorted(matched):
            rows.append(
                {
                    "community_id": community_id,
                    "event_uri": event.get("event_uri"),
                    "event_id_cnty": event.get("event_id_cnty"),
                    "actor1_slug": actor1_slug,
                    "actor2_slug": actor2_slug,
                    **event,
                }
            )
    return pd.DataFrame(rows)


def summarize_communities(
    assigned: pd.DataFrame,
    community_to_actors: dict[str, set[str]],
    diagnostics: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for community_id, actors in community_to_actors.items():
        subset = assigned[assigned["community_id"] == community_id]
        diag = diagnostics.get(community_id, {})
        row = {
            "community_id": community_id,
            "actor_count": len(actors),
            "event_count": int(subset["event_uri"].nunique()) if not subset.empty else 0,
            "assigned_rows": int(len(subset)),
            "fatalities_sum": int(subset["fatalities"].fillna(0).sum()) if "fatalities" in subset else 0,
            "fatalities_mean": float(subset["fatalities"].fillna(0).mean()) if "fatalities" in subset and not subset.empty else 0.0,
            "first_date": subset["event_date_dt"].min().date().isoformat() if "event_date_dt" in subset and not subset.empty else "",
            "last_date": subset["event_date_dt"].max().date().isoformat() if "event_date_dt" in subset and not subset.empty else "",
            "country_count": int(subset["country"].nunique()) if "country" in subset and not subset.empty else 0,
            "event_type_count": int(subset["event_type"].nunique()) if "event_type" in subset and not subset.empty else 0,
            "sub_event_type_count": int(subset["sub_event_type"].nunique()) if "sub_event_type" in subset and not subset.empty else 0,
            "original_community_count": diag.get("original_community_count"),
            "seed_community_id": diag.get("seed_community_id"),
        }
        for column in ["country", "event_type", "sub_event_type", "disorder_type"]:
            if column in subset and not subset.empty:
                mode = subset[column].dropna().astype(str).value_counts()
                row[f"top_{column}"] = mode.index[0] if not mode.empty else ""
                row[f"top_{column}_share"] = float(mode.iloc[0] / len(subset)) if not mode.empty else 0.0
            else:
                row[f"top_{column}"] = ""
                row[f"top_{column}_share"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["event_count", "actor_count"], ascending=False)


def categorical_counts(assigned: pd.DataFrame, columns: list[str], top_n: int) -> pd.DataFrame:
    rows = []
    for community_id, subset in assigned.groupby("community_id"):
        total = len(subset)
        for column in columns:
            if column not in subset.columns:
                continue
            counts = subset[column].fillna("Unknown").astype(str).value_counts().head(top_n)
            for rank, (value, count) in enumerate(counts.items(), start=1):
                rows.append(
                    {
                        "community_id": community_id,
                        "variable": column,
                        "rank": rank,
                        "value": value,
                        "count": int(count),
                        "share": float(count / total) if total else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def yearly_counts(assigned: pd.DataFrame) -> pd.DataFrame:
    if "year" not in assigned:
        return pd.DataFrame()
    return (
        assigned.groupby(["community_id", "year"], dropna=False)
        .agg(event_count=("event_uri", "nunique"), fatalities_sum=("fatalities", "sum"))
        .reset_index()
        .sort_values(["community_id", "year"])
    )


def numeric_summary(assigned: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = ensure_columns(assigned, columns)
    if not available:
        return pd.DataFrame()
    summary = (
        assigned.groupby("community_id")[available]
        .agg(["count", "mean", "median", "sum", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(str(part) for part in column if part)
        for column in summary.columns
    ]
    return summary


def actor_summary(
    actors: pd.DataFrame,
    assigned: pd.DataFrame,
    community_to_actors: dict[str, set[str]],
) -> pd.DataFrame:
    actor_meta = actors.copy()
    actor_meta["actor_slug"] = actor_meta["actor_uri"].map(slug_from_uri)
    actor_meta = actor_meta.drop_duplicates("actor_slug").set_index("actor_slug")

    rows = []
    for community_id, slugs in community_to_actors.items():
        subset = assigned[assigned["community_id"] == community_id]
        actor1_counts = subset["actor1_slug"].value_counts() if not subset.empty else pd.Series(dtype=int)
        actor2_counts = subset["actor2_slug"].value_counts() if not subset.empty else pd.Series(dtype=int)
        for slug in sorted(slugs):
            meta = actor_meta.loc[slug].to_dict() if slug in actor_meta.index else {}
            events_as_actor1 = int(actor1_counts.get(slug, 0))
            events_as_actor2 = int(actor2_counts.get(slug, 0))
            rows.append(
                {
                    "community_id": community_id,
                    "actor_slug": slug,
                    "actor_name": meta.get("actor_name", ""),
                    "inter": meta.get("inter", ""),
                    "owl_class": meta.get("owl_class", ""),
                    "events_as_actor1": events_as_actor1,
                    "events_as_actor2": events_as_actor2,
                    "events_total": events_as_actor1 + events_as_actor2,
                }
            )
    return pd.DataFrame(rows).sort_values(["community_id", "events_total"], ascending=[True, False])


def pivot_top_values(counts: pd.DataFrame, variable: str, top_values: int) -> pd.DataFrame:
    subset = counts[counts["variable"] == variable]
    if subset.empty:
        return pd.DataFrame()
    values = subset.groupby("value")["count"].sum().nlargest(top_values).index
    subset = subset[subset["value"].isin(values)]
    return subset.pivot_table(
        index="community_id",
        columns="value",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )


def build_profile_matrix(assigned: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = ensure_columns(assigned, columns)
    frames = []
    for column in available:
        dummies = pd.get_dummies(
            assigned[["community_id", column]].assign(**{column: assigned[column].fillna("Unknown").astype(str)}),
            columns=[column],
            prefix=column,
            dtype=float,
        )
        frames.append(dummies.groupby("community_id").mean())
    if not frames:
        return pd.DataFrame({"community_id": sorted(assigned["community_id"].unique())})
    profile = pd.concat(frames, axis=1).fillna(0)
    profile.insert(0, "community_id", profile.index)
    return profile.reset_index(drop=True)


def compute_embedding(profile: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    feature_columns = [column for column in profile.columns if column != "community_id"]
    x = profile[feature_columns].to_numpy(dtype=float)
    if x.shape[0] < 2 or x.shape[1] < 2:
        embedding = np.zeros((x.shape[0], 2))
        method = "constant"
    else:
        try:
            from sklearn.preprocessing import StandardScaler

            x_scaled = StandardScaler().fit_transform(x)
        except ImportError:
            x_scaled = x - x.mean(axis=0)

        try:
            from umap import UMAP

            n_neighbors = min(5, max(2, x_scaled.shape[0] - 1))
            embedding = UMAP(
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=42,
            ).fit_transform(x_scaled)
            method = "umap"
        except ImportError:
            try:
                from sklearn.decomposition import PCA

                embedding = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
                method = "pca"
            except ImportError:
                _, _, vt = np.linalg.svd(x_scaled, full_matrices=False)
                components = vt[:2].T
                embedding = x_scaled @ components
                if embedding.shape[1] == 1:
                    embedding = np.column_stack([embedding[:, 0], np.zeros(len(embedding))])
                method = "svd"

    return pd.DataFrame(
        {
            "community_id": profile["community_id"].to_list(),
            "x": embedding[:, 0],
            "y": embedding[:, 1],
        }
    ), method


def color_map(values: pd.Series):
    unique = sorted(values.fillna("Unknown").astype(str).unique())
    cmap = plt_colormap("tab20", len(unique))
    return {value: cmap(index) for index, value in enumerate(unique)}


def plt_colormap(name: str, size: int):
    import matplotlib.pyplot as plt

    return plt.get_cmap(name, max(size, 1))


def save_centroid_map(
    embedding: pd.DataFrame,
    summary: pd.DataFrame,
    method: str,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    data = embedding.merge(summary, on="community_id", how="left")
    colors = color_map(data["top_country"])
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    disorder_values = sorted(data["top_disorder_type"].fillna("Unknown").astype(str).unique())
    marker_by_disorder = {
        value: markers[index % len(markers)]
        for index, value in enumerate(disorder_values)
    }
    max_events = max(float(data["event_count"].max()), 1.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    for _, row in data.iterrows():
        country = str(row.get("top_country", "Unknown"))
        disorder = str(row.get("top_disorder_type", "Unknown"))
        size = 90 + 650 * np.sqrt(float(row.get("event_count", 0)) / max_events)
        ax.scatter(
            row["x"],
            row["y"],
            s=size,
            color=colors.get(country),
            marker=marker_by_disorder.get(disorder, "o"),
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.85,
        )
        ax.text(row["x"], row["y"], row["community_id"].replace("graphrag_", ""), fontsize=8)

    country_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=country, markersize=8)
        for country, color in colors.items()
    ]
    disorder_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="#333333",
            linestyle="None",
            label=disorder,
            markersize=7,
        )
        for disorder, marker in marker_by_disorder.items()
    ]
    first_legend = ax.legend(handles=country_handles, title="Dominant country", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.add_artist(first_legend)
    ax.legend(handles=disorder_handles, title="Dominant disorder", loc="lower left", bbox_to_anchor=(1.02, 0))
    ax.set_title(f"Community centroid profile map ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "community_centroid_profile_map.png", dpi=180)
    plt.close(fig)


def save_bipartite_profile_graph(
    counts: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    top_attributes_per_variable: int = 2,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("networkx/matplotlib not installed; skipping bipartite profile graph")
        return

    top_communities = summary.head(top_n)["community_id"].tolist()
    selected_counts = counts[
        counts["community_id"].isin(top_communities)
        & counts["variable"].isin(PROFILE_COLUMNS)
        & (counts["rank"] <= top_attributes_per_variable)
    ]

    graph = nx.Graph()
    summary_by_id = summary.set_index("community_id")
    colors = color_map(summary["top_country"])
    for community_id in top_communities:
        row = summary_by_id.loc[community_id]
        graph.add_node(
            community_id,
            bipartite="community",
            event_count=float(row["event_count"]),
            top_country=str(row["top_country"]),
        )

    for row in selected_counts.itertuples(index=False):
        attribute_id = f"{row.variable}: {row.value}"
        graph.add_node(attribute_id, bipartite="attribute")
        graph.add_edge(row.community_id, attribute_id, weight=float(row.share))

    if graph.number_of_edges() == 0:
        return

    community_nodes = [node for node, data in graph.nodes(data=True) if data["bipartite"] == "community"]
    attribute_nodes = [node for node, data in graph.nodes(data=True) if data["bipartite"] == "attribute"]
    pos = nx.bipartite_layout(graph, community_nodes, align="horizontal", scale=2)

    fig, ax = plt.subplots(figsize=(16, 10))
    edge_widths = [1.0 + 8.0 * graph.edges[edge]["weight"] for edge in graph.edges]
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.35, edge_color="#555555", ax=ax)

    max_events = max([graph.nodes[node]["event_count"] for node in community_nodes] or [1.0])
    community_sizes = [
        600 + 2200 * np.sqrt(graph.nodes[node]["event_count"] / max_events)
        for node in community_nodes
    ]
    community_colors = [
        colors.get(graph.nodes[node]["top_country"], "#999999")
        for node in community_nodes
    ]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=community_nodes,
        node_size=community_sizes,
        node_color=community_colors,
        edgecolors="#222222",
        linewidths=0.8,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=attribute_nodes,
        node_size=420,
        node_color="#eeeeee",
        edgecolors="#666666",
        linewidths=0.5,
        ax=ax,
    )
    labels = {
        node: node.replace("graphrag_", "")
        if node in community_nodes
        else node
        for node in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title("Bipartite community-attribute profile graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "community_attribute_bipartite_graph.png", dpi=180)
    plt.close(fig)


def save_plots(
    summary: pd.DataFrame,
    counts: pd.DataFrame,
    yearly: pd.DataFrame,
    profile: pd.DataFrame,
    embedding: pd.DataFrame,
    embedding_method: str,
    output_dir: Path,
    top_n: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_centroid_map(embedding, summary, embedding_method, plots_dir)
    save_bipartite_profile_graph(counts, summary, plots_dir, top_n)

    top_summary = summary.head(top_n).set_index("community_id")

    fig, ax = plt.subplots(figsize=(12, 6))
    top_summary["event_count"].sort_values().plot(kind="barh", ax=ax, color="#4c78a8")
    ax.set_title(f"Top {top_n} communities by event count")
    ax.set_xlabel("Events")
    ax.set_ylabel("Community")
    fig.tight_layout()
    fig.savefig(plots_dir / "top_communities_by_events.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    top_summary["fatalities_sum"].sort_values().plot(kind="barh", ax=ax, color="#b279a2")
    ax.set_title(f"Top {top_n} communities by fatalities")
    ax.set_xlabel("Fatalities")
    ax.set_ylabel("Community")
    fig.tight_layout()
    fig.savefig(plots_dir / "top_communities_by_fatalities.png", dpi=180)
    plt.close(fig)

    for variable in ["country", "event_type", "sub_event_type", "disorder_type"]:
        pivot = pivot_top_values(counts, variable, top_values=min(top_n, 12))
        if pivot.empty:
            continue
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(top_n).index]
        row_sums = pivot.sum(axis=1).replace(0, pd.NA)
        pivot_norm = pivot.div(row_sums, axis=0).fillna(0)
        fig, ax = plt.subplots(figsize=(14, 7))
        im = ax.imshow(pivot_norm.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_title(f"Community distribution by {variable}")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(im, ax=ax, label="Row-normalized proportion")
        fig.tight_layout()
        fig.savefig(plots_dir / f"heatmap_{variable}.png", dpi=180)
        plt.close(fig)

    if not yearly.empty:
        top_ids = summary.head(min(8, top_n))["community_id"].tolist()
        yearly_top = yearly[yearly["community_id"].isin(top_ids)]
        pivot = yearly_top.pivot_table(
            index="year",
            columns="community_id",
            values="event_count",
            aggfunc="sum",
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(13, 7))
        pivot.plot(ax=ax)
        ax.set_title("Yearly event counts for largest communities")
        ax.set_xlabel("Year")
        ax.set_ylabel("Events")
        fig.tight_layout()
        fig.savefig(plots_dir / "yearly_events_largest_communities.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Leiden GraphRAG communities.")
    parser.add_argument("--communities", type=Path, default=DEFAULT_COMMUNITIES)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--actors", type=Path, default=DEFAULT_ACTORS)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument(
        "--save-event-assignments",
        action="store_true",
        help="Also save the potentially large community-event assignment table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading communities...")
    communities = load_json(args.communities)
    diagnostics = load_diagnostics(args.diagnostics)
    community_to_actors, actor_to_communities = build_actor_community_maps(communities)

    print("Loading events and actors...")
    events = pd.read_csv(args.events)
    actors = pd.read_csv(args.actors)

    events["actor1_slug"] = events["actor1_uri"].map(slug_from_uri)
    events["actor2_slug"] = events["actor2_uri"].map(slug_from_uri)
    events["event_date_dt"] = pd.to_datetime(events["event_date"], errors="coerce")
    events["year"] = events["event_date_dt"].dt.year
    events["month"] = events["event_date_dt"].dt.to_period("M").astype(str)

    print("Assigning events to communities...")
    assigned = assign_events_to_communities(events, actor_to_communities)
    if assigned.empty:
        raise RuntimeError("No events matched the community actors.")

    categorical_columns = ensure_columns(assigned, CATEGORICAL_COLUMNS + ["year", "month"])
    numeric_columns = ensure_columns(assigned, NUMERIC_COLUMNS)

    print("Building summaries...")
    summary = summarize_communities(assigned, community_to_actors, diagnostics)
    counts = categorical_counts(assigned, categorical_columns, args.top_n)
    yearly = yearly_counts(assigned)
    numeric = numeric_summary(assigned, numeric_columns)
    actors_out = actor_summary(actors, assigned, community_to_actors)
    profile = build_profile_matrix(assigned, PROFILE_COLUMNS)
    embedding, embedding_method = compute_embedding(profile)
    embedding_out = embedding.merge(summary, on="community_id", how="left")

    summary.to_csv(args.output_dir / "community_summary.csv", index=False)
    counts.to_csv(args.output_dir / "community_variable_counts.csv", index=False)
    yearly.to_csv(args.output_dir / "community_yearly_counts.csv", index=False)
    numeric.to_csv(args.output_dir / "community_numeric_summary.csv", index=False)
    actors_out.to_csv(args.output_dir / "community_actor_summary.csv", index=False)
    profile.to_csv(args.output_dir / "community_profile_matrix.csv", index=False)
    embedding_out.to_csv(args.output_dir / "community_centroid_embedding.csv", index=False)

    if args.save_event_assignments:
        assigned.to_csv(args.output_dir / "community_event_assignments.csv", index=False)

    metadata = {
        "communities": len(community_to_actors),
        "unique_community_actors": len(actor_to_communities),
        "events": int(events["event_uri"].nunique()),
        "assigned_rows": int(len(assigned)),
        "assigned_unique_events": int(assigned["event_uri"].nunique()),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "profile_columns": PROFILE_COLUMNS,
        "embedding_method": embedding_method,
    }
    with (args.output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("Saving plots...")
    save_plots(summary, counts, yearly, profile, embedding, embedding_method, args.output_dir, args.top_n)

    print(f"Done. Outputs saved to: {args.output_dir.resolve()}")
    print(summary.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
