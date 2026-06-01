"""
Visualizes the T-Box (Terminology Box / Schema) of an RDF Knowledge Graph.
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL


def extract_tbox_schema(graph: Graph) -> nx.DiGraph:
    """Extracts T-Box schema relations from an RDF graph.

    This function looks for OWL/RDFS schema definitions such as `rdfs:subClassOf`,
    `rdfs:domain`, and `rdfs:range` to build a directed graph representing
    the ontology's structure.

    Args:
        graph (Graph): An rdflib Graph instance containing the parsed RDF data.

    Returns:
        nx.DiGraph: A NetworkX directed graph containing the T-Box schema.
    """
    G = nx.DiGraph()

    # Extract class hierarchies (rdfs:subClassOf)
    for subj, pred, obj in graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(subj, URIRef) and isinstance(obj, URIRef):
            subj_name = subj.split('#')[-1].split('/')[-1]
            obj_name = obj.split('#')[-1].split('/')[-1]
            G.add_edge(subj_name, obj_name, label="subClassOf")

    # Extract object properties with domain and range
    for prop in graph.subjects(RDF.type, OWL.ObjectProperty):
        if isinstance(prop, URIRef):
            prop_name = prop.split('#')[-1].split('/')[-1]
            domains = list(graph.objects(prop, RDFS.domain))
            ranges = list(graph.objects(prop, RDFS.range))
            
            for d in domains:
                for r in ranges:
                    if isinstance(d, URIRef) and isinstance(r, URIRef):
                        d_name = d.split('#')[-1].split('/')[-1]
                        r_name = r.split('#')[-1].split('/')[-1]
                        
                        # Filtro difensivo: riconduce le sottoclassi alla macro-classe Actor
                        subclasses = ["StateForces", "RebelGroup", "PoliticalMilitia", 
                                      "IdentityMilitia", "Rioters", "Protesters", 
                                      "Civilian", "ExternalOther"]
                        if r_name in subclasses:
                            r_name = "Actor"
                            
                        # Se l'arco esiste già, concatena l'etichetta per non perderla
                        if G.has_edge(d_name, r_name):
                            existing_label = G[d_name][r_name].get("label", "")
                            if prop_name not in existing_label:
                                G[d_name][r_name]["label"] = f"{existing_label} /\n{prop_name}"
                        else:
                            G.add_edge(d_name, r_name, label=prop_name)

    return G


def plot_tbox_graph(G: nx.DiGraph, output_path: str = "tbox_visualization.png") -> None:
    """Generates and saves a visual representation of the T-Box graph.

    Args:
        G (nx.DiGraph): The NetworkX directed graph representing the T-Box.
        output_path (str): The file path where the plot image will be saved.
    """
    if G.number_of_nodes() == 0:
        print("Warning: The graph has no nodes. The provided RDF file might not contain any explicit T-Box definitions (like rdfs:subClassOf, rdfs:domain, rdfs:range).")
        return

    plt.figure(figsize=(14, 10))
    
    # Layout for the nodes
    pos = nx.spring_layout(G, seed=42, k=1.2)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=3500,
        node_color="#b5e2fa",
        edgecolors="#333333",
        linewidths=1.5
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color="#888888",
        arrows=True,
        arrowsize=20,
        width=2,
        connectionstyle="arc3,rad=0.1"
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight="bold",
        font_color="#333333"
    )
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="#d00000",
        label_pos=0.5
    )
    
    plt.title("Knowledge Graph Schema (T-Box)", fontsize=16, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"T-Box visualization successfully saved to: {os.path.abspath(output_path)}")
    plt.close()


def extract_critical_abox(graph: Graph, max_events: int = 15) -> tuple[nx.DiGraph, str]:
    """Extracts an A-Box ego-network centered on a critical actor.

    This function executes SPARQL queries to find the most active Actor 
    (highest degree in events), then retrieves a subgraph of a few events 
    involving this actor, including other actors and locations connected.

    Args:
        graph (Graph): The RDF graph containing A-Box data.
        max_events (int): Maximum number of events to extract for the ego-network.

    Returns:
        tuple[nx.DiGraph, str]: The NetworkX graph representing the A-Box sample,
            and the string label of the critical central node.
    """
    G = nx.DiGraph()
    
    # 1. Find the most critical node (Actor involved in the most events)
    query_critical = """
        PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
        SELECT ?actor (COUNT(?event) AS ?degree)
        WHERE {
            ?event a conf:ConflictEvent .
            { ?event conf:hasActor1 ?actor } UNION { ?event conf:hasActor2 ?actor }
        }
        GROUP BY ?actor
        ORDER BY DESC(?degree)
        LIMIT 1
    """
    
    critical_actor_uri = None
    for row in graph.query(query_critical):
        critical_actor_uri = str(row.actor)
        break

    if not critical_actor_uri:
        print("Warning: No ConflictEvents found in the graph. Cannot generate A-Box.")
        return G, ""
        
    critical_actor_name = critical_actor_uri.split('/')[-1]
    
    # 2. Extract a set of events for this critical actor
    query_event_list = f"""
        PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
        SELECT ?event
        WHERE {{
            ?event a conf:ConflictEvent .
            {{ ?event conf:hasActor1 <{critical_actor_uri}> }} 
            UNION 
            {{ ?event conf:hasActor2 <{critical_actor_uri}> }}
        }}
        LIMIT {max_events}
    """
    
    event_uris = [str(row.event) for row in graph.query(query_event_list)]
    
    if not event_uris:
        return G, critical_actor_name

    # 3. Extract the relations for these specific events
    values_clause = " ".join([f"<{uri}>" for uri in event_uris])
    query_relations = f"""
        PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
        SELECT ?event ?p ?o
        WHERE {{
            VALUES ?event {{ {values_clause} }}
            ?event ?p ?o .
            FILTER(?p IN (conf:hasActor1, conf:hasActor2, conf:locatedIn))
        }}
    """
    
    # Initialize the critical node
    G.add_node(f"Actor:\n{critical_actor_name}", node_type="critical")
    
    for row in graph.query(query_relations):
        event_uri = str(row.event)
        prop_uri = str(row.p)
        obj_uri = str(row.o)
        
        event_name = f"Event:\n{event_uri.split('/')[-1]}"
        prop_name = prop_uri.split('#')[-1]
        obj_name = obj_uri.split('/')[-1]
        
        G.add_node(event_name, node_type="event")
        
        # Determine object type based on property name
        if "hasActor" in prop_name:
            obj_label = f"Actor:\n{obj_name}"
            node_type = "critical" if obj_uri == critical_actor_uri else "actor"
            G.add_node(obj_label, node_type=node_type)
            G.add_edge(event_name, obj_label, label=prop_name)
        elif "locatedIn" in prop_name:
            obj_label = f"Location:\n{obj_name}"
            G.add_node(obj_label, node_type="location")
            G.add_edge(event_name, obj_label, label=prop_name)
            
    return G, critical_actor_name


def plot_abox_graph(G: nx.DiGraph, critical_node_name: str, output_path: str = "abox_visualization.png") -> None:
    """Generates and saves a visual representation of the A-Box ego-network.

    Args:
        G (nx.DiGraph): The NetworkX directed graph representing the A-Box.
        critical_node_name (str): The name of the central critical node.
        output_path (str): The file path where the plot image will be saved.
    """
    if G.number_of_nodes() == 0:
        return

    plt.figure(figsize=(16, 12))
    
    # Layout for the nodes
    pos = nx.spring_layout(G, seed=42, k=0.8)
    
    # Assign colors based on node_type
    colors = []
    for node, data in G.nodes(data=True):
        ntype = data.get("node_type", "")
        if ntype == "critical":
            colors.append("#ffb703") # Gold/Orange for critical node
        elif ntype == "event":
            colors.append("#ffcdb2") # Peach for events
        elif ntype == "actor":
            colors.append("#b5e2fa") # Light blue for other actors
        elif ntype == "location":
            colors.append("#e2ece9") # Light green/gray for locations
        else:
            colors.append("#cccccc")
            
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=4000,
        node_color=colors,
        edgecolors="#333333",
        linewidths=1.5
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color="#888888",
        arrows=True,
        arrowsize=15,
        width=2
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        font_weight="bold",
        font_color="#333333"
    )
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="#555555"
    )
    
    plt.title(f"A-Box Ego-Network centered on Critical Node: {critical_node_name}", fontsize=18, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"A-Box visualization successfully saved to: {os.path.abspath(output_path)}")
    plt.close()


def main() -> None:
    """Main execution function."""
    tbox_files = [
        "acled_ontologyCle.ttl",
        "cle/acled_ontologyCle.ttl"
    ]
    
    abox_files = [
        "acled_kg.ttl",
        "cle/acled_kg.ttl"
    ]
    
    # 1. Visualize T-Box
    tbox_file = next((f for f in tbox_files if os.path.exists(f)), None)
    if not tbox_file:
        tbox_file = next((f for f in abox_files if os.path.exists(f)), None)
        
    if tbox_file:
        print(f"\n--- T-BOX VISUALIZATION ---")
        print(f"Loading RDF graph for T-Box from: {tbox_file}")
        g_tbox = Graph()
        g_tbox.parse(tbox_file, format="turtle")
        print(f"Successfully parsed {len(g_tbox)} triples.")
        
        print("Extracting T-Box schema...")
        tbox_graph = extract_tbox_schema(g_tbox)
        print(f"Extracted {tbox_graph.number_of_nodes()} classes and {tbox_graph.number_of_edges()} relationships.")
        
        plot_tbox_graph(tbox_graph, "tbox_schema_visualization.png")
    else:
        print("\nError: Could not find any .ttl file for T-Box visualization.")
        
    # 2. Visualize A-Box (Ego-Network of a Critical Node)
    abox_file = next((f for f in abox_files if os.path.exists(f)), None)
    if abox_file:
        print(f"\n--- A-BOX EGO-NETWORK VISUALIZATION ---")
        print(f"Loading RDF graph for A-Box from: {abox_file} (This might take a minute depending on file size...)")
        g_abox = Graph()
        g_abox.parse(abox_file, format="turtle")
        print(f"Successfully parsed {len(g_abox)} triples.")
        
        print("Executing SPARQL query to find a critical node and extract its ego-network...")
        abox_graph, critical_node = extract_critical_abox(g_abox, max_events=15)
        
        if abox_graph.number_of_nodes() > 0:
            print(f"Extracted A-Box graph centered on '{critical_node}' with {abox_graph.number_of_nodes()} nodes and {abox_graph.number_of_edges()} relationships.")
            plot_abox_graph(abox_graph, critical_node, f"abox_ego_network_{critical_node}.png")
    else:
        print("\nWarning: Could not find 'acled_kg.ttl' for A-Box visualization. Please generate it using the kg_builder script.")


if __name__ == "__main__":
    main()
