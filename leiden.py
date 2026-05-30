import json
import os
import requests
import networkx as nx
import matplotlib.pyplot as plt
from cdlib import algorithms, evaluation

def load_graph_from_graphdb_and_run_leiden(repo_url, output_json_path):
    print(f"--- Start data extraction from GraphDB: {repo_url} ---")

    # 1. SPARQL query to extract the co-occurrence network of actors in events
    # We use the conf:hasActor1 and conf:hasActor2 predicates as defined in kg_builder.py
    query = """
    PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
    
    SELECT ?event ?actor WHERE {
        ?event a conf:ConflictEvent .
        { ?event conf:hasActor1 ?actor }
        UNION
        { ?event conf:hasActor2 ?actor }
    }
    """
    
    headers = {
        'Accept': 'application/sparql-results+json',
        'Content-Type': 'application/sparql-query'
    }
    
    print("Executing SPARQL query on GraphDB...")
    response = requests.post(repo_url, data=query, headers=headers)
    
    if response.status_code != 200:
        print(f"Error during extraction from GraphDB (Code {response.status_code}): {response.text}")
        return
        
    bindings = response.json().get('results', {}).get('bindings', [])
    print(f"Retrieved {len(bindings)} event-actor associations.")

    # Build a temporary bipartite graph (Actor -> Event)
    event_actor_map = {}
    for row in bindings:
        event_uri = row['event']['value']
        actor_uri = row['actor']['value']
        
        if event_uri not in event_actor_map:
            event_actor_map[event_uri] = []
        event_actor_map[event_uri].append(actor_uri)

    # 2. Creation of the Weighted Graph with NetworkX
    print("Building the actors network with NetworkX...")
    G = nx.Graph()

    for event, actors in event_actor_map.items():
        # Remove duplicates in the same event
        unique_actors = list(set(actors))
        if len(unique_actors) > 1:
            # If there are multiple actors in the same event, create/update an edge between them (co-occurrence)
            for i in range(len(unique_actors)):
                for j in range(i + 1, len(unique_actors)):
                    u, v = unique_actors[i], unique_actors[j]
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)
        elif len(unique_actors) == 1:
            # If there is only one actor, add it as an isolated node
            G.add_node(unique_actors[0])

    print(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    if G.number_of_nodes() == 0:
        print("Error: The NetworkX graph is empty. Check that the data has been loaded into GraphDB.")
        return

    # 3. Execution of the Leiden algorithm via CDLIB
    print("Executing the Leiden algorithm...")
    leiden_partition = algorithms.leiden(G, weights='weight')
    
    # Evaluation Metrics
    mod = evaluation.erdos_renyi_modularity(G, leiden_partition)
    print(f"Global Modularity: {mod.score:.4f}")

    cond = evaluation.conductance(G, leiden_partition)
    print(f"Average Conductance (lower is better): {cond.score:.4f}")

    # 4. Saving the result in JSON (Staging for GraphRAG)
    communities_dict = {}
    node_community_map = {}
    for i, community in enumerate(leiden_partition.communities):
        for node in community:
            node_community_map[node] = i
            
        # Clean the URIs to keep only the final slug of the actor's name
        clean_actors = [actor.split("/")[-1] for actor in community]
        communities_dict[f"community_{i}"] = clean_actors

    print(f"Leiden identified {len(communities_dict)} distinct macro-communities.")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(communities_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Communities successfully saved in: {os.path.abspath(output_json_path)}")

    # 5. Visualization (Matplotlib)
    print("Generating static visualization...")
    colors = [node_community_map[node] for node in G.nodes()]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50, cmap=plt.cm.jet)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    plt.title("Leiden Clusters Visualization on ACLED data")
    plt.savefig("leiden_clusters.png", dpi=300)
    plt.close()
    print(f"Static visualization saved in: {os.path.abspath('leiden_clusters.png')}")

    # 6. Export for Gephi
    print("Exporting network for Gephi...")
    for node in G.nodes():
        G.nodes[node]['community'] = node_community_map[node]

    nx.write_gexf(G, "acled_leiden_network.gexf")
    print(f"Gephi network saved in: {os.path.abspath('acled_leiden_network.gexf')}")

if __name__ == "__main__":
    # URL of the GraphDB repository 
    # Replace 'MiddleEastConflict' if you used a different ID in the db
    GRAPHDB_REPO_URL = "http://localhost:7200/repositories/MiddleEastConflict"
    
    # Path of the output JSON file
    OUTPUT_JSON_FILE = "leiden_communities.json"
    
    load_graph_from_graphdb_and_run_leiden(GRAPHDB_REPO_URL, OUTPUT_JSON_FILE)