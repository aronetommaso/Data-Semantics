import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

def plot_ego_abox(df, entity_name, entity_type, output_path, max_events=20):
    """Genera una rete A-Box (Ego Network) centrata su uno specifico nodo critico."""
    print(f"  -> Generating A-Box visualization for {entity_name}...")
    if entity_type == 'location':
        subset = df[df['location'] == entity_name].head(max_events)
    elif entity_type == 'actor1':
        subset = df[df['actor1'] == entity_name].head(max_events)
    elif entity_type == 'actor2':
        subset = df[df['actor2'] == entity_name].head(max_events)
    else:
        subset = df[(df['actor1'] == entity_name) | (df['actor2'] == entity_name)].head(max_events)
        
    G = nx.DiGraph()
    
    for _, row in subset.iterrows():
        event_node = f"Event:\n{row['event_id_cnty']}"
        G.add_node(event_node)
        
        if not pd.isna(row.get('actor1')):
            a1_node = f"Actor:\n{str(row['actor1'])[:20]}..."
            G.add_edge(event_node, a1_node, label='initiatedBy')
            
        if not pd.isna(row.get('actor2')):
            a2_node = f"Actor:\n{str(row['actor2'])[:20]}..."
            G.add_edge(event_node, a2_node, label='targetedActor')
            
        if not pd.isna(row.get('location')):
            loc_node = f"Loc:\n{str(row['location'])[:20]}"
            G.add_edge(event_node, loc_node, label='locatedIn')

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.8)
    
    # Colora i nodi: Evidenzia in "oro" il nodo critico centrale, usa i colori pastello per gli altri
    colors = ['#ffcdb2' if 'Event:' in n else '#ffb703' if str(entity_name)[:15] in n else '#b5e2fa' if 'Actor:' in n else '#e2ece9' for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color=colors, edgecolors='#333333')
    nx.draw_networkx_edges(G, pos, edge_color='#888888', arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='#555555')
    plt.title(f"A-Box Ego Network: {entity_name} ({entity_type.upper()})\n(Sample of {max_events} events)", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def identify_critical_nodes(file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return
        
    df = pd.read_csv(file_path)
    print("Analyzing Critical Nodes...")

    # --- 1. Top Locations (Most Conflicts) ---
    plt.figure(figsize=(12, 6))
    top_locs = df['location'].value_counts().head(10)
    sns.barplot(x=top_locs.values, y=top_locs.index, palette='Reds_r')
    plt.title('Top 10 Critical Locations (Most Conflict Events)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Events')
    plt.ylabel('Location')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_locations.png'), dpi=300)
    plt.close()
    
    top_loc_name = top_locs.index[0]
    safe_name = "".join([c if c.isalnum() else "_" for c in str(top_loc_name)])
    plot_ego_abox(df, top_loc_name, 'location', os.path.join(output_dir, f'abox_loc_{safe_name}.png'))

    # --- 2. Top Initiating Actors (Actor 1) ---
    plt.figure(figsize=(12, 6))
    top_actor1 = df['actor1'].value_counts().head(10)
    sns.barplot(x=top_actor1.values, y=top_actor1.index, palette='Oranges_r')
    plt.title('Top 10 Most Active Initiating Actors (Attacking the most)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Events Initiated')
    plt.ylabel('Actor')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_initiating_actors.png'), dpi=300)
    plt.close()
    
    top_a1_name = top_actor1.index[0]
    safe_name = "".join([c if c.isalnum() else "_" for c in str(top_a1_name)])
    plot_ego_abox(df, top_a1_name, 'actor1', os.path.join(output_dir, f'abox_initiator_{safe_name}.png'))

    # --- 3. Top Targeted Actors (Actor 2) ---
    plt.figure(figsize=(12, 6))
    top_actor2 = df['actor2'].value_counts().head(10)
    sns.barplot(x=top_actor2.values, y=top_actor2.index, palette='Blues_r')
    plt.title('Top 10 Most Targeted Actors (Defending/Attacked the most)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Events Targeted')
    plt.ylabel('Actor')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_targeted_actors.png'), dpi=300)
    plt.close()
    
    top_a2_name = top_actor2.index[0]
    safe_name = "".join([c if c.isalnum() else "_" for c in str(top_a2_name)])
    plot_ego_abox(df, top_a2_name, 'actor2', os.path.join(output_dir, f'abox_target_{safe_name}.png'))

    # --- 4. Most Lethal Actors (Sum of Fatalities where Actor 1 initiated) ---
    plt.figure(figsize=(12, 6))
    most_lethal = df.groupby('actor1')['fatalities'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=most_lethal.values, y=most_lethal.index, palette='Greys_r')
    plt.title('Top 10 Most Lethal Actors (Total Fatalities Caused)', fontsize=14, fontweight='bold')
    plt.xlabel('Total Fatalities')
    plt.ylabel('Actor')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_lethal_actors.png'), dpi=300)
    plt.close()
    
    top_lethal_name = most_lethal.index[0]
    safe_name = "".join([c if c.isalnum() else "_" for c in str(top_lethal_name)])
    plot_ego_abox(df, top_lethal_name, 'actor1', os.path.join(output_dir, f'abox_lethal_{safe_name}.png'))

    # --- 5. Graph Analysis: Degree Centrality ---
    print("Building Network Graph for Centrality Analysis...")
    G = nx.from_pandas_edgelist(df.dropna(subset=['actor1', 'actor2']), source='actor1', target='actor2')
    
    # Compute degree centrality to find the most interconnected actors in the conflict graph
    centrality = nx.degree_centrality(G)
    top_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    cent_actors, cent_scores = zip(*top_centrality)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(cent_scores), y=list(cent_actors), palette='Purples_r')
    plt.title('Top 10 Critical Actors by Graph Degree Centrality (Most Interconnected)', fontsize=14, fontweight='bold')
    plt.xlabel('Degree Centrality')
    plt.ylabel('Actor Node')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_centrality_actors.png'), dpi=300)
    plt.close()
    
    top_central_name = cent_actors[0]
    safe_name = "".join([c if c.isalnum() else "_" for c in str(top_central_name)])
    plot_ego_abox(df, top_central_name, 'actor', os.path.join(output_dir, f'abox_central_{safe_name}.png'))

    print(f"Success! All critical node visualizations have been saved in: {output_dir}")

if __name__ == "__main__":
    file_path = "c:/Users/tomma/Desktop/magistrale/second_semester/Data_Semantic_Project/acled_unified_middle_east.csv"
    output_directory = "c:/Users/tomma/Desktop/magistrale/second_semester/Data_Semantic_Project/critical_nodes_plots"
    identify_critical_nodes(file_path, output_directory)
