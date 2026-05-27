import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import urllib.parse
import os

try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, OWL, XSD
except ImportError:
    raise ImportError("Please install rdflib using: pip install rdflib")

# ==========================================
# NAMESPACES (Prefixes)
# ==========================================
CONF = Namespace("http://metadataregistry.org/conflict/ontology/")
RES = Namespace("http://metadataregistry.org/conflict/resource/")
SCHEMA = Namespace("https://schema.org/")
DBO = Namespace("http://dbpedia.org/ontology/")
GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")

def clean_uri(text):
    """Cleans a string to be safely used as a URI component."""
    if pd.isna(text) or not str(text).strip():
        return None
    clean_str = str(text).strip().replace(" ", "_").replace('"', '').replace("'", "")
    return urllib.parse.quote(clean_str, safe="_")

def build_tbox(g):
    """
    Defines the Semantic Classes (T-Box) and Properties 
    aligning them to global ontologies (Schema.org, DBpedia).
    """
    print("Building Ontology Schema (T-Box)...")
    
    # --- CLASSES ---
    # ConflictEvent
    g.add((CONF.ConflictEvent, RDF.type, OWL.Class))
    g.add((CONF.ConflictEvent, RDFS.subClassOf, SCHEMA.Event))
    g.add((CONF.ConflictEvent, RDFS.label, Literal("Conflict Event", lang="en")))
    g.add((CONF.ConflictEvent, RDFS.comment, Literal("Central node representing a single geopolitical event recorded by ACLED.", lang="en")))

    # Battle
    g.add((CONF.Battle, RDF.type, OWL.Class))
    g.add((CONF.Battle, RDFS.subClassOf, CONF.ConflictEvent))
    g.add((CONF.Battle, OWL.equivalentClass, DBO.MilitaryConflict))

    # Protest
    g.add((CONF.Protest, RDF.type, OWL.Class))
    g.add((CONF.Protest, RDFS.subClassOf, CONF.ConflictEvent))
    g.add((CONF.Protest, OWL.equivalentClass, DBO.Protest))

    # Actor
    g.add((CONF.Actor, RDF.type, OWL.Class))
    g.add((CONF.Actor, RDFS.subClassOf, SCHEMA.Organization))
    g.add((CONF.Actor, RDFS.subClassOf, DBO.Agent))
    g.add((CONF.Actor, RDFS.label, Literal("Geopolitical Actor", lang="en")))

    # GeographicLocation
    g.add((CONF.GeographicLocation, RDF.type, OWL.Class))
    g.add((CONF.GeographicLocation, OWL.equivalentClass, SCHEMA.Place))
    g.add((CONF.GeographicLocation, OWL.equivalentClass, DBO.Place))

    # --- OBJECT PROPERTIES ---
    # hasSubnationalActor
    g.add((CONF.hasSubnationalActor, RDF.type, OWL.ObjectProperty))
    g.add((CONF.hasSubnationalActor, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.hasSubnationalActor, RDFS.range, CONF.Actor))
    g.add((CONF.hasSubnationalActor, RDFS.subPropertyOf, DBO.participant))

    # initiatedBy
    g.add((CONF.initiatedBy, RDF.type, OWL.ObjectProperty))
    g.add((CONF.initiatedBy, RDFS.subPropertyOf, CONF.hasSubnationalActor))
    g.add((CONF.initiatedBy, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.initiatedBy, RDFS.range, CONF.Actor))

    # targetedActor
    g.add((CONF.targetedActor, RDF.type, OWL.ObjectProperty))
    g.add((CONF.targetedActor, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.targetedActor, RDFS.range, CONF.Actor))

    # locatedIn
    g.add((CONF.locatedIn, RDF.type, OWL.ObjectProperty))
    g.add((CONF.locatedIn, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.locatedIn, RDFS.range, CONF.GeographicLocation))
    g.add((CONF.locatedIn, RDFS.subPropertyOf, SCHEMA.location))

    # --- DATA PROPERTIES ---
    g.add((CONF.eventDate, RDF.type, OWL.DatatypeProperty))
    g.add((CONF.eventDate, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.eventDate, RDFS.range, XSD.date))
    g.add((CONF.eventDate, RDFS.subPropertyOf, SCHEMA.startDate))

    g.add((CONF.fatalities, RDF.type, OWL.DatatypeProperty))
    g.add((CONF.fatalities, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.fatalities, RDFS.range, XSD.nonNegativeInteger))

    g.add((CONF.latitude, RDF.type, OWL.DatatypeProperty))
    g.add((CONF.latitude, RDFS.domain, CONF.GeographicLocation))
    g.add((CONF.latitude, RDFS.range, XSD.float))
    g.add((CONF.latitude, OWL.equivalentProperty, GEO.lat))

    g.add((CONF.longitude, RDF.type, OWL.DatatypeProperty))
    g.add((CONF.longitude, RDFS.domain, CONF.GeographicLocation))
    g.add((CONF.longitude, RDFS.range, XSD.float))
    g.add((CONF.longitude, OWL.equivalentProperty, GEO.long))

    g.add((CONF.narrativeNotes, RDF.type, OWL.DatatypeProperty))
    g.add((CONF.narrativeNotes, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.narrativeNotes, RDFS.range, XSD.string))
    g.add((CONF.narrativeNotes, RDFS.subPropertyOf, SCHEMA.description))

def populate_abox(df, g):
    """
    Populates the graph with instances (A-Box) extracting 
    data from the ACLED Pandas DataFrame.
    """
    print(f"Populating Knowledge Graph (A-Box) with {len(df)} events...")
    
    for idx, row in df.iterrows():
        event_id = clean_uri(row['event_id_cnty'])
        if not event_id: continue
        
        event_uri = RES[event_id]
        
        # Event Instance & Basic Class
        g.add((event_uri, RDF.type, CONF.ConflictEvent))
        
        # Sub-classing based on event_type
        event_type = str(row.get('event_type', '')).lower()
        if 'battle' in event_type:
            g.add((event_uri, RDF.type, CONF.Battle))
        elif 'protest' in event_type:
            g.add((event_uri, RDF.type, CONF.Protest))
            
        # Data Properties (Date, Fatalities, Notes)
        if not pd.isna(row.get('event_date')):
            # Formatting date to YYYY-MM-DD
            date_str = str(row['event_date']).split(' ')[0] 
            g.add((event_uri, CONF.eventDate, Literal(date_str, datatype=XSD.date)))
            
        if not pd.isna(row.get('fatalities')):
            g.add((event_uri, CONF.fatalities, Literal(int(row['fatalities']), datatype=XSD.nonNegativeInteger)))
            
        if not pd.isna(row.get('notes')):
            g.add((event_uri, CONF.narrativeNotes, Literal(str(row['notes']), datatype=XSD.string)))
            
        # Actor 1 (Initiator)
        actor1_id = clean_uri(row.get('actor1'))
        if actor1_id:
            actor1_uri = RES[actor1_id]
            g.add((actor1_uri, RDF.type, CONF.Actor))
            g.add((event_uri, CONF.initiatedBy, actor1_uri))
            
        # Actor 2 (Target)
        actor2_id = clean_uri(row.get('actor2'))
        if actor2_id:
            actor2_uri = RES[actor2_id]
            g.add((actor2_uri, RDF.type, CONF.Actor))
            g.add((event_uri, CONF.targetedActor, actor2_uri))
            
        # Geographic Location
        loc_name = clean_uri(row.get('location'))
        country_name = clean_uri(row.get('country'))
        if loc_name and country_name:
            loc_uri = RES[f"{loc_name}_{country_name}"]
            g.add((loc_uri, RDF.type, CONF.GeographicLocation))
            g.add((event_uri, CONF.locatedIn, loc_uri))
            
            if not pd.isna(row.get('latitude')):
                g.add((loc_uri, CONF.latitude, Literal(float(row['latitude']), datatype=XSD.float)))
            if not pd.isna(row.get('longitude')):
                g.add((loc_uri, CONF.longitude, Literal(float(row['longitude']), datatype=XSD.float)))
            
            # Link to generic string for country reference
            g.add((loc_uri, SCHEMA.containedInPlace, Literal(str(row.get('country')), datatype=XSD.string)))

def visualize_ontology_schema():
    """Generates a NetworkX plot of the formal T-Box Schema."""
    print("Generating T-Box Schema Visualization...")
    G = nx.DiGraph()

    schema_edges = [
        ("ConflictEvent", "Actor", "hasSubnationalActor\n(Domain -> Range)"),
        ("ConflictEvent", "Actor", "initiatedBy\n(subProp: hasSubnationalActor)"),
        ("ConflictEvent", "Actor", "targetedActor"),
        ("ConflictEvent", "GeographicLocation", "locatedIn\n(subProp: schema:location)"),
        ("Battle", "ConflictEvent", "rdfs:subClassOf"),
        ("Protest", "ConflictEvent", "rdfs:subClassOf"),
    ]

    for u, v, label in schema_edges:
        G.add_edge(u, v, label=label)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.9)
    
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color='#d8e2dc', edgecolors='#333333', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, edge_color='#666666', arrows=True, arrowsize=20, width=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='#9d0208')

    plt.title("ACLED Conflict Ontology Schema (T-Box)", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    plt.savefig("acled_schema_visualization.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved schema visualization as 'acled_schema_visualization.png'")

def visualize_sample_abox(df, num_samples=2):
    """Generates a NetworkX plot of a small A-Box data sample."""
    print(f"Generating A-Box Sample Visualization for {num_samples} events...")
    
    sample_df = df.head(num_samples)
    G = nx.DiGraph()
    
    for _, row in sample_df.iterrows():
        event_node = f"Event:\n{row['event_id_cnty']}"
        
        G.add_node(event_node, color="#ffcdb2")
        
        if not pd.isna(row.get('actor1')):
            a1_node = f"Actor:\n{str(row['actor1'])[:15]}..."
            G.add_edge(event_node, a1_node, label='initiatedBy')
            
        if not pd.isna(row.get('actor2')):
            a2_node = f"Actor:\n{str(row['actor2'])[:15]}..."
            G.add_edge(event_node, a2_node, label='targetedActor')
            
        if not pd.isna(row.get('location')):
            loc_node = f"Loc:\n{row['location']}"
            G.add_edge(event_node, loc_node, label='locatedIn')

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.7)
    
    # Assign colors based on node text prefix
    colors = ['#ffcdb2' if 'Event:' in n else '#b5e2fa' if 'Actor:' in n else '#e2ece9' for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color=colors, edgecolors='#333333')
    nx.draw_networkx_edges(G, pos, edge_color='#888888', arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='#555555')

    plt.title("ACLED Knowledge Graph Data Sample (A-Box)", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    plt.savefig("acled_sample_abox_visualization.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved sample visualization as 'acled_sample_abox_visualization.png'")

def main():
    file_path = "c:/Users/tomma/Desktop/magistrale/second_semester/Data_Semantic_Project/acled_unified_middle_east.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return
        
    df = pd.read_csv(file_path)

    print(df.shape)
    
    # Initialize RDF Graph
    g = Graph()
    g.bind("conf", CONF)
    g.bind("res", RES)
    g.bind("schema", SCHEMA)
    g.bind("dbo", DBO)
    g.bind("geo", GEO)
    
    build_tbox(g)
    populate_abox(df, g)
    
    output_ttl = "c:/Users/tomma/Desktop/magistrale/second_semester/Data_Semantic_Project/acled_ontology.ttl"
    g.serialize(destination=output_ttl, format='turtle')
    print(f"Successfully exported full RDF Knowledge Graph to {output_ttl} (Total triples: {len(g)})")
    
    visualize_ontology_schema()
    visualize_sample_abox(df, num_samples=3)

if __name__ == "__main__":
    main()