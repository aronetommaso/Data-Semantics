import pandas as pd
import urllib.parse
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

# ==========================================
# 1. DEFINIZIONE DEI NAMESPACE
# ==========================================
CONF = Namespace("http://ontology.conflict.org/schema/")
RES = Namespace("http://ontology.conflict.org/resource/")
SCHEMA = Namespace("https://schema.org/")
GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")

def clean_uri(text):
    """Pulisce le stringhe per creare URI validi e conformi agli standard RDF."""
    if pd.isna(text) or not str(text).strip():
        return None
    clean_str = str(text).strip().replace(" ", "_").replace('"', '').replace("'", "")
    return urllib.parse.quote(clean_str, safe="_")

def build_tbox(g):
    """
    Definisce lo Schema (T-Box) del Knowledge Graph.
    Imposta le gerarchie delle Classi (SubClasses) e i vincoli di Dominio/Range.
    """
    print("Costruzione della T-Box (Schema Ontologico)...")

    # --- HIERARCHY DELLE CLASSI ---
    
    # Eventi
    g.add((CONF.ConflictEvent, RDF.type, OWL.Class))
    g.add((CONF.ConflictEvent, RDFS.subClassOf, SCHEMA.Event))
    
    # Attori
    g.add((CONF.Actor, RDF.type, OWL.Class))
    g.add((CONF.Actor, RDFS.subClassOf, SCHEMA.Organization))
    
    # Entità Geografiche (Modellazione Gerarchica: Country e Location)
    g.add((CONF.GeographicEntity, RDF.type, OWL.Class))
    g.add((CONF.GeographicEntity, RDFS.subClassOf, SCHEMA.Place))
    
    g.add((CONF.Country, RDF.type, OWL.Class))
    g.add((CONF.Country, RDFS.subClassOf, CONF.GeographicEntity))
    
    g.add((CONF.Location, RDF.type, OWL.Class))
    g.add((CONF.Location, RDFS.subClassOf, CONF.GeographicEntity))

    # Fonti
    g.add((CONF.Source, RDF.type, OWL.Class))
    g.add((CONF.Source, RDFS.subClassOf, SCHEMA.Organization))

    # --- OBJECT PROPERTIES (Relazioni tra nodi) ---
    
    # Event -> initiatedBy -> Actor
    g.add((CONF.initiatedBy, RDF.type, OWL.ObjectProperty))
    g.add((CONF.initiatedBy, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.initiatedBy, RDFS.range, CONF.Actor))

    # Event -> targetedActor -> Actor
    g.add((CONF.targetedActor, RDF.type, OWL.ObjectProperty))
    g.add((CONF.targetedActor, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.targetedActor, RDFS.range, CONF.Actor))

    # Actor -> helpedBy -> Actor (Per assoc_actor)
    g.add((CONF.helpedBy, RDF.type, OWL.ObjectProperty))
    g.add((CONF.helpedBy, RDFS.domain, CONF.Actor))
    g.add((CONF.helpedBy, RDFS.range, CONF.Actor))

    # Event -> locatedIn -> Location
    g.add((CONF.locatedIn, RDF.type, OWL.ObjectProperty))
    g.add((CONF.locatedIn, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.locatedIn, RDFS.range, CONF.Location))

    # Location -> withinCountry -> Country (Gerarchia Spaziale)
    g.add((CONF.withinCountry, RDF.type, OWL.ObjectProperty))
    g.add((CONF.withinCountry, RDFS.domain, CONF.Location))
    g.add((CONF.withinCountry, RDFS.range, CONF.Country))

    # Event -> postedBy -> Source
    g.add((CONF.postedBy, RDF.type, OWL.ObjectProperty))
    g.add((CONF.postedBy, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.postedBy, RDFS.range, CONF.Source))

    # --- DATA PROPERTIES (Attributi Letterali) ---
    
    # Date, coordinate e metadati essenziali per GraphRAG
    g.add((CONF.when, RDF.type, OWL.DatatypeProperty))
    g.add((CONF.when, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.when, RDFS.range, XSD.date))

    g.add((CONF.narrativeNotes, RDF.type, OWL.DatatypeProperty))
    g.add((CONF.narrativeNotes, RDFS.domain, CONF.ConflictEvent))
    g.add((CONF.narrativeNotes, RDFS.range, XSD.string))

    g.add((CONF.fatalities, RDF.type, OWL.DatatypeProperty))
    g.add((CONF.fatalities, RDFS.range, XSD.integer))

    g.add((GEO.lat, RDF.type, OWL.DatatypeProperty))
    g.add((GEO.long, RDF.type, OWL.DatatypeProperty))

def populate_abox(df, g):
    """
    Popola il Knowledge Graph (A-Box) iterando sui record del CSV.
    """
    print(f"Popolamento A-Box con {len(df)} eventi...")

    for _, row in df.iterrows():
        event_id = clean_uri(row['event_id_cnty'])
        if not event_id:
            continue
            
        event_uri = RES[f"Event_{event_id}"]
        
        # 1. Creazione Istanza Evento
        g.add((event_uri, RDF.type, CONF.ConflictEvent))
        
        # Data (When)
        if not pd.isna(row.get('event_date')):
            date_str = str(row['event_date']).split(' ')[0]
            g.add((event_uri, CONF.when, Literal(date_str, datatype=XSD.date)))
            
        # Note (Narrative Notes per GraphRAG)
        if not pd.isna(row.get('notes')):
            g.add((event_uri, CONF.narrativeNotes, Literal(str(row['notes']), datatype=XSD.string)))
            
        # Tassonomia e Dati quantitativi
        for col, prop in [('disorder_type', 'disorderType'), ('event_type', 'eventType'), ('sub_event_type', 'subEventType')]:
            if not pd.isna(row.get(col)):
                g.add((event_uri, URIRef(CONF + prop), Literal(str(row[col]), datatype=XSD.string)))
                
        if not pd.isna(row.get('fatalities')):
            try:
                g.add((event_uri, CONF.fatalities, Literal(int(float(row['fatalities'])), datatype=XSD.integer)))
            except ValueError:
                pass

        # 2. Creazione Attori e Relazioni Iniziatore/Target
        actor1_uri = None
        actor2_uri = None
        
        if not pd.isna(row.get('actor1')):
            actor1_uri = RES[f"Actor_{clean_uri(row['actor1'])}"]
            g.add((actor1_uri, RDF.type, CONF.Actor))
            g.add((actor1_uri, SCHEMA.name, Literal(str(row['actor1']), datatype=XSD.string)))
            g.add((event_uri, CONF.initiatedBy, actor1_uri))
            
        if not pd.isna(row.get('actor2')):
            actor2_uri = RES[f"Actor_{clean_uri(row['actor2'])}"]
            g.add((actor2_uri, RDF.type, CONF.Actor))
            g.add((actor2_uri, SCHEMA.name, Literal(str(row['actor2']), datatype=XSD.string)))
            g.add((event_uri, CONF.targetedActor, actor2_uri))

        # 3. Relazione HelpedBy (Attori associati)
        # ACLED separa spesso gli associati con un punto e virgola ';'
        if actor1_uri and not pd.isna(row.get('assoc_actor1')):
            assoc_list = str(row['assoc_actor1']).split(';')
            for assoc in assoc_list:
                assoc_uri = RES[f"Actor_{clean_uri(assoc)}"]
                g.add((assoc_uri, RDF.type, CONF.Actor))
                g.add((actor1_uri, CONF.helpedBy, assoc_uri))

        if actor2_uri and not pd.isna(row.get('assoc_actor2')):
            assoc_list = str(row['assoc_actor2']).split(';')
            for assoc in assoc_list:
                assoc_uri = RES[f"Actor_{clean_uri(assoc)}"]
                g.add((assoc_uri, RDF.type, CONF.Actor))
                g.add((actor2_uri, CONF.helpedBy, assoc_uri))

        # 4. Modellazione Geografica Gerarchica (Event -> Location -> Country)
        loc_name = clean_uri(row.get('location'))
        country_name = clean_uri(row.get('country'))
        iso_code = clean_uri(row.get('iso'))
        
        if loc_name and country_name:
            # Creazione Nazione
            country_uri = RES[f"Country_{iso_code}" if iso_code else f"Country_{country_name}"]
            g.add((country_uri, RDF.type, CONF.Country))
            g.add((country_uri, SCHEMA.name, Literal(str(row['country']), datatype=XSD.string)))
            
            # Creazione Location
            loc_uri = RES[f"Location_{loc_name}_{country_name}"]
            g.add((loc_uri, RDF.type, CONF.Location))
            g.add((loc_uri, SCHEMA.name, Literal(str(row['location']), datatype=XSD.string)))
            
            # Coordinate
            if not pd.isna(row.get('latitude')):
                g.add((loc_uri, GEO.lat, Literal(float(row['latitude']), datatype=XSD.float)))
            if not pd.isna(row.get('longitude')):
                g.add((loc_uri, GEO.long, Literal(float(row['longitude']), datatype=XSD.float)))

            # Archi Spaziali
            g.add((event_uri, CONF.locatedIn, loc_uri))
            g.add((loc_uri, CONF.withinCountry, country_uri))

        # 5. Creazione Sorgente (Source)
        source_name = clean_uri(row.get('source'))
        if source_name:
            source_uri = RES[f"Source_{source_name}"]
            g.add((source_uri, RDF.type, CONF.Source))
            g.add((source_uri, SCHEMA.name, Literal(str(row['source']), datatype=XSD.string)))
            g.add((event_uri, CONF.postedBy, source_uri))

def main():
    # Caricamento del dataset ACLED
    # Sostituisci con il path reale del tuo file CSV
    file_path = "acled_unified_middle_east.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Errore: File {file_path} non trovato. Assicurati che il path sia corretto.")
        return

    # Inizializzazione del Grafo RDF
    g = Graph()
    g.bind("conf", CONF)
    g.bind("res", RES)
    g.bind("schema", SCHEMA)
    g.bind("geo", GEO)
    
    # Costruzione
    build_tbox(g)
    populate_abox(df, g)
    
    # Esportazione (Serializzazione in formato Turtle, standard per GraphDB)
    output_ttl = "acled_knowledge_graph.ttl"
    g.serialize(destination=output_ttl, format='turtle')
    print(f"Grafo esportato con successo in {output_ttl}. Triplette totali: {len(g)}")

if __name__ == "__main__":
    main()