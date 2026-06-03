import json
from collections import Counter
from datetime import datetime
import requests

# Load your communities and the KG
with open("leiden_graphrag_report_input.json", "r") as f:
    communities = json.load(f)

GRAPHDB_URL = "http://localhost:7200/repositories/MiddleEastConflict"
print(f"Querying GraphDB directly at {GRAPHDB_URL}...")

sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
top_10_communities = sorted_communities[:17]

# Analyze all communities
for comm_id, actors in top_10_communities:
    dates = []
    countries = []
    
    if not actors:
        continue
        
    # Use VALUES to search all actors in this community with a single fast query
    values_clause = " ".join([f"<http://data-semantics-2526.org/acled/resource/actor/{actor}>" for actor in actors])
    
    query = f"""
    PREFIX conf: <http://data-semantics-2526.org/acled/ontology#>
    
    SELECT ?date ?countryName WHERE {{
        VALUES ?actorUri {{ {values_clause} }}
        
        ?event a conf:ConflictEvent .
        {{ ?event conf:hasActor1 ?actorUri }}
        UNION
        {{ ?event conf:hasActor2 ?actorUri }}
        
        OPTIONAL {{ ?event conf:eventDate ?date . }}
        OPTIONAL {{ 
            ?event conf:locatedIn ?country .
            ?country conf:countryName ?countryName .
        }}
    }}
    """
    
    headers = {
        'Accept': 'application/sparql-results+json',
        'Content-Type': 'application/sparql-query'
    }
    
    response = requests.post(GRAPHDB_URL, data=query, headers=headers)
    
    if response.status_code == 200:
        bindings = response.json().get('results', {}).get('bindings', [])
        for row in bindings:
            if 'date' in row:
                dates.append(row['date']['value'])
            if 'countryName' in row:
                countries.append(row['countryName']['value'])
    else:
        print(f"Error querying {comm_id}: {response.text}")

    if dates:
        dates.sort()
        print(f"\nAnalysis of {comm_id} (Actors: {len(actors)}):")
        print(f"  -> First detected date: {dates[0]}")
        print(f"  -> Last detected date: {dates[-1]}")
        
        # Calculate the distance in days/years
        try:
            d1 = datetime.strptime(dates[0], "%Y-%m-%d")
            d2 = datetime.strptime(dates[-1], "%Y-%m-%d")
            delta_years = (d2 - d1).days / 365.25
            print(f"  -> Temporal extension of the cluster: {delta_years:.2f} years")
        except ValueError:
            pass
            
    if countries:
        top_country = Counter(countries).most_common(1)[0]
        print(f"  -> Most popular country: {top_country[0]} ({top_country[1]} events) of {len(countries)} total")