import os
import json
import re
import sys
import pandas as pd
from collections import defaultdict

# Fix radicale per i terminali Windows: forza lo standard output a usare UTF-8 se non lo fa già
if sys.platform == "win32":
    import sys, codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def load_env_file(dotenv_path=".env"):
    """Carica le variabili d'ambiente da un file .env locale senza librerie esterne."""
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

def slugify(value):
    """Normalizza il nome del paese per i nomi dei file."""
    value = str(value).lower().strip()
    value = re.sub(r'[^\w\s-]', '', value)
    value = re.sub(r'[\s_-]+', '-', value)
    return value

def main():
    # 0. Caricamento configurazioni da .env
    load_env_file()
    
    input_dir = os.environ.get("PROJECT_DIR", "").strip()
    output_dir = os.environ.get("GRAPHRAG_OUTPUT_DIR", "./graphrag_input/").strip()
    
    events_path = os.path.join(input_dir, 'events.csv')
    actors_path = os.path.join(input_dir, 'actors.csv')
    sources_path = os.path.join(input_dir, 'sources.csv')
    countries_path = os.path.join(input_dir, 'countries.csv')
    
    index_file_path = os.path.join(output_dir, 'event_index.json')
    
    MAX_EVENTS_PER_FILE = 800 
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Configurazione percorsi:")
    print(f" - Cartella Sorgente (Input): {input_dir if input_dir else 'Directory Corrente'}")
    print(f" - Cartella Destinazione (Output): {output_dir}")
    print("\n1. Caricamento dei 4 dataset puliti...")
    
    if not all(os.path.exists(f) for f in [events_path, actors_path, sources_path, countries_path]):
        raise FileNotFoundError(
            f"Impossibile trovare uno o più CSV in '{input_dir}'. "
            f"Controlla che events.csv, actors.csv, sources.csv e countries.csv siano presenti."
        )
        
    df_events = pd.read_csv(events_path, dtype={'iso': str, 'event_id_cnty': str})
    df_actors = pd.read_csv(actors_path, dtype={'actor_uri': str})
    df_sources = pd.read_csv(sources_path, dtype={'event_id_cnty': str})
    df_countries = pd.read_csv(countries_path, dtype={'iso': str})
    
    print("2. Preprocessing e Join dei dizionari d'anagrafica...")
    
    country_name_col = 'country_name' if 'country_name' in df_countries.columns else 'country'
    country_map = dict(zip(df_countries['iso'], df_countries[country_name_col]))
    actor_map = dict(zip(df_actors['actor_uri'], df_actors['actor_name']))
    
    df_sources_grouped = df_sources.groupby('event_id_cnty')['source'].apply(
        lambda x: '; '.join(x.dropna().astype(str).unique())
    ).reset_index()
    df_sources_grouped.columns = ['event_id_cnty', 'source_list']
    
    df_events = df_events.merge(df_sources_grouped, on='event_id_cnty', how='left')
    df_events['source_list'] = df_events['source_list'].fillna('Unknown Source')
    
    df_events['event_date_dt'] = pd.to_datetime(df_events['event_date'])
    df_events = df_events.sort_values(by='event_date_dt')
    df_events['year'] = df_events['event_date_dt'].dt.year
    df_events['quarter'] = df_events['event_date_dt'].dt.quarter
    
    event_index = {}
    stats_country_docs = defaultdict(int)
    total_events_processed = 0
    warnings_split = []
    
    print("3. Generazione dei documenti ibridi ottimizzati...")
    grouped = df_events.groupby(['iso', 'year', 'quarter'])
    
    for (iso, year, quarter), group in grouped:
        fallback_country = group['country'].iloc[0] if 'country' in group.columns else "Unknown"
        country_name_official = country_map.get(iso, fallback_country)
        
        country_slug = slugify(country_name_official)
        base_doc_id = f"{iso}_{country_slug}_{year}_Q{quarter}"
        
        records = group.to_dict('records')
        total_records = len(records)
        
        chunks = [records[i:i + MAX_EVENTS_PER_FILE] for i in range(0, total_records, MAX_EVENTS_PER_FILE)]
        
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunks) > 1:
                doc_id = f"{base_doc_id}_part{chunk_idx + 1}"
                if chunk_idx == 0:
                    # Rimosso l'emoji ⚠️ per evitare conflitti con la console di Windows
                    warnings_split.append(
                        f"[WARNING] {country_name_official} ({year} Q{quarter}) ha {total_records} eventi. "
                        f"Splittato in {len(chunks)} file."
                    )
            else:
                doc_id = base_doc_id
                
            filename = f"{doc_id}.txt"
            file_path = os.path.join(output_dir, filename)
            
            event_ids_in_doc = []
            doc_records = []
            
            header = (
                f"==================================================\n"
                f"DOCUMENT CONTEXT\n"
                f"Country: {country_name_official} (ISO Code: {iso})\n"
                f"Temporal Period: Year {year} - Quarter Q{quarter}\n"
                f"Total Events in Chunk: {len(chunk)} / {total_records}\n"
                f"==================================================\n\n"
            )
            doc_records.append(header)
            
            for row in chunk:
                ev_id = str(row['event_id_cnty'])
                event_ids_in_doc.append(ev_id)
                total_events_processed += 1
                
                act1_name = actor_map.get(row['actor1_uri'], "Unknown Actor 1")
                inter1_val = f" [{row['inter1']}]" if pd.notna(row['inter1']) else ""
                actor1_line = f"Actor1: {act1_name}{inter1_val}"
                
                actor2_line = ""
                if pd.notna(row['actor2_uri']) and str(row['actor2_uri']).strip() != "":
                    act2_name = actor_map.get(row['actor2_uri'], "Unknown Actor 2")
                    inter2_val = f" [{row['inter2']}]" if pd.notna(row['inter2']) else ""
                    actor2_line = f"\nActor2: {act2_name}{inter2_val}"
                
                fatalities_val = int(row['fatalities']) if pd.notna(row['fatalities']) else 0
                source_scale_val = row['source_scale'] if pd.notna(row['source_scale']) else 'Unknown Scale'
                notes_text = row['notes'] if pd.notna(row['notes']) else "No description available."
                
                record_str = (
                    f"[EVENT: {ev_id}]\n"
                    f"Date: {row['event_date']}\n"
                    f"Country: {country_name_official} | Location: {row['location']}\n"
                    f"Type: {row['event_type']} > {row['sub_event_type']} | Disorder: {row['disorder_type']}\n"
                    f"{actor1_line}{actor2_line}\n"
                    f"Fatalities: {fatalities_val}\n"
                    f"Sources: {row['source_list']} [{source_scale_val}]\n\n"
                    f"DESCRIPTION:\n"
                    f"{notes_text}"
                )
                doc_records.append(record_str)
                
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc_records[0] + "\n\n---\n\n".join(doc_records[1:]))
                
            event_index[doc_id] = event_ids_in_doc
            stats_country_docs[country_name_official] += 1

    print("4. Scrittura del file di Indice JSON...")
    with open(index_file_path, 'w', encoding='utf-8') as f:
        json.dump(event_index, f, indent=2, ensure_ascii=False)
    
    # Ora la stampa a terminale è sicura al 100%
    if warnings_split:
        print("\n--- AVVISI DI SPLIT DIMENSIONALE ---")
        for w in warnings_split:
            print(w)
            
    print("\n================ STATISTICHE FINALI ================")
    print(f"Numero totale di documenti .txt prodotti: {len(event_index)}")
    print(f"Numero totale di eventi mappati:          {total_events_processed}")
    print("\nDistribuzione file generati per Paese (da countries.csv):")
    for country, count in sorted(stats_country_docs.items()):
        print(f" - {country}: {count} file di testo")
    print("====================================================")

if __name__ == "__main__":
    main()