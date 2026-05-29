import os
import re
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# ==========================================
# CONFIGURAZIONE DINAMICA E SICURA AMBIENTE
# ==========================================

# Trova la cartella dove si trova FISICAMENTE questo script (es. la cartella /cle/)
SCRIPT_DIR = Path(__file__).resolve().parent

# Cerca il file .env nella cartella dello script o in quella superiore (Project/Data-Semantics)
env_path = SCRIPT_DIR / ".env"
if not env_path.exists():
    env_path = SCRIPT_DIR.parent / ".env"

# Carica esplicitamente il file .env trovato
load_dotenv(dotenv_path=env_path)

# Recupera il PATH dal .env. Se non esiste, usa la cartella superiore come fallback intelligente
if os.getenv("PROJECT_DIR"):
    PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
else:
    PROJECT_DIR = SCRIPT_DIR.parent

print(f"[INFO] Cartella di progetto impostata su: {PROJECT_DIR}")

# Definizione file di input e output (Tutti nella cartella principale)
FILE_INPUT_ACLED   = PROJECT_DIR / "acled_unified_middle_east.csv"
FILE_OUT_EVENTS    = PROJECT_DIR / "events.csv"
FILE_OUT_ACTORS    = PROJECT_DIR / "actors.csv"
FILE_OUT_SOURCES   = PROJECT_DIR / "sources.csv"
FILE_OUT_COUNTRIES = PROJECT_DIR / "countries.csv"

# ==========================================
# FUNZIONE: genera uno slug da un nome
# ==========================================
def slugify(text):
    if pd.isna(text):
        return None
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')

# ==========================================
# CARICAMENTO
# ==========================================
if not FILE_INPUT_ACLED.exists():
    raise FileNotFoundError(f"Impossibile trovare il dataset iniziale in: {FILE_INPUT_ACLED.resolve()}")

df = pd.read_csv(FILE_INPUT_ACLED)
print(f"Caricato: {len(df):,} eventi da {FILE_INPUT_ACLED}")

# Verifica rapida
print(df[['event_id_cnty', 'actor1', 'source']].head(3))
# ==========================================
# BLOCCO 2: events.csv
# ==========================================

df_events = df.copy()

# Precalcolo URI per ogni entità collegata all'evento
df_events['event_uri']   = 'res:event/'   + df_events['event_id_cnty']
df_events['actor1_uri']  = 'res:actor/'   + df_events['actor1'].apply(slugify)
df_events['actor2_uri']  = 'res:actor/'   + df_events['actor2'].apply(slugify)
df_events['country_uri'] = 'res:country/' + df_events['iso'].astype(str)

# actor2_uri → None dove actor2 è assente (40% dei casi)
df_events.loc[df_events['actor2'].isna(), 'actor2_uri'] = None

# Selezione colonne finali per events.csv
# (source NON va qui — andrà in sources.csv separato)
colonne_events = [
    'event_uri', 'event_id_cnty', 'event_date',
    'disorder_type', 'event_type', 'sub_event_type',
    'actor1_uri', 'inter1',
    'actor2_uri', 'inter2',
    'country_uri', 'iso', 'country',
    'location', 'latitude', 'longitude',
    'source_scale', 'fatalities', 'notes', 'tags'
]

df_events = df_events[colonne_events]

# Salvataggio
df_events.to_csv(FILE_OUT_EVENTS, index=False)
print(f"events.csv -> {len(df_events):,} righe")

# Verifica: mostra una riga trasposta per leggere tutto
print("\nEsempio riga 0:")
print(df_events.iloc[0])

# ==========================================
# BLOCCO 3: actors.csv
# ==========================================

# Prendiamo actor1 con il suo tipo (inter1)
actors1 = df[['actor1', 'inter1']].copy()
actors1.columns = ['actor_name', 'inter']

# Prendiamo actor2 con il suo tipo (inter2) — escludiamo i null
actors2 = df[['actor2', 'inter2']].dropna(subset=['actor2']).copy()
actors2.columns = ['actor_name', 'inter']

# Uniamo le due serie
df_actors = pd.concat([actors1, actors2], ignore_index=True)

# Deduplicazione: stesso nome → stessa riga
# Se un attore appare con inter diversi (raro ma possibile), teniamo il primo
df_actors = df_actors.drop_duplicates(subset=['actor_name'])

# Precalcolo URI
df_actors['actor_uri'] = 'res:actor/' + df_actors['actor_name'].apply(slugify)

# Mappiamo inter → classe OWL
inter_to_class = {
    'State forces':          'conf:StateForces',
    'Rebel group':           'conf:RebelGroup',
    'Political militia':     'conf:PoliticalMilitia',
    'Identity militia':      'conf:IdentityMilitia',
    'Rioters':               'conf:Rioters',
    'Protesters':            'conf:Protesters',
    'Civilians':             'conf:Civilian',
    'External/Other forces': 'conf:ExternalOther',
}
df_actors['owl_class'] = df_actors['inter'].map(inter_to_class)

# Riordino colonne
df_actors = df_actors[['actor_uri', 'actor_name', 'inter', 'owl_class']]

df_actors.to_csv(FILE_OUT_ACTORS, index=False)
print(f"actors.csv -> {len(df_actors):,} attori unici")

# Verifica
print("\nEsempio prime 5 righe:")
print(df_actors.head())

print("\nAttori senza classe OWL mappata (da investigare):")
print(df_actors[df_actors['owl_class'].isna()])

# ==========================================
# BLOCCO 4: sources.csv
# ==========================================

# Partiamo da event_id + source dal dataframe originale
df_sources = df[['event_id_cnty', 'source']].copy()

# Esplodi il campo source: "Arab 48; Twitter" → due righe separate
df_sources['source'] = df_sources['source'].str.split(';')
df_sources = df_sources.explode('source')

# Pulizia: rimuovi spazi bianchi lasciati dal split
df_sources['source'] = df_sources['source'].str.strip()

# Precalcolo URI
df_sources['event_uri']  = 'res:event/'  + df_sources['event_id_cnty']
df_sources['source_uri'] = 'res:source/' + df_sources['source'].apply(slugify)

# Riordino colonne
df_sources = df_sources[['event_uri', 'source_uri', 'event_id_cnty', 'source']]

df_sources.to_csv(FILE_OUT_SOURCES, index=False)
print(f"sources.csv -> {len(df_sources):,} righe (evento-fonte)")

# Verifica: mostra le righe dell'evento con più fonti
esempio = df_sources[df_sources['event_id_cnty'] == 'SYR122137']
print(f"\nEsempio evento SYR122137 (aveva 'Liveuamap; SHAAM; Telegram'):")
print(esempio)

# Quante fonti uniche nel grafo?
print(f"\nFonti uniche: {df_sources['source_uri'].nunique():,}")
# ==========================================
# BLOCCO 5: countries.csv
# ==========================================

df_countries = df[['iso', 'country']].drop_duplicates(subset=['iso'])

df_countries['country_uri'] = 'res:country/' + df_countries['iso'].astype(str)

df_countries = df_countries[['country_uri', 'iso', 'country']]

df_countries.to_csv(FILE_OUT_COUNTRIES, index=False)
print(f"countries.csv -> {len(df_countries):,} paesi unici")
print(df_countries)