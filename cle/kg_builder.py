"""
ACLED Conflict Knowledge Graph Builder — Versione Ibrida Definitiva
====================================================================
Approccio:
  - Scrittura diretta su file Turtle (no rdflib in RAM) → scalabile su 91k+ eventi
  - Chunk processing per events.csv e sources.csv → memoria controllata
  - Deduplicazione fonti con set Python → nessun nodo duplicato
  - Classi OWL assegnate via dict esplicita → robusto, indipendente dal CSV
  - escape_turtle_string → file Turtle sintaticamente corretto
  - Direzione reportedBy corretta → Event → Source (rispetta rdfs:domain/range)
  - Report finale con contatori → verificabile

Dipendenze:
    pip install pandas tqdm

Uso:
    python kg_builder_final.py
    (I 4 CSV devono essere nella stessa cartella, o modifica DATA_DIR)

Output:
    acled_kg.ttl    → Knowledge Graph pronto per GraphDB
    kg_report.txt   → Statistiche triple prodotte
"""

import os
import re
import time
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURAZIONE
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Namespace — identici all'ontologia OWL
ONTOLOGY_NS = "http://data-semantics-2526.org/acled/ontology#"
RESOURCE_NS  = "http://data-semantics-2526.org/acled/resource/"

TURTLE_PREFIXES = f"""\
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix conf: <{ONTOLOGY_NS}> .
@prefix res:  <{RESOURCE_NS}> .

"""

# Mappa inter → classe OWL
# Fonte di verità unica: se il preprocessing cambia, si modifica solo qui.
INTER_TO_CLASS: dict[str, str] = {
    "State forces":          "conf:StateForces",
    "Rebel group":           "conf:RebelGroup",
    "Political militia":     "conf:PoliticalMilitia",
    "Identity militia":      "conf:IdentityMilitia",
    "Rioters":               "conf:Rioters",
    "Protesters":            "conf:Protesters",
    "Civilians":             "conf:Civilian",
    "External/Other forces": "conf:ExternalOther",
}

# Chunk size per i file grandi
CHUNK_EVENTS  = 10_000   # events.csv:  91k righe
CHUNK_SOURCES = 20_000   # sources.csv: 157k righe


# ──────────────────────────────────────────────────────────────────────────────
# 2. FUNZIONI DI UTILITÀ (MODIFICATE PER FIX DI GRAPHDB)
# ──────────────────────────────────────────────────────────────────────────────

def escape_turtle(text) -> str:
    """
    Sanifica una stringa per l'inserimento in un literal Turtle.
    Gestisce: virgolette, backslash, newline, carriage return, tab.
    Senza questo, una nota con virgolette rompe la sintassi del file.
    """
    if pd.isna(text):
        return ""
    s = str(text)
    s = s.replace("\\", "\\\\")
    s = s.replace('"',  '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s


def is_empty(value) -> bool:
    """True se il valore è nullo, NaN o stringa vuota."""
    if value is None:
        return True
    if pd.isna(value):
        return True
    return str(value).strip() in ("", "nan", "NaN", "None")


def make_valid_uri(value: str) -> str:
    """
    Risolve l'errore di parsing di GraphDB.
    Se uno slug contiene un carattere '/' (es. 'actor/hamas-movement'), 
    non può essere usato come prefisso contratto 'res:actor/hamas-movement'.
    Questa funzione lo trasforma in un URI assoluto racchiuso tra parentesi angolari:
    <{RESOURCE_NS}actor/hamas-movement>
    """
    clean_val = str(value).strip().removeprefix("res:")
    return f"<{RESOURCE_NS}{clean_val}>"


def count_csv_rows(path: str) -> int:
    """Conta le righe di un CSV senza caricarlo in memoria (per tqdm)."""
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # -1 per l'header


# ──────────────────────────────────────────────────────────────────────────────
# 3. MAPPING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def map_countries(csv_path: str, out_file, stats: dict) -> None:
    """
    countries.csv → triple conf:Country
    """
    log.info("→ Countries: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)

    triple_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Countries"):
        if is_empty(row.get("iso")):
            return

        iso = str(row["iso"]).strip()
        name = escape_turtle(row.get("country", ""))

        # Usiamo l'URI assoluto per uniformità e sicurezza sintattica
        country_uri = f"<{RESOURCE_NS}country/{iso}>"

        out_file.write(f"{country_uri} a conf:Country ;\n")
        out_file.write(f'    conf:countryName "{name}"^^xsd:string ;\n')
        out_file.write(f'    conf:isoCode "{iso}"^^xsd:integer .\n\n')
        triple_count += 3

    stats["countries"] = len(df)
    stats["triples_countries"] = triple_count
    log.info("   %d paesi → %d triple", stats["countries"], triple_count)


def map_actors(csv_path: str, out_file, stats: dict) -> None:
    """
    actors.csv → triple conf:Actor (sottoclasse specifica)
    """
    log.info("→ Actors: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)

    triple_count = 0
    unknown_inter: set[str] = set()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Actors"):
        if is_empty(row.get("actor_uri")):
            continue

        # Genera un URI assoluto <http://.../resource/actor/slug> evitando l'errore dello slash
        actor_uri = make_valid_uri(row["actor_uri"])
        actor_name = escape_turtle(row.get("actor_name", ""))

        inter_val = str(row.get("inter", "")).strip()
        owl_class = INTER_TO_CLASS.get(inter_val)
        if owl_class is None:
            owl_class = "conf:Actor"
            unknown_inter.add(inter_val)

        out_file.write(f"{actor_uri} a {owl_class} ;\n")
        out_file.write(f'    conf:actorName "{actor_name}"^^xsd:string .\n\n')
        triple_count += 2

    if unknown_inter:
        log.warning(
            "   Valori inter non mappati (fallback conf:Actor): %s", unknown_inter
        )

    stats["actors"] = len(df)
    stats["triples_actors"] = triple_count
    log.info("   %d attori → %d triple", stats["actors"], triple_count)


def map_sources(csv_path: str, out_file, stats: dict) -> None:
    """
    sources.csv → triple conf:Source + relazione conf:reportedBy
    """
    log.info("→ Sources: %s", csv_path)
    total_rows = count_csv_rows(csv_path)

    log.info("   Passata 1/2: raccolta fonti uniche...")
    seen_sources: dict[str, str] = {}
    for chunk in pd.read_csv(csv_path, dtype=str, chunksize=CHUNK_SOURCES):
        for _, row in chunk.iterrows():
            raw_slug = str(row.get("source_uri", "")).strip()
            name = str(row.get("source", "")).strip()
            if not is_empty(raw_slug) and raw_slug not in seen_sources:
                seen_sources[raw_slug] = name

    for raw_slug, name in seen_sources.items():
        source_uri = make_valid_uri(raw_slug)
        out_file.write(f"{source_uri} a conf:Source ;\n")
        out_file.write(f'    conf:sourceName "{escape_turtle(name)}"^^xsd:string .\n\n')

    triple_count = len(seen_sources) * 2

    log.info("   Passata 2/2: mapping relazioni evento→fonte...")
    event_sources: dict[str, list[str]] = {}
    row_count = 0

    with tqdm(total=total_rows, desc="Sources") as pbar:
        for chunk in pd.read_csv(csv_path, dtype=str, chunksize=CHUNK_SOURCES):
            for _, row in chunk.iterrows():
                evt_raw = str(row.get("event_uri", "")).strip()
                src_raw = str(row.get("source_uri", "")).strip()
                if is_empty(evt_raw) or is_empty(src_raw):
                    continue
                event_sources.setdefault(evt_raw, []).append(src_raw)
                row_count += 1
            pbar.update(len(chunk))

    for evt_raw, src_list in event_sources.items():
        evt_uri = make_valid_uri(evt_raw)
        src_uris = ", ".join(make_valid_uri(s) for s in src_list)
        out_file.write(f"{evt_uri} conf:reportedBy {src_uris} .\n\n")
        triple_count += len(src_list)

    stats["sources_rows"]  = total_rows
    stats["sources_unique"] = len(seen_sources)
    stats["triples_sources"] = triple_count
    log.info(
        "   %d righe → %d fonti uniche → %d triple",
        total_rows, len(seen_sources), triple_count,
    )


def map_events(csv_path: str, out_file, stats: dict) -> None:
    """
    events.csv → triple conf:ConflictEvent
    """
    log.info("→ Events: %s", csv_path)
    total_rows = count_csv_rows(csv_path)

    triple_count = 0
    skipped_actor2 = 0
    skipped_date   = 0

    with tqdm(total=total_rows, desc="Events") as pbar:
        for chunk in pd.read_csv(csv_path, dtype=str, chunksize=CHUNK_EVENTS):
            for _, row in chunk.iterrows():
                event_id = str(row.get("event_id_cnty", "")).strip()
                if not event_id or event_id == "nan":
                    continue

                subj = f"<{RESOURCE_NS}event/{event_id}>"

                triples: list[str] = []
                triples.append(f"{subj} a conf:ConflictEvent")

                date_val = str(row.get("event_date", "")).strip()
                if date_val and date_val != "nan":
                    try:
                        pd.Timestamp(date_val)
                        triples.append(f'    conf:eventDate "{date_val}"^^xsd:date')
                    except Exception:
                        skipped_date += 1
                else:
                    skipped_date += 1

                for pred, col in [
                    ("conf:disorderType", "disorder_type"),
                    ("conf:eventType",    "event_type"),
                    ("conf:subEventType", "sub_event_type"),
                    ("conf:locationName", "location"),
                    ("conf:sourceScale",  "source_scale"),
                    ("conf:notes",        "notes"),
                    ("conf:tags",         "tags"),
                ]:
                    val = row.get(col)
                    if not is_empty(val):
                        triples.append(f'    {pred} "{escape_turtle(val)}"^^xsd:string')

                fat = row.get("fatalities")
                if not is_empty(fat):
                    try:
                        triples.append(f'    conf:fatalities "{int(float(fat))}"^^xsd:integer')
                    except (ValueError, TypeError):
                        pass

                for pred, col in [("conf:latitude", "latitude"), ("conf:longitude", "longitude")]:
                    val = row.get(col)
                    if not is_empty(val):
                        try:
                            triples.append(f'    {pred} "{float(val)}"^^xsd:decimal')
                        except (ValueError, TypeError):
                            pass

                iso = str(row.get("iso", "")).strip()
                if not is_empty(iso):
                    try:
                        triples.append(f"    conf:locatedIn <{RESOURCE_NS}country/{int(float(iso))}>")
                    except (ValueError, TypeError):
                        pass

                a1 = str(row.get("actor1_uri", "")).strip()
                if not is_empty(a1) and a1 != "nan":
                    triples.append(f"    conf:hasActor1 {make_valid_uri(a1)}")

                a2 = str(row.get("actor2_uri", "")).strip()
                if not is_empty(a2) and a2 != "nan":
                    triples.append(f"    conf:hasActor2 {make_valid_uri(a2)}")
                else:
                    skipped_actor2 += 1

                if len(triples) == 1:
                    out_file.write(triples[0] + " .\n\n")
                else:
                    for i, line in enumerate(triples):
                        if i < len(triples) - 1:
                            out_file.write(line + " ;\n")
                        else:
                            out_file.write(line + " .\n\n")

                triple_count += len(triples)

            pbar.update(len(chunk))

    stats["events"]         = total_rows
    stats["triples_events"] = triple_count
    stats["skipped_actor2"] = skipped_actor2
    stats["skipped_date"]   = skipped_date
    log.info(
        "   %d eventi → %d triple  (senza actor2: %d, date malformate: %d)",
        total_rows, triple_count, skipped_actor2, skipped_date,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. REPORT FINALE
# ──────────────────────────────────────────────────────────────────────────────

def write_report(stats: dict, output_path: str, elapsed: float) -> None:
    total = (
        stats.get("triples_countries", 0)
        + stats.get("triples_actors", 0)
        + stats.get("triples_events", 0)
        + stats.get("triples_sources", 0)
    )
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)

    report = f"""
╔══════════════════════════════════════════════════════════╗
║         ACLED KG Builder — Report Finale                 ║
╠══════════════════════════════════════════════════════════╣
║  Tempo di esecuzione : {elapsed:>6.1f}s
║  File output         : {output_path}
║  Dimensione file     : {size_mb:>6.2f} MB
║
║  Entità processate
║    Paesi             : {stats.get('countries', 0):>8,}
║    Attori            : {stats.get('actors', 0):>8,}
║    Eventi            : {stats.get('events', 0):>8,}
║    Coppie evt-fonte  : {stats.get('sources_rows', 0):>8,}
║    Fonti uniche      : {stats.get('sources_unique', 0):>8,}
║
║  Triple prodotte
║    Da countries.csv  : {stats.get('triples_countries', 0):>8,}
║    Da actors.csv     : {stats.get('triples_actors', 0):>8,}
║    Da events.csv     : {stats.get('triples_events', 0):>8,}
║    Da sources.csv    : {stats.get('triples_sources', 0):>8,}
║    ──────────────────────────────────────
║    TOTALE            : {total:>8,}
║
║  Note
║    Eventi senza actor2  : {stats.get('skipped_actor2', 0):>7,}
║    Date malformate      : {stats.get('skipped_date', 0):>7,}
╚══════════════════════════════════════════════════════════╝
"""
    try:
        print(report)
    except UnicodeEncodeError:
        print("\n[SUCCESS] Knowledge Graph Generato! Controlla il file 'kg_report.txt' per le statistiche dettagliate.")
    report_path = Path(output_path).parent / "kg_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info("Report salvato in %s", report_path)


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv

    # 1. Trova la cartella dove si trova fisicamente questo script (la cartella /cle/)
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # 2. Cerca e carica il file .env (nella cartella attuale o in quella superiore)
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        env_path = SCRIPT_DIR.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    # 3. Recupera la cartella principale del progetto (dove risiedono i 4 CSV)
    if os.getenv("PROJECT_DIR"):
        PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
    else:
        PROJECT_DIR = SCRIPT_DIR.parent

    # 4. Impostazione dei percorsi di INPUT (leggiamo i CSV dalla cartella principale)
    FILE_COUNTRIES = PROJECT_DIR / "countries.csv"
    FILE_ACTORS    = PROJECT_DIR / "actors.csv"
    FILE_SOURCES   = PROJECT_DIR / "sources.csv"
    FILE_EVENTS    = PROJECT_DIR / "events.csv"

    # 5. Impostazione del percorso di OUTPUT (il file .ttl va forzatamente in /cle/)
    OUTPUT_PATH = SCRIPT_DIR / "acled_kg.ttl"

    log.info("=== ACLED Knowledge Graph Builder (versione ibrida) ===")
    log.info("Cartella Sorgente Dati (CSV): %s", PROJECT_DIR)
    log.info("Destinazione Grafo (.ttl): %s", OUTPUT_PATH.resolve())

    # Verifica che i file di input esistano davvero
    missing = [
        str(f) for f in [FILE_COUNTRIES, FILE_ACTORS, FILE_SOURCES, FILE_EVENTS]
        if not Path(f).exists()
    ]
    if missing:
        log.error("File CSV mancanti nella cartella principale: %s", missing)
        log.error("Assicurati di aver eseguito con successo il preprocessing_cle2.py")
        raise SystemExit(1)

    start = time.time()
    stats: dict = {}

    # Inizializzazione file (sovrascrive se esiste)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(TURTLE_PREFIXES)

    # Pipeline sequenziale — append su file aperto in modalità 'a'
    try:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            map_countries(str(FILE_COUNTRIES), f, stats)  # 1. Paesi
            map_actors(str(FILE_ACTORS),       f, stats)  # 2. Attori
            map_sources(str(FILE_SOURCES),     f, stats)  # 3. Fonti + relazioni
            map_events(str(FILE_EVENTS),       f, stats)  # 4. Eventi

    except Exception as e:
        log.exception("Errore durante la generazione del Knowledge Graph: %s", e)
        raise SystemExit(1)

    elapsed = time.time() - start
    write_report(stats, str(OUTPUT_PATH), elapsed)