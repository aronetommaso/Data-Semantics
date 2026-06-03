import os
import json
import time
import math
import pandas as pd
from google import genai

def load_env_file(dotenv_path="C:\\Users\\HP\\Desktop\\data_science\\primo_anno\\DataSemantics\\Project\\Data-Semantics\\trio\\.env.txt"):
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

def setup_gemini():
    load_env_file()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found!")
    return genai.Client(api_key=api_key)

def generate_with_backoff(client, prompt, max_retries=6, base_delay=20.0):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',  # FIX 1: coerente con settings.yaml
                contents=prompt,
            )
            # FIX ANTI-VUOTO: Se la risposta è vuota o bloccata, forza un'eccezione per far scattare il riavvio
            if not response.text:
                raise Exception("API returned an empty text block (possible safety block or server glitch).")
                
            return response.text
            
        except Exception as e:
            wait_time = base_delay * (2 ** attempt)
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = max(wait_time, 70.0)
            print(f"    [!] Error (attempt {attempt+1}/{max_retries}). Waiting {wait_time:.0f}s... | {e}")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded.")

def calculate_severity_metrics(comm_events):
    hist_events_count = len(comm_events)
    hist_fatalities = comm_events['fatalities'].sum()
    hist_impact = hist_events_count + (hist_fatalities * 5)
    hist_severity = min(100.0, (math.log1p(hist_impact) / math.log1p(50000)) * 100)

    events_2023 = comm_events[comm_events['year'] == 2023]
    curr_impact = len(events_2023) + (events_2023['fatalities'].sum() * 5)
    curr_severity = min(100.0, (math.log1p(curr_impact) / math.log1p(10000)) * 100)

    events_2022_count = len(comm_events[comm_events['year'] == 2022])
    recent = len(events_2023)
    previous = events_2022_count
    if recent + previous == 0:
        escalation_risk = 50.0
    else:
        sgr = (recent - previous) / (recent + previous)
        escalation_risk = (sgr + 1.0) * 50.0

    return round(hist_severity, 1), round(curr_severity, 1), round(escalation_risk, 1)

def main():
    events_path = "C:\\Users\\HP\\Desktop\\data_science\\primo_anno\\DataSemantics\\Project\\Data-Semantics\\events.csv"
    actors_path = "C:\\Users\\HP\\Desktop\\data_science\\primo_anno\\DataSemantics\\Project\\Data-Semantics\\actors.csv"          # FIX 2: usiamo actor_name reale
    communities_path = "C:\\Users\\HP\\Desktop\\data_science\\primo_anno\\DataSemantics\\Project\\Data-Semantics\\leiden_communities.json"
    output_dir = "C:\\Users\\HP\\Desktop\\data_science\\primo_anno\\DataSemantics\\Project\\Data-Semantics\\trio\\community_reports"
    os.makedirs(output_dir, exist_ok=True)

    client = setup_gemini()

    print("1. Loading data...")
    with open(communities_path, "r", encoding="utf-8") as f:
        communities = json.load(f)

    df_events = pd.read_csv(events_path)
    df_actors = pd.read_csv(actors_path)

    # FIX 2: dizionario slug → nome reale da actors.csv
    actor_name_map = dict(zip(df_actors['actor_uri'], df_actors['actor_name']))

    df_events['event_date_dt'] = pd.to_datetime(df_events['event_date'])
    df_events['year'] = df_events['event_date_dt'].dt.year

    def get_slug(uri):
        if pd.isna(uri): return ""
        return str(uri).split('/')[-1]

    df_events['actor1_slug'] = df_events['actor1_uri'].apply(get_slug)
    df_events['actor2_slug'] = df_events['actor2_uri'].apply(get_slug)

    print(f"Found {len(communities)} communities.")

    for comm_id, actors_list in communities.items():

        # FIX 3: checkpoint — salta se il report esiste già
        file_name = os.path.join(output_dir, f"report_{comm_id}.md")
        if os.path.exists(file_name):
            print(f"[SKIP] {comm_id} — report already exists.")
            continue

        print(f"\n--- Processing {comm_id} ({len(actors_list)} actors) ---")

        mask = df_events['actor1_slug'].isin(actors_list) | df_events['actor2_slug'].isin(actors_list)
        comm_events = df_events[mask]

        if comm_events.empty:
            print(f"No events found. Skipping.")
            continue

        print(f"Total events: {len(comm_events)}")
        hist_sev, curr_sev, esc_risk = calculate_severity_metrics(comm_events)
        print(f"Metrics -> Historical: {hist_sev} | Current: {curr_sev} | Escalation: {esc_risk}")

        # FIX 4: top 50 per anno
        sampled_events = (
            comm_events
            .sort_values('fatalities', ascending=False)
            .groupby('year')
            .head(50)
            .sort_values('event_date_dt')
        )
        print(f"Sampled events for LLM: {len(sampled_events)}")

        context_blocks = []
        for _, row in sampled_events.iterrows():
            # FIX 2: nome reale dall'actors.csv
            a1 = actor_name_map.get(row['actor1_slug'], row['actor1_slug'])
            a2 = actor_name_map.get(row['actor2_slug'], "") if row['actor2_slug'] else ""
            actor2_str = f" vs {a2}" if a2 else ""

            block = (
                f"Date: {row['event_date']} | Location: {row['country']}, {row['location']}\n"
                f"Actors: {a1}{actor2_str}\n"
                f"Type: {row['sub_event_type']} | Fatalities: {row['fatalities']}\n"
                f"Description: {row['notes']}\n"
            )
            context_blocks.append(block)

        full_context = "\n---\n".join(context_blocks)

        prompt = f"""You are an expert geopolitical analyst. Below is a chronologically stratified sample of conflict events from a specific actor cluster in the Middle East.

Write a structured "Community Report" in Markdown with YAML Frontmatter.

RULES:
- Use ONLY the pre-calculated metrics below. Do NOT invent or modify them.
- community_id: "{comm_id}"
- Historical Severity (0-100): {hist_sev}
- Current Severity (0-100): {curr_sev}  
- Escalation Risk (0-100): {esc_risk}

CONFLICT DATA:
{full_context}

REPORT STRUCTURE:
1. YAML Frontmatter (community_id, region_or_name, severity_metrics, primary_actors)
2. Executive Summary
3. Primary Actors & Dynamics
4. Temporal Evolution (2015-2023)
5. Severity Assessment (must cite the exact metrics and justify them)
6. Strategic Impact & Humanitarian Consequences

Tone: academic, formal, objective. Language: English."""

        print(f"Calling Gemini API...")
        try:
            report_text = generate_with_backoff(client, prompt)
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"Saved: {file_name}")
        except Exception as e:
            print(f"CRITICAL ERROR for {comm_id}: {e}")

        # FIX 5: sleep più lungo per stare sotto TPM con comunità grandi
        print("    Cooling down 5 seconds...")
        time.sleep(5)

    print("\nDone. Check ./community_reports/")

if __name__ == "__main__":
    main()