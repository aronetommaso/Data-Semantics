import pandas as pd
import re
import os

# ==========================================
# FUNZIONE: genera uno slug da un nome
# ==========================================
def slugify(text):
    """
    Trasforma una stringa in un URI-safe slug.
    Esempio: "Hamas Movement" → "hamas-movement"
    """
    if pd.isna(text):
        return None
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)   # rimuove caratteri speciali
    text = re.sub(r'\s+', '-', text)             # spazi → trattini
    text = re.sub(r'-+', '-', text)              # trattini multipli → uno solo
    return text

# ==========================================
# CARICAMENTO
# ==========================================
df = pd.read_csv("acled_unified_middle_east.csv")
print(f"Caricato: {len(df):,} eventi")

# Verifica rapida
print(df[['event_id_cnty', 'actor1', 'source']].head(3))
