import pandas as pd

df = pd.read_csv("acled_unified_middle_east.csv")

# Quante fonti distinte esistono nel dataset?
fonti = df['source'].str.split(';').explode().str.strip()
print(f"Fonti uniche nel dataset: {fonti.nunique():,}")
print("\nTop 10 fonti:")
print(fonti.value_counts().head(10))

# Com'è distribuito inter1? (ci dice quali classi OWL creare)
print("\n=== VALORI INTER1 ===")
print(df['inter1'].value_counts())

print("\n=== VALORI INTER2 (escludendo null) ===")
print(df['inter2'].dropna().value_counts())

# Quanti event_id sono unici?
print(f"\nEvent ID unici: {df['event_id_cnty'].nunique():,}")
print(f"Totale righe:   {len(df):,}")