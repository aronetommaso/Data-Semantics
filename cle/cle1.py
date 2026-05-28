import pandas as pd

df = pd.read_csv("acled_unified_middle_east.csv")

print("=== DIMENSIONI ===")
print(f"Righe: {len(df):,}  |  Colonne: {len(df.columns)}")

print("\n=== COLONNE PRESENTI ===")
print(df.columns.tolist())

print("\n=== PRIME 3 RIGHE (trasposto per leggibilità) ===")
print(df.head(3).T)

print("\n=== VALORI NULLI PER COLONNA ===")
print(df.isnull().sum())

print("\n=== ESEMPIO CELLE CON PUNTO E VIRGOLA ===")
for col in ['source', 'assoc_actor1', 'assoc_actor2', 'tags']:
    if col in df.columns:
        multi = df[col].dropna().str.contains(';').sum()
        print(f"  {col}: {multi:,} righe con ';'")

print("\n=== RANGE TEMPORALE ===")
print(f"  Da: {df['event_date'].min()}  A: {df['event_date'].max()}")

print("\n=== TOP 5 PAESI ===")
print(df['country'].value_counts().head())