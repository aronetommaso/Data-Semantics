import requests
import pandas as pd
import json
import os
import time
from dotenv import load_dotenv

# Carica le variabili nascoste dal file .env
load_dotenv()

# ==========================================
# 1. RECUPERO CREDENZIALI SICURO
# ==========================================
# os.getenv va a cercare il nome esatto che hai scritto nel file .env
USERNAME = os.getenv("ACLED_USERNAME")
PASSWORD = os.getenv("ACLED_PASSWORD")

REGION_CODE = 11
ANNO_INIZIO = 2023

# Controllo di sicurezza: se hai dimenticato il file .env, il programma si ferma
if not USERNAME or not PASSWORD:
    raise ValueError("Errore: Credenziali non trovate! Hai creato il file .env?")

# ==========================================
# 2. DEFINISCI COSA VUOI SCARICARE
# ==========================================
# Selezioniamo un paio di nazioni e l'anno. 
# NOTA: ACLED usa la sintassi ":OR:" per combinare più nazioni
NAZIONI = "Georgia:OR:country=Armenia" 
ANNO = 2023

# ==========================================
# FUNZIONE 1: OTTENERE IL TOKEN (OAuth)
# ==========================================
def get_access_token(username, password):
    print("Richiesta dell'Access Token in corso...")
    token_url = "https://acleddata.com/oauth/token"
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = {
          'username': username,
          'password': password,
          'grant_type': "password",     # Valore fisso richiesto da ACLED
          'client_id': "acled",         # Valore fisso richiesto da ACLED
          'scope': "authenticated"      # Valore fisso richiesto da ACLED
    }

    response = requests.post(token_url, headers=headers, data=data)

    if response.status_code == 200:
        token_data = response.json()
        print("Token ottenuto con successo!")
        return token_data['access_token']
    else:
        raise Exception(f"Errore di Autenticazione: {response.status_code} {response.text}")


# ==========================================
# FUNZIONE 2: SCARICARE I DATI
# ==========================================
def scarica_dati(token):
    print(f"Scaricamento eventi per {NAZIONI} nell'anno {ANNO}...")
    
    url = "https://acleddata.com/api/acled/read"
    
    # Parametri passati come dizionario (Opzione #2 della documentazione)
    parameters = {
        "_format": "json",
        "country": NAZIONI,
        "year": ANNO,
        "limit": 5000  # Limite di sicurezza per non scaricare dataset immensi
    }
    
    # Inseriamo il token nell'intestazione (Header) della richiesta
    headers = {
        "Authorization": f"Bearer {token}", 
        "Content-Type": "application/json"
    }

    response = requests.get(url, params=parameters, headers=headers)
    
    if response.status_code == 200:
        dati_json = response.json()
        
        # ACLED inserisce i dati veri e propri sotto la chiave 'data'
        eventi = dati_json.get("data", [])
        print(f"Successo! Trovati {len(eventi)} eventi.")
        
        # Trasformiamo in un DataFrame Pandas
        df = pd.DataFrame(eventi)
        
        # Selezioniamo le colonne che ci interessano per il Knowledge Graph
        colonne_utili = [
            'event_id_cnty', 'event_date', 'event_type', 
            'actor1', 'actor2', 'country', 'location', 'fatalities'
        ]
        
        df_pulito = df[[c for c in colonne_utili if c in df.columns]]
        
        # Salviamo in CSV
        nome_file = f"dati_grezzi_acled.csv"
        df_pulito.to_csv(nome_file, index=False)
        print(f"File salvato correttamente: {nome_file}")
        
    else:
        print(f"Errore nello scaricamento: {response.status_code} {response.text}")

def scarica_dati_middle_east(token):
    print(f"Scaricamento eventi per la Region Code {REGION_CODE} dall'anno {ANNO_INIZIO}...")
    
    url = "https://acleddata.com/api/acled/read"
    
    headers = {
        "Authorization": f"Bearer {token}", 
        "Content-Type": "application/json"
    }

    tutti_gli_eventi = []
    pagina_corrente = 1
    limite_per_pagina = 5000 # Massimo consentito dall'API per singola chiamata
    
    # Ciclo di paginazione per superare il limite di 5000 righe
    while True:
        print(f"Scaricando la pagina {pagina_corrente}...")
        
        parameters = {
            "_format": "json",
            "region": REGION_CODE,
            "year": ANNO_INIZIO,
            "limit": limite_per_pagina,
            "page": pagina_corrente
        }
        
        response = requests.get(url, params=parameters, headers=headers)
        
        if response.status_code != 200:
            print(f"Errore nello scaricamento: {response.status_code} {response.text}")
            break
            
        dati_json = response.json()
        eventi = dati_json.get("data", [])
        
        # Condizione di uscita 1: L'API non restituisce più dati
        if not eventi:
            print("Estrazione completata. Nessun altro dato disponibile.")
            break
            
        tutti_gli_eventi.extend(eventi)
        
        # Condizione di uscita 2: La pagina contiene meno risultati del limite
        # Questo implica matematicamente che è l'ultima pagina disponibile
        if len(eventi) < limite_per_pagina:
            break
            
        pagina_corrente += 1
        time.sleep(1) # Pausa di 1 secondo per rispettare i rate limit di ACLED
        
    print(f"Successo totale! Trovati {len(tutti_gli_eventi)} eventi per il Medio Oriente.")
    
    # Trasformazione e pulizia in Pandas
    df = pd.DataFrame(tutti_gli_eventi)
    
    # Colonne ottimizzate per l'ingestion in un Knowledge Graph
    colonne_utili = [
        'event_id_cnty', 'event_date','disorder_type', 'event_type', 'sub_event_type',
        'actor1','assoc_actor1', 'inter1', 'actor2','assoc_actor2', 'inter2', 'iso',
        'country', 'location', 'latitude', 'longitude', 'source', 'source_scale',
        'fatalities', 'notes', 'tags'
    ]
    
    df_pulito = df[[c for c in colonne_utili if c in df.columns]]
    
    # Salvataggio
    nome_file = "dati_grezzi_acled_middle_east.csv"
    df_pulito.to_csv(nome_file, index=False)
    print(f"File salvato correttamente: {nome_file}")
    
    return df_pulito

# ==========================================
# FUNZIONE 3: BACKFILL STORICO MACRO-EVENTI
# ==========================================
def scarica_backfill_storico(token):
    print(f"Scaricamento backfill storico (2015-2022) per la Region Code {REGION_CODE}...")
    
    url = "https://acleddata.com/api/acled/read"
    headers = {
        "Authorization": f"Bearer {token}", 
        "Content-Type": "application/json"
    }

    tutti_gli_eventi = []
    limite_per_pagina = 5000
    
    for anno in range(2015, 2023):
        # 1. Scaricamento eventi di tipo "Strategic developments"
        pagina_corrente = 1
        while True:
            print(f"  [Anno {anno} - Strategic] Pagina {pagina_corrente}...")
            params_strategic = {
                "_format": "json",
                "region": REGION_CODE,
                "year": anno,
                "event_type": "Strategic developments",
                "limit": limite_per_pagina,
                "page": pagina_corrente
            }
            resp = requests.get(url, params=params_strategic, headers=headers)
            if resp.status_code != 200: break
            
            eventi = resp.json().get("data", [])
            if not eventi: break
            
            tutti_gli_eventi.extend(eventi)
            if len(eventi) < limite_per_pagina: break
            pagina_corrente += 1
            time.sleep(1)
            
        # 2. Scaricamento eventi ad alto impatto sistemico (Fatalities >= 50)
        pagina_corrente = 1
        while True:
            print(f"  [Anno {anno} - Fatalities >= 50] Pagina {pagina_corrente}...")
            params_fatalities = {
                "_format": "json",
                "region": REGION_CODE,
                "year": anno,
                "fatalities": 50,
                "fatalities_where": ">=",
                "limit": limite_per_pagina,
                "page": pagina_corrente
            }
            resp = requests.get(url, params=params_fatalities, headers=headers)
            if resp.status_code != 200: break
            
            eventi = resp.json().get("data", [])
            if not eventi: break
            
            tutti_gli_eventi.extend(eventi)
            if len(eventi) < limite_per_pagina: break
            pagina_corrente += 1
            time.sleep(1)

    print(f"Backfill completato! Trovati {len(tutti_gli_eventi)} eventi macro-storici.")
    
    if not tutti_gli_eventi:
        return pd.DataFrame()
        
    df = pd.DataFrame(tutti_gli_eventi)
    
    colonne_utili = [
        'event_id_cnty', 'event_date','disorder_type', 'event_type', 'sub_event_type',
        'actor1','assoc_actor1', 'inter1', 'actor2','assoc_actor2', 'inter2', 'iso',
        'country', 'location', 'latitude', 'longitude', 'source', 'source_scale',
        'fatalities', 'notes', 'tags'
    ]
    
    df_pulito = df[[c for c in colonne_utili if c in df.columns]]
    
    # Rimuoviamo duplicati (ad es. un evento 'Strategic developments' che ha anche >50 fatalities)
    df_pulito = df_pulito.drop_duplicates(subset=['event_id_cnty'])
    
    return df_pulito

# ==========================================
# ESECUZIONE DEL PROGRAMMA
# ==========================================
if __name__ == "__main__":
    try:
        mio_token = get_access_token(USERNAME, PASSWORD)
        
        # 1. Caricamento dati post-2023 già scaricati
        file_recente = "dati_grezzi_acled_middle_east.csv"
        if os.path.exists(file_recente):
            print(f"\nCaricamento dei dati recenti dal file: {file_recente}...")
            df_recenti = pd.read_csv(file_recente)
        else:
            print(f"\nFile {file_recente} non trovato. Scaricamento dati recenti (post-2023)...")
            df_recenti = scarica_dati_middle_east(mio_token)
        
        # 2. Esecuzione backfill storico
        print("\nAvvio del backfill storico a bassa risoluzione (2015-2022)...")
        df_storico = scarica_backfill_storico(mio_token)
        
        # 3. Unione e salvataggio
        print("\nUnione delle fonti dati in corso...")
        df_unito = pd.concat([df_storico, df_recenti], ignore_index=True)
        
        # Rimozione di eventuali duplicati globali e ordinamento per data
        df_unito = df_unito.drop_duplicates(subset=['event_id_cnty'])
        df_unito['event_date'] = pd.to_datetime(df_unito['event_date'])
        df_unito = df_unito.sort_values(by='event_date', ascending=False)
        
        nome_file_unito = "acled_unified_middle_east.csv"
        df_unito.to_csv(nome_file_unito, index=False)
        
        print(f"Successo totale! Dataset unico salvato: {nome_file_unito} ({len(df_unito)} eventi).")
        print("\nAnteprima dati uniti (testa):")
        print(df_unito.head(3))
        
    except Exception as e:
        print(f"Si è verificato un errore: {e}")