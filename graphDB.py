import requests

def upload_to_graphdb(file_path, repo_id):
    print(f"Caricamento di {file_path} nel repository '{repo_id}' di GraphDB...")
    
    # Endpoint standard di GraphDB per l'inserimento dati via REST
    url = f"http://localhost:7200/repositories/{repo_id}/statements"
    
    headers = {
        "Content-Type": "text/turtle"
    }
    
    try:
        with open(file_path, 'rb') as f:
            response = requests.put(url, data=f, headers=headers)
            
        if response.status_code in [200, 204]:
            print("KG caricato con successo su GraphDB localhost!")
            print(f"Url: {url}")
        else:
            print(f"Errore durante il caricamento: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Impossibile connettersi a GraphDB: {e}")

if __name__ == '__main__':

    upload_to_graphdb('./cle/acled_populated_kg.ttl', 'MiddleEastConflict')