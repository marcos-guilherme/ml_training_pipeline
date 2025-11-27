from google.cloud import bigquery
from src.config import GCP_PROJECT, DATASET, SPLITS

def load_split(client, split_name):
    table_ref = f"`{GCP_PROJECT}.{DATASET}.{split_name}`"
    limit = 100000 if split_name == "treino" else 20000
    
    # Ordena pela data para pegar os mais recentes
    query = f"""
        SELECT * FROM {table_ref}
        ORDER BY data_ref DESC
        LIMIT {limit}
    """
    
    print(f"Carregando {split_name}...")
    
    # O BigQuery baixa os dados
    df = client.query(query).to_dataframe()
    

    if 'uf' in df.columns:
        print("Tratando coluna UF (Array)...")
        # Pega o primeiro elemento da lista, se houver. Se for vazia, põe None.
        df['uf'] = df['uf'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
        # Garante que virou string
        df['uf'] = df['uf'].astype(str)
    
    print(f"Carregado {split_name}: {len(df)} registros.")
    return df

def load_all_splits():
    client = bigquery.Client(project=GCP_PROJECT)
    splits = {}
    for split_type, split_name in SPLITS.items():
        splits[split_type] = load_split(client, split_name)
    return splits["treino"], splits["validacao"], splits["teste"]