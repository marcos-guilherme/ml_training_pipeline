from google.cloud import bigquery
from src.config import GCP_PROJECT, DATASET, SPLITS

def load_split(client, split_name):
    """
    Carrega dados do BigQuery com limite de linhas para economizar memória.
    """
    table_ref = f"`{GCP_PROJECT}.{DATASET}.{split_name}`"
    
    # LIMITES DE AMOSTRAGEM
    # Treino: 100k linhas 
    # Validação/Teste: 20k linhas (suficiente para estatística confiável)
    limit = 100000 if split_name == "treino" else 20000
    
    query = f"""
        SELECT * FROM {table_ref}
        LIMIT {limit}
    """
    
    print(f"Carregando {split_name} (Limit: {limit})...")
    
    # O BigQuery converte para Pandas automaticamente
    df = client.query(query).to_dataframe()
    
    print(f"Carregado {split_name}: {len(df)} registros")
    return df

def load_all_splits():
    """Carrega todos os splits"""
    client = bigquery.Client(project=GCP_PROJECT)
    
    splits = {}
    for split_type, split_name in SPLITS.items():
        splits[split_type] = load_split(client, split_name)
    
    return splits["treino"], splits["validacao"], splits["teste"]