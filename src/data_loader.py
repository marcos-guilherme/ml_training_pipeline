from google.cloud import bigquery
from src.config import GCP_PROJECT, DATASET, SPLITS

def load_split(client, split_name):
    """
    Carrega a 'janela' mais recente de dados do BigQuery.
    """
    table_ref = f"`{GCP_PROJECT}.{DATASET}.{split_name}`"
    
    # Treino: 100k mais recentes
    # Validação/Teste: 20k mais recentes
    limit = 100000 if split_name == "treino" else 20000
    
    # AQUI ESTÁ A MÁGICA:
    # ORDER BY data_ref DESC -> Ordena do mais novo para o mais velho
    # LIMIT -> Pega apenas o topo da lista (os mais novos)
    query = f"""
        SELECT * FROM {table_ref}
        ORDER BY data_ref DESC
        LIMIT {limit}
    """
    
    print(f"Carregando {split_name} (Janela Recente de {limit} linhas)...")
    
    # O BigQuery usa o Storage API para baixar rápido
    df = client.query(query).to_dataframe()
    
    print(f"Carregado {split_name}: {len(df)} registros. De {df['data_ref'].min()} até {df['data_ref'].max()}")
    return df

def load_all_splits():
    """Carrega todos os splits"""
    client = bigquery.Client(project=GCP_PROJECT)
    
    splits = {}
    for split_type, split_name in SPLITS.items():
        splits[split_type] = load_split(client, split_name)
    
    return splits["treino"], splits["validacao"], splits["teste"]