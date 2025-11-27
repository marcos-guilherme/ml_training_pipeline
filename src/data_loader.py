from google.cloud import bigquery
from src.config import GCP_PROJECT, DATASET, SPLITS, BQ_LOCATION

def load_split(client, split_name):
    """Carrega uma tabela já pronta do BigQuery"""
    
    table_id = SPLITS[split_name]
    table_ref = f"`{GCP_PROJECT}.{DATASET}.{table_id}`"
    
    limit_clause = "LIMIT 100000" if split_name == "treino" else "LIMIT 20000"

    query = f"""
        SELECT * FROM {table_ref}
        {limit_clause}
    """
    
    print(f"Carregando tabela {table_id}...")
    df = client.query(query).to_dataframe()
    
    # Tratamento UF (Array -> String) 
    if 'uf' in df.columns:
        df['uf'] = df['uf'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
        df['uf'] = df['uf'].astype(str)

    for col in ['total_debito', 'em_risco']:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    print(f"Carregado {split_name}: {len(df)} registros.")
    return df

def get_data_splits(): # Mantive o nome da função pra não quebrar o pipeline.py
    """Carrega as 3 tabelas prontas"""
    client = bigquery.Client(project=GCP_PROJECT, location=BQ_LOCATION)
    
    df_train = load_split(client, "treino")
    df_val = load_split(client, "validacao")
    df_test = load_split(client, "teste")
    
    return df_train, df_val, df_test