from google.cloud import bigquery
import pandas as pd
import numpy as np
from src.config import GCP_PROJECT, DATASET, SPLITS, BQ_LOCATION

def optimize_floats(df):
    """Converte float64 para float32 para economizar 50% de RAM nas numéricas"""
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].astype('float32')
    return df

def optimize_objects(df):
    """Converte strings repetitivas em Categoria para economizar até 80% de RAM"""

    cat_cols = ['uf', 'cnae_fiscal_principal', 'mudou_situacao', 'tem_debito_governo', 'tem_acao_judicial', 'em_risco']
    
    for col in cat_cols:
        if col in df.columns:
            # O tipo 'category' usa inteiros por trás, gastando muito menos memória que 'object'
            df[col] = df[col].astype('category')
    return df

def load_split(client, split_name):
    table_id = SPLITS[split_name]
    table_ref = f"`{GCP_PROJECT}.{DATASET}.{table_id}`"
    
    limit_clause = "LIMIT 100000" if split_name == "treino" else "LIMIT 20000"
    
    query = f"""
        SELECT * EXCEPT(uf), 
        CAST(uf[SAFE_OFFSET(0)] AS STRING) as uf 
        FROM {table_ref} 
        ORDER BY data_ref DESC
        {limit_clause}
    """
    
    print(f"Carregando {table_id} (Ordenado por data)...")
    
    df = client.query(query).to_dataframe(create_bqstorage_client=True)
    
    # Otimização de tipos (Low Memory)
    df = optimize_floats(df)
    df = optimize_objects(df)

    print(f"Carregado {split_name}: {len(df)} registros.")
    return df

def get_data_splits():
    client = bigquery.Client(project=GCP_PROJECT, location=BQ_LOCATION)
    
    df_train = load_split(client, "treino")
    df_val = load_split(client, "validacao")
    df_test = load_split(client, "teste")
    
    return df_train, df_val, df_test