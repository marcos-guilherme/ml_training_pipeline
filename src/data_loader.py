from google.cloud import bigquery
from src.config import GCP_PROJECT, DATASET, TABLE_NAME, BQ_LOCATION

def load_full_dataset():
    """Carrega uma janela de dados da tabela única"""
    client = bigquery.Client(project=GCP_PROJECT, location=BQ_LOCATION)
    
    # 140k linhas no total (100k treino + 20k val + 20k teste)
    LIMIT = 140000
    table_ref = f"`{GCP_PROJECT}.{DATASET}.{TABLE_NAME}`"
    
    # Ordena por data DESC (do mais novo para o mais velho)
    query = f"""
        SELECT * FROM {table_ref}
        ORDER BY data_ref DESC
        LIMIT {LIMIT}
    """
    
    print(f"Carregando {LIMIT} linhas da tabela {TABLE_NAME}...")
    df = client.query(query).to_dataframe()
    
    # Tratamento UF (Array -> String)
    if 'uf' in df.columns:
        df['uf'] = df['uf'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
        df['uf'] = df['uf'].astype(str)
        
    print(f"Total carregado: {len(df)} linhas.")
    return df

def get_data_splits():
    """Divide o dataframe único em Treino, Validação e Teste"""
    df = load_full_dataset()
    
    if len(df) < 1000:
        raise ValueError("Poucos dados para dividir! Verifique a tabela.")

    # Como ordenamos DESC (novos primeiro), o fatiamento é:
    # 0 a 20k: Teste (O futuro)
    # 20k a 40k: Validação
    # 40k em diante: Treino (O passado)
    
    test_size = 20000
    val_size = 20000
    
    # Ajuste de segurança caso tenha menos de 140k linhas
    if len(df) < (test_size + val_size + 100):
        # Se tiver poucos dados, faz divisão percentual simples (20% teste, 20% val)
        test_end = int(len(df) * 0.2)
        val_end = int(len(df) * 0.4)
    else:
        test_end = test_size
        val_end = test_size + val_size

    # Fatiamento (Slicing)
    df_test = df.iloc[0:test_end].copy()
    df_val = df.iloc[test_end:val_end].copy()
    df_train = df.iloc[val_end:].copy()
    
    print(f"Splits criados:")
    print(f"Treino: {len(df_train)} (Passado)")
    print(f"Validacao: {len(df_val)}")
    print(f"Teste: {len(df_test)} (Futuro/Recente)")
    
    return df_train, df_val, df_test