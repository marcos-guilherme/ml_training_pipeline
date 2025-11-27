import os
from dotenv import load_dotenv

load_dotenv()

# GCP
GCP_PROJECT = os.getenv("GCP_PROJECT")
DATASET = os.getenv("DATASET", "main_database")
REGION = "us-central1"

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_EXPERIMENT = "CNPJ-Risk-Prediction"

# Modelo
MODEL_NAME = "CNPJ-Risk-RandomForest"
SPLITS = {
    "treino": "treino",
    "validacao": "validacao",
    "teste": "teste"
}

# Features
CATEGORICAL_FEATURES = ["cnae_fiscal_principal", "uf"]
NUMERIC_FEATURES = [
    "tempo_atividade_anos",
    "pgfn_fgts_valor_acumulado_t_minus_1",
    "pgfn_naoprev_valor_acumulado_t_minus_1",
    "pgfn_prev_valor_acumulado_t_minus_1",
    "mudou_situacao",
    "tem_debito_governo",
    "total_debito",
    "tem_acao_judicial"
]

# Hiperparâmetros padrão
DEFAULT_PARAMS = {
    "numTrees": int(os.getenv("NUM_TREES", 100)),
    "maxDepth": int(os.getenv("MAX_DEPTH", 15)),
    "minInstancesPerNode": int(os.getenv("MIN_INSTANCES_PER_NODE", 5))
}

# Threshold de performance
MIN_AUC_THRESHOLD = 0.70
