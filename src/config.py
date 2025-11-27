import os
from dotenv import load_dotenv

load_dotenv()

# GCP
GCP_PROJECT = os.getenv("GCP_PROJECT") # Lembre que o cloudbuild.yaml já injeta isso
DATASET = os.getenv("DATASET", "main_database")
REGION = "us-central1"
BQ_LOCATION = "us-central1" # Confirme se é US ou us-central1 no console

# --- MUDANÇA AQUI ---
# Coloque o nome EXATO da sua tabela única aqui
TABLE_NAME = "dataset_silver" 
# --------------------

# MLflow e Modelo (Mantém igual)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_EXPERIMENT = "CNPJ-Risk-Prediction"
MODEL_NAME = "CNPJ-Risk-RandomForest"

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

DEFAULT_PARAMS = {
    "numTrees": 100,
    "maxDepth": 10,
    "minInstancesPerNode": 5
}

MIN_AUC_THRESHOLD = 0.70