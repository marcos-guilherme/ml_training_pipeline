import os
from dotenv import load_dotenv

load_dotenv()

# GCP
GCP_PROJECT = os.getenv("GCP_PROJECT") 
DATASET = os.getenv("DATASET", "main_database")
REGION = "us-central1"
BQ_LOCATION = "us-central1"


TABLE_NAME = "dataset_silver" 


# MLflow e Modelo
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_EXPERIMENT = "CNPJ-Risk-Prediction"
MODEL_NAME = "CNPJ-Risk-RandomForest"

CATEGORICAL_FEATURES = ["cnae_fiscal_principal", "uf"]
NUMERIC_FEATURES = [
    "tempo_atividade_anos",
    "total_debito",      # Criada no SQL
    "mudou_situacao",    # Criada no SQL
    "tem_debito_governo",# Criada no SQL
    "tem_acao_judicial"  # Criada no SQL
]

SPLITS = {
    "treino": "treino",
    "validacao": "validacao",
    "teste": "teste"
}

DEFAULT_PARAMS = {"numTrees": 100, "maxDepth": 10, "minInstancesPerNode": 5}
MIN_AUC_THRESHOLD = 0.70