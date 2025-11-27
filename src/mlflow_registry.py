from mlflow.tracking import MlflowClient
from src.config import MODEL_NAME

def register_model_to_registry(run_id, stage="Staging"):
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    
    registered_model = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id,
        description=f"Modelo treinado - Run {run_id}"
    )
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered_model.version,
        stage=stage
    )
    
    print(f"Modelo registrado. Versao: {registered_model.version}")
    return registered_model