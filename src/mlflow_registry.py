from mlflow.tracking import MlflowClient
from src.config import MODEL_NAME

def register_model_to_registry(run_id, stage="Staging"):
    """Registra modelo no Model Registry"""
    
    client = MlflowClient()
    
    model_uri = f"runs:/{run_id}/model"
    
    registered_model = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id,
        description=f"Modelo treinado - Run {run_id}"
    )
    
    # Transicionar stage
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered_model.version,
        stage=stage
    )
    
    print(f"Modelo registrado no registry")
    print(f"Nome: {MODEL_NAME}")
    print(f"Versao: {registered_model.version}")
    print(f"Stage: {stage}")
    
    return registered_model
