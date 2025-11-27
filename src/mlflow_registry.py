from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from src.config import MODEL_NAME

def register_model_to_registry(run_id, stage="Staging"):
    """
    Registra modelo no Model Registry.
    Cria o Registered Model se ele ainda não existir.
    """
    
    client = MlflowClient()
    
    try:
        print(f"Verificando se o modelo '{MODEL_NAME}' existe...")
        client.create_registered_model(MODEL_NAME)
        print(f"Modelo '{MODEL_NAME}' criado com sucesso.")
    except RestException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
            print(f"Modelo '{MODEL_NAME}' já existe. Criando nova versão...")
        else:
            # Se for outro erro, lançamos
            raise e
    
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Criando versão do modelo a partir de: {model_uri}")
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