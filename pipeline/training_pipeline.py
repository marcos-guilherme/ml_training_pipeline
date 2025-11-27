import mlflow
import os
import traceback
from src.config import MLFLOW_TRACKING_URI, MLFLOW_USERNAME, MLFLOW_PASSWORD, MLFLOW_EXPERIMENT, DEFAULT_PARAMS, MIN_AUC_THRESHOLD
from src.data_loader import get_data_splits
from src.preprocessing import get_preprocessor
from src.model_training import train_random_forest
from src.model_evaluation import evaluate_on_test # Lembre de atualizar este para sklearn também!
from src.mlflow_registry import register_model_to_registry

class TrainingPipeline:
    
    def __init__(self):
        self.setup_mlflow()
    
    def setup_mlflow(self):
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    def run(self, params=None):
        try:
            params = params or DEFAULT_PARAMS
            print(f"Iniciando Treino LIGHT (Pandas). Params: {params}")
            
            # 1. Carregar dados (Limitados a 100k)
            print("1. Carregando dados...")
            df_train, df_val, df_test = get_data_splits()
            
            # 2. Pré-processador
            print("2. Criando processador...")
            preprocessor = get_preprocessor()
            
            print("3. Treinando...")
            from src.model_training import train_random_forest

            model, run_id, auc_val = train_random_forest(
                        df_train, df_val, preprocessor, params, run_name="Full_Metrics_Run"
                    )
            
            # 4. Avaliar
            # Reabrimos a run pelo ID para adicionar as métricas de teste
            with mlflow.start_run(run_id=run_id):
                print("4. Avaliando no teste (Anexando à mesma Run)...")
                from src.model_evaluation import evaluate_on_test
                auc_test = evaluate_on_test(model, df_test)

            # 5. Registrar
            if auc_test < MIN_AUC_THRESHOLD:
                print(f"Performance baixa ({auc_test}). Não registrando.")
                return {"status": "failed", "reason": "auc_low"}
                
            print("5. Registrando no MLflow...")
            register_model_to_registry(run_id)
            
            return {
                "status": "success",
                "run_id": run_id,
                "auc_test": auc_test
            }

        except Exception as e:
            print(f"Erro fatal: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def close(self):
        pass