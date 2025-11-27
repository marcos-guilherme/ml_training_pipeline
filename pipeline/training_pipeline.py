import mlflow
import os
import traceback
from src.config import MLFLOW_TRACKING_URI, MLFLOW_USERNAME, MLFLOW_PASSWORD, MLFLOW_EXPERIMENT, DEFAULT_PARAMS, MIN_AUC_THRESHOLD
from src.data_loader import get_data_splits
from src.preprocessing import get_preprocessor
from src.mlflow_registry import register_model_to_registry
from src.models.factory import ModelFactory
from src.model_training import train_generic_model

class TrainingPipeline:
    
    def __init__(self, experiment_name=None):
        # Permite sobrescrever o nome do experimento
        self.experiment_name = experiment_name or MLFLOW_EXPERIMENT
        self.setup_mlflow()
    
    def setup_mlflow(self):
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"Definindo experimento: {self.experiment_name}")
        mlflow.set_experiment(self.experiment_name)
    
    def run(self, params=None):
        try:
            params = params or DEFAULT_PARAMS
            model_type = params.get("model_type", "random_forest")
            
            print(f"Iniciando Pipeline. Modelo: {model_type}. Params: {params}")
            
            print("1. Carregando dados...")
            df_train, df_val, df_test = get_data_splits()
            
            print("2. Preprocessamento...")
            preprocessor = get_preprocessor()
            
            model_strategy = ModelFactory.get_strategy(model_type, params)
            
            with mlflow.start_run(run_name=f"Train_{model_type}") as run:
                run_id = run.info.run_id
                
                # 4. Treina usando o executor generico
                from src.model_training import train_generic_model
                model, _, auc_val = train_generic_model(
                    df_train, df_val, model_strategy, preprocessor
                )
                
                print("5. Avaliando teste...")
                from src.model_evaluation import evaluate_on_test
                auc_test = evaluate_on_test(model, df_test)

                if auc_test < MIN_AUC_THRESHOLD:
                    print(f"Performance baixa ({auc_test}). Descartado.")
                    mlflow.set_tag("status_modelo", "descartado")
                    return {"status": "failed", "reason": "auc_low"}
                
                print("Modelo Aprovado. Registrando...")
                mlflow.set_tag("status_modelo", "aprovado")
                register_model_to_registry(run_id)
                
                return {
                    "status": "success",
                    "run_id": run_id,
                    "auc_test": auc_test,
                    "model_type": model_type
                }

        except Exception as e:
            print(f"Erro fatal: {str(e)}")
            traceback.print_exc()
            if mlflow.active_run():
                mlflow.end_run()
            return {"status": "error", "message": str(e)}

    def close(self):
        pass