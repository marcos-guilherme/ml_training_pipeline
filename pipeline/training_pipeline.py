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
            
            # 3. Treinar
            print("3. Treinando...")
            model, run_id, auc_val = train_random_forest(
                df_train, df_val, preprocessor, params, run_name="Pandas_100k"
            )
            
            # 4. Avaliar (Você precisará adaptar o model_evaluation.py para sklearn também)
            # Como o modelo é um pipeline, ele já faz o transform internamente no predict
            # auc_test = evaluate_on_test(model, df_test) 
            # (Simplificando aqui para garantir o fluxo):
            from sklearn.metrics import roc_auc_score
            auc_test = roc_auc_score(df_test["em_risco"], model.predict_proba(df_test.drop(columns="em_risco"))[:, 1])
            print(f"AUC Teste: {auc_test:.4f}")

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