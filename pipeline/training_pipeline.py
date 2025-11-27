from pyspark.sql import SparkSession
import mlflow
import os
from src.config import MLFLOW_TRACKING_URI, MLFLOW_USERNAME, MLFLOW_PASSWORD, MLFLOW_EXPERIMENT, DEFAULT_PARAMS, MIN_AUC_THRESHOLD
from src.data_loader import load_all_splits
from src.preprocessing import prepare_all_splits
from src.model_training import train_random_forest
from src.model_evaluation import evaluate_on_test
from src.mlflow_registry import register_model_to_registry

class TrainingPipeline:
    
    def __init__(self):
            jar_path = "/app/jars/spark-bigquery-with-dependencies.jar"

            self.spark = SparkSession.builder \
                .appName("CNPJ-Risk-Training") \
                .config("spark.sql.shuffle.partitions", "200") \
                .config("spark.jars", jar_path) \
                .getOrCreate()
            
            self.setup_mlflow()
    
    def setup_mlflow(self):
        """Configurar MLflow"""
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        
        print("MLflow configurado")
    
    def run(self, params=None):
        """Executar pipeline completo"""
        
        try:
            params = params or DEFAULT_PARAMS
            
            print("Iniciando pipeline de treinamento...")
            print(f"Parametros: {params}")
            
            # 1. Carregar dados
            print("\n1. Carregando dados...")
            df_train, df_val, df_test = load_all_splits(self.spark)
            
            # 2. Preparar dados
            print("\n2. Preparando dados...")
            df_train_prep, df_val_prep, df_test_prep = prepare_all_splits(df_train, df_val, df_test)
            
            # 3. Treinar modelo
            print("\n3. Treinando modelo...")
            model, run_id, auc_val = train_random_forest(
                df_train_prep,
                df_val_prep,
                params,
                run_name="AutomatedTraining"
            )
            
            # 4. Avaliar no teste
            print("\n4. Avaliando no teste...")
            auc_test = evaluate_on_test(model, df_test_prep)
            
            # 5. Validar threshold
            print("\n5. Validando performance...")
            if auc_test < MIN_AUC_THRESHOLD:
                print(f"Aviso: AUC {auc_test:.4f} abaixo do threshold {MIN_AUC_THRESHOLD}")
                print("Modelo nao registrado")
                return {"status": "failed", "reason": "auc_threshold"}
            
            # 6. Registrar modelo
            print("\n6. Registrando modelo...")
            registered_model = register_model_to_registry(run_id, stage="Staging")
            
            print("\nPipeline completado com sucesso!")
            
            return {
                "status": "success",
                "run_id": run_id,
                "model_version": registered_model.version,
                "auc_validation": auc_val,
                "auc_test": auc_test
            }
        
        except Exception as e:
            print(f"Erro no pipeline: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def close(self):
        """Fechar Spark"""
        self.spark.stop()
