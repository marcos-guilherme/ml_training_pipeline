from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

def train_random_forest(df_train, df_val, params, run_name):
    """Treina Random Forest com MLflow"""
    
    with mlflow.start_run(run_name=run_name):
        # Log parâmetros
        mlflow.log_params(params)
        
        # Treinar
        rf = RandomForestClassifier(
            numTrees=params["numTrees"],
            maxDepth=params["maxDepth"],
            minInstancesPerNode=params["minInstancesPerNode"],
            seed=42,
            labelCol="em_risco",
            featuresCol="features_scaled"
        )
        
        model = rf.fit(df_train)
        
        # Avaliação
        evaluator = BinaryClassificationEvaluator(
            labelCol="em_risco",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        pred_train = model.transform(df_train)
        auc_train = evaluator.evaluate(pred_train)
        
        pred_val = model.transform(df_val)
        auc_val = evaluator.evaluate(pred_val)
        
        # Log métricas
        mlflow.log_metric("auc_train", float(auc_train))
        mlflow.log_metric("auc_validation", float(auc_val))
        
        # Log modelo
        mlflow.spark.log_model(model, "model")
        
        run_id = mlflow.active_run().info.run_id
        
        print(f"Modelo treinado - Run ID: {run_id}")
        print(f"AUC Treino: {auc_train:.4f}")
        print(f"AUC Validacao: {auc_val:.4f}")
        
        return model, run_id, auc_val
