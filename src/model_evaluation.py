from pyspark.ml.evaluation import BinaryClassificationEvaluator

def evaluate_on_test(model, df_test):
    """Avalia modelo no teste"""
    
    pred_test = model.transform(df_test)
    
    evaluator = BinaryClassificationEvaluator(
        labelCol="em_risco",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    auc_test = evaluator.evaluate(pred_test)
    
    print(f"AUC Teste: {auc_test:.4f}")
    
    return auc_test
