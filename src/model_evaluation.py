from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import mlflow

def evaluate_on_test(model, df_test):
    """
    Avalia modelo no teste e loga métricas no MLflow (na mesma Run ativa ou nova)
    """
    print("Avaliando no set de Teste...")
    
    X_test = df_test.drop(columns=["em_risco"])
    y_test = df_test["em_risco"]
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Dicionário de métricas
    test_metrics = {
        "test_auc": roc_auc_score(y_test, y_pred_proba),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Tenta logar no MLflow se houver uma run ativa
    if mlflow.active_run():
        mlflow.log_metrics(test_metrics)
        print("Métricas de teste logadas na Run ativa.")
    
    print(f"Resultados Teste: {test_metrics}")
    
    return test_metrics["test_auc"] # Retorna AUC para verificação de threshold