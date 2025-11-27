from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn

def train_random_forest(df_train, df_val, preprocessor, params, run_name):
    """Treina Random Forest (Scikit-Learn)"""
    
    # Separar Features (X) e Alvo (y)
    X_train = df_train.drop(columns=["em_risco"])
    y_train = df_train["em_risco"]
    
    X_val = df_val.drop(columns=["em_risco"])
    y_val = df_val["em_risco"]

    with mlflow.start_run(run_name=run_name):
        # Cria um "Super Pipeline" que contém o pré-processamento E o modelo
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=params.get("numTrees", 100), 
                max_depth=params.get("maxDepth", 10),
                min_samples_leaf=params.get("minInstancesPerNode", 5),
                random_state=42,
                n_jobs=-1  # Usa todos os núcleos da CPU
            ))
        ])

        print("Treinando modelo (Fit)...")
        clf.fit(X_train, y_train)

        # Avaliação (Probabilidade da classe 1)
        y_pred_proba_train = clf.predict_proba(X_train)[:, 1]
        y_pred_proba_val = clf.predict_proba(X_val)[:, 1]
        
        auc_train = roc_auc_score(y_train, y_pred_proba_train)
        auc_val = roc_auc_score(y_val, y_pred_proba_val)

        # Logs
        mlflow.log_params(params)
        mlflow.log_metric("auc_train", auc_train)
        mlflow.log_metric("auc_validation", auc_val)
        
        # Logar o modelo (Pipeline completo)
        mlflow.sklearn.log_model(clf, "model")
        
        run_id = mlflow.active_run().info.run_id
        
        print(f"Modelo treinado - Run ID: {run_id}")
        print(f"AUC Treino: {auc_train:.4f} | AUC Validacao: {auc_val:.4f}")
        
        return clf, run_id, auc_val