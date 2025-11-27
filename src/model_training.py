import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def train_random_forest(df_train, df_val, preprocessor, params, run_name=None):
    X_train = df_train.drop(columns=["em_risco"])
    y_train = df_train["em_risco"]
    X_val = df_val.drop(columns=["em_risco"])
    y_val = df_val["em_risco"]

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=params.get("numTrees", 100), 
            max_depth=params.get("maxDepth", 10),
            min_samples_leaf=params.get("minInstancesPerNode", 5),
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Treinando modelo...")
    clf.fit(X_train, y_train)

    y_pred_val = clf.predict(X_val)
    y_pred_proba_val = clf.predict_proba(X_val)[:, 1]
    
    metrics = {
        "auc_val": roc_auc_score(y_val, y_pred_proba_val),
        "accuracy_val": accuracy_score(y_val, y_pred_val),
        "precision_val": precision_score(y_val, y_pred_val, zero_division=0),
        "f1_val": f1_score(y_val, y_pred_val, zero_division=0)
    }
    
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    
    # Gerar artefatos graficos
    print("Gerando graficos...")
    
    # Matriz de Confusao
    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusao')
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # Feature Importance
    try:
        feature_names = preprocessor.get_feature_names_out()
        importances = clf.named_steps['classifier'].feature_importances_
        feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feat_df = feat_df.sort_values(by='importance', ascending=False).head(20)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x='importance', y='feature', data=feat_df, palette='viridis')
        plt.title('Top Features')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
        mlflow.log_artifact("feature_importance.png")
    except Exception as e:
        print(f"Erro plot features: {e}")

    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.log_model(clf, "model", signature=signature)
    
    if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")
    if os.path.exists("feature_importance.png"): os.remove("feature_importance.png")
    
    run_id = mlflow.active_run().info.run_id
    
    return clf, run_id, metrics["auc_val"]