import pandas as pd
import numpy as np
import matplotlib
# Configura Matplotlib para não tentar abrir janela (erro comum em serverless)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def train_random_forest(df_train, df_val, preprocessor, params, run_name):
    """Treina modelo, calcula métricas completas e salva gráficos"""
    
    X_train = df_train.drop(columns=["em_risco"])
    y_train = df_train["em_risco"]
    
    X_val = df_val.drop(columns=["em_risco"])
    y_val = df_val["em_risco"]

    with mlflow.start_run(run_name=run_name) as run:
        # 1. Pipeline e Treino
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

        # 2. Predições (Probabilidade e Classe)
        y_pred_val = clf.predict(X_val)
        y_pred_proba_val = clf.predict_proba(X_val)[:, 1]
        
        # 3. Métricas Numéricas
        metrics = {
            "auc_val": roc_auc_score(y_val, y_pred_proba_val),
            "accuracy_val": accuracy_score(y_val, y_pred_val),
            "precision_val": precision_score(y_val, y_pred_val, zero_division=0),
            "recall_val": recall_score(y_val, y_pred_val, zero_division=0),
            "f1_val": f1_score(y_val, y_pred_val, zero_division=0)
        }
        
        # Loga parâmetros e métricas
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # 4. ARTEFATO 1: Matriz de Confusão
        print("Gerando Matriz de Confusão...")
        cm = confusion_matrix(y_val, y_pred_val)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Matriz de Confusão (Validação)')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path) # Envia pro MLflow

        # 5. ARTEFATO 2: Feature Importance
        print("Gerando Feature Importance...")
        try:
            # Tenta extrair nomes das features do pré-processador
            feature_names = preprocessor.get_feature_names_out()
            importances = clf.named_steps['classifier'].feature_importances_
            
            # Cria DataFrame para plotar
            feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feat_df = feat_df.sort_values(by='importance', ascending=False).head(20) # Top 20
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feat_df, palette='viridis')
            plt.title('Top 20 Features Mais Importantes')
            plt.tight_layout()
            
            fi_path = "feature_importance.png"
            plt.savefig(fi_path)
            plt.close()
            mlflow.log_artifact(fi_path)
        except Exception as e:
            print(f"Aviso: Não foi possível gerar plot de feature importance: {e}")

        # 6. Log do Modelo com Assinatura (Schema)
        # Isso ajuda o MLflow a saber quais tipos de dados o modelo espera
        signature = infer_signature(X_train, clf.predict(X_train))
        mlflow.sklearn.log_model(clf, "model", signature=signature)
        
        # Limpeza de arquivos temporários
        if os.path.exists(cm_path): os.remove(cm_path)
        if os.path.exists("feature_importance.png"): os.remove("feature_importance.png")
        
        print(f"Run ID: {run.info.run_id}")
        print(f"Métricas: {metrics}")
        
        return clf, run.info.run_id, metrics["auc_val"]