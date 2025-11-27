import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.models.signature import infer_signature

# Importamos as listas de features que definimos como "Oficiais"
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

def train_generic_model(df_train, df_val, model_strategy, preprocessor, run_name=None):
    """
    Treina o modelo garantindo que APENAS as features selecionadas entrem.
    """
    
    selected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    print(f"Selecionando apenas as {len(selected_features)} features oficiais...")

    cols_to_keep = selected_features + ["em_risco"]
    
    df_train = df_train[cols_to_keep]
    df_val = df_val[cols_to_keep]

    X_train = df_train.drop(columns=["em_risco"])
    y_train = df_train["em_risco"]
    
    X_val = df_val.drop(columns=["em_risco"])
    y_val = df_val["em_risco"]

    clf = model_strategy.build_pipeline(preprocessor)

    print(f"Treinando estratégia: {model_strategy.__class__.__name__}...")
    clf.fit(X_train, y_train)

    y_pred_val = clf.predict(X_val)
    y_pred_proba_val = clf.predict_proba(X_val)[:, 1]
    
    metrics = {
        "auc_val": roc_auc_score(y_val, y_pred_proba_val),
        "accuracy_val": accuracy_score(y_val, y_pred_val),
        "precision_val": precision_score(y_val, y_pred_val, zero_division=0),
        "recall_val": recall_score(y_val, y_pred_val, zero_division=0),
        "f1_val": f1_score(y_val, y_pred_val, zero_division=0)
    }
    
    mlflow.log_params(model_strategy.get_params())
    mlflow.log_metrics(metrics)
    
    print("Gerando gráficos...")
    try:
        cm = confusion_matrix(y_val, y_pred_val, normalize='true')
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.1%', 
            cmap='RdBu_r', 
            cbar=True,
            xticklabels=['Sem Risco', 'Com Risco'],
            yticklabels=['Sem Risco', 'Com Risco']
        )
        plt.title('Matriz de Confusão (Normalizada)')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")
    except Exception as e:
        print(f"Erro grafico CM: {e}")

    if hasattr(clf.named_steps['classifier'], 'feature_importances_'):
        try:
            feature_names = preprocessor.get_feature_names_out()
            importances = clf.named_steps['classifier'].feature_importances_
            
            feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feat_df = feat_df.sort_values(by='importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feat_df, palette='viridis')
            plt.title('Top 20 Variáveis Mais Importantes')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            plt.close()
            mlflow.log_artifact("feature_importance.png")
        except Exception as e:
            pass

 
    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow_sklearn.log_model(clf, "model", signature=signature)
    
    if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")
    if os.path.exists("feature_importance.png"): os.remove("feature_importance.png")
    
    run_id = mlflow.active_run().info.run_id
    
    return clf, run_id, metrics["auc_val"]