from sklearn.metrics import roc_auc_score
import pandas as pd

def evaluate_on_test(model, df_test):
    """
    Avalia modelo no teste (Versão Scikit-Learn)
    """
    print("Avaliando modelo no set de teste...")
    
    # 1. Separar Features (X) e Alvo (y)
    # Removemos a coluna alvo para passar pro modelo prever
    X_test = df_test.drop(columns=["em_risco"])
    y_test = df_test["em_risco"]
    
    # 2. Fazer Predição de Probabilidade
    # O modelo é um Pipeline, então ele aplica o pré-processamento automaticamente aqui
    # model.predict_proba retorna uma matriz com [prob_classe_0, prob_classe_1]
    # Nós queremos apenas a probabilidade da classe 1 (índice [:, 1])
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 3. Calcular AUC
    auc_test = roc_auc_score(y_test, y_pred_proba)
    
    print(f"AUC Teste: {auc_test:.4f}")
    
    return auc_test