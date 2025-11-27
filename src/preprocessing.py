from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

def get_preprocessor():
    """Cria o pipeline de pré-processamento do Scikit-Learn"""
    
    # 1. Numéricas: Preenche nulos com a média + Padroniza
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # 2. Categóricas: Preenche nulos + OneHotEncoding
    # handle_unknown='ignore' garante que categorias novas no teste não quebrem o modelo
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Junta tudo
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
        
    return preprocessor