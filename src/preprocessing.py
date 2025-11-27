from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql.functions import col
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

def prepare_data(df, fit_pipeline=False, pipeline_obj=None):
    """Prepara dados com feature engineering"""
    
    # Limpar nulos
    df_clean = df.dropna(subset=["em_risco"])
    
    stages = []
    
    # StringIndexer + OneHotEncoder para categóricas
    for feature in CATEGORICAL_FEATURES:
        indexer = StringIndexer(
            inputCol=feature, 
            outputCol=f"{feature}_indexed",
            handleInvalid="skip"
        )
        encoder = OneHotEncoder(
            inputCol=f"{feature}_indexed",
            outputCol=f"{feature}_encoded"
        )
        stages.append(indexer)
        stages.append(encoder)
    
    # VectorAssembler
    feature_cols = [f"{f}_encoded" for f in CATEGORICAL_FEATURES] + NUMERIC_FEATURES
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    stages.append(assembler)
    
    # StandardScaler
    scaler = StandardScaler(
        inputCol="features",
        outputCol="features_scaled"
    )
    stages.append(scaler)
    
    # Executar pipeline
    pipeline = Pipeline(stages=stages)
    
    if fit_pipeline:
        pipeline_model = pipeline.fit(df_clean)
        df_prepared = pipeline_model.transform(df_clean)
        return df_prepared.select("features_scaled", "em_risco"), pipeline_model
    else:
        df_prepared = pipeline_obj.transform(df_clean)
        return df_prepared.select("features_scaled", "em_risco")

def prepare_all_splits(df_train, df_val, df_test):
    """Prepara todos os splits"""
    print("Preparando dados...")
    
    df_train_prep, pipeline_model = prepare_data(df_train, fit_pipeline=True)
    df_val_prep = prepare_data(df_val, pipeline_obj=pipeline_model)
    df_test_prep = prepare_data(df_test, pipeline_obj=pipeline_model)
    
    print("Dados preparados")
    print(f"Treino: {df_train_prep.count()}")
    print(f"Validacao: {df_val_prep.count()}")
    print(f"Teste: {df_test_prep.count()}")
    
    return df_train_prep, df_val_prep, df_test_prep
