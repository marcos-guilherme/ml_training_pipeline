from pyspark.sql import SparkSession
from src.config import GCP_PROJECT, DATASET, SPLITS

def load_split(spark, split_name):
    """Carrega um split do BigQuery"""
    table = f"`{GCP_PROJECT}.{DATASET}.{split_name}`"
    df = spark.read.format("bigquery").option("table", table).load()
    
    record_count = df.count()
    print(f"Carregado {split_name}: {record_count} registros")
    
    if record_count == 0:
        raise ValueError(f"Split {split_name} vazio!")
    
    return df

def load_all_splits(spark):
    """Carrega todos os splits"""
    splits = {}
    for split_type, split_name in SPLITS.items():
        splits[split_type] = load_split(spark, split_name)
    
    return splits["treino"], splits["validacao"], splits["teste"]
