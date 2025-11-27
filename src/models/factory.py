from src.models.random_forest import RandomForestStrategy
from src.models.logistic_regression import LogisticRegressionStrategy

class ModelFactory:
    @staticmethod
    def get_strategy(model_type, params):
        if model_type == "random_forest":
            return RandomForestStrategy(params)
        elif model_type == "logistic_regression":
            return LogisticRegressionStrategy(params)
        else:
            raise ValueError(f"Modelo desconhecido: {model_type}")