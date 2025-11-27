from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.models.base import BaseModelStrategy

class LogisticRegressionStrategy(BaseModelStrategy):
    def __init__(self, params):
        self.params = params
        self.C = params.get("C", 1.0)

    def build_pipeline(self, preprocessor):
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(C=self.C, random_state=42))
        ])
        
    def get_params(self):
        return self.params