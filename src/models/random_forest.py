from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.models.base import BaseModelStrategy

class RandomForestStrategy(BaseModelStrategy):
    def __init__(self, params):
        self.params = params
        self.num_trees = params.get("numTrees", 100)
        self.max_depth = params.get("maxDepth", 10)
        self.min_samples = params.get("minInstancesPerNode", 5)

    def build_pipeline(self, preprocessor):
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=self.num_trees,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples,
                random_state=42,
                n_jobs=-1
            ))
        ])
    
    def get_params(self):
        return self.params