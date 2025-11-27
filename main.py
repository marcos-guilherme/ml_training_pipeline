import sys
import json
from pipeline.training_pipeline import TrainingPipeline

def main(custom_params=None):
    """Executar pipeline de treinamento"""
    
    pipeline = TrainingPipeline()
    
    try:
        result = pipeline.run(params=custom_params)
        print(json.dumps(result, indent=2))
        return result
    
    finally:
        pipeline.close()

def parse_args():
    """Parser robusto de argumentos"""
    custom_params = None
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        try:
            # Tentar parsear como JSON
            custom_params = json.loads(arg)
        except json.JSONDecodeError:
            print(f"Erro ao parsear JSON: {arg}")
            print("Use: python main.py ou python main.py '{\"numTrees\": 150}'")
            sys.exit(1)
    
    return custom_params

if __name__ == "__main__":
    custom_params = parse_args()
    result = main(custom_params=custom_params)
    sys.exit(0 if result["status"] == "success" else 1)
