import functions_framework
from flask import jsonify
from pipeline.training_pipeline import TrainingPipeline
import os

@functions_framework.http
def train_model(request):
    """
    Função de entrada do Cloud Functions.
    Nome deve bater com --entry-point no cloudbuild.yaml
    """
    
    # 1. Parsing de argumentos opcionais (via JSON no POST)
    # Ex: {"numTrees": 50, "maxDepth": 10}
    request_json = request.get_json(silent=True)
    params = None
    
    if request_json and 'params' in request_json:
        params = request_json['params']
    
    print(f"Recebida requisicao de treino. Params: {params}")

    try:
        pipeline = TrainingPipeline()
    except Exception as e:
        print(f"Erro fatal ao iniciar Spark: {e}")
        return jsonify({"status": "error", "message": "Falha ao iniciar Spark/Pipeline", "details": str(e)}), 500

    try:
        # Roda o treino
        result = pipeline.run(params=params)
        
        status_code = 200 if result.get("status") == "success" else 400
        return jsonify(result), status_code

    except Exception as e:
        print(f"Erro durante execucao do pipeline: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
        
    finally:

        if pipeline:
            pipeline.close()