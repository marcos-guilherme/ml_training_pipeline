import functions_framework
from flask import jsonify
# Importar a classe não roda código pesado, então pode ficar aqui fora
from pipeline.training_pipeline import TrainingPipeline

@functions_framework.http
def train_model(request):
    """Função principal"""
    request_json = request.get_json(silent=True)
    params = None
    
    if request_json and 'params' in request_json:
        params = request_json['params']
    
    print(f"Iniciando requisicao. Params: {params}")

    try:
        # Instancia e roda AQUI DENTRO
        pipeline = TrainingPipeline()
        result = pipeline.run(params=params)
        
        return jsonify(result), 200 if result.get("status") == "success" else 400

    except Exception as e:
        # Importante: Printar o erro para aparecer no Log do GCP
        print(f"ERRO FATAL NA FUNÇÃO: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500