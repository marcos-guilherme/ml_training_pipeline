import os
from flask import Flask, request, jsonify
from pipeline.training_pipeline import TrainingPipeline

app = Flask(__name__)

@app.route("/", methods=["POST"])
def entry_point():
    try:
        content = request.json
        params = content.get('params', None)
        
        print(f"Recebendo requisicao. Params: {params}")

        pipeline = TrainingPipeline()
        result = pipeline.run(params=params)
        
        status_code = 200 if result.get("status") == "success" else 400
        return jsonify(result), status_code

    except Exception as e:
        print(f"Erro no servidor: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200

if __name__ == "__main__":
    # Garante que usa a PORTA injetada pelo ambiente
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)