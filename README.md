# ML Training Pipeline - CNPJ Risk Prediction

Este repositório contém um microsserviço containerizado hospedado no Google Cloud Run, responsável pelo treinamento, avaliação e registro de modelos de Machine Learning para previsão de risco de empresas (CNPJ).


## Stack Tecnológico

* **Linguagem:** Python 3.11
* **Web Server:** Flask & Gunicorn
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Data Warehouse:** Google BigQuery
* **MLops:** MLflow
* **Infraestrutura:** Google Cloud Run, Cloud Build, Artifact Registry

## Estrutura do Projeto

```text
.
├── cloudbuild.yaml          # Configuração de CI/CD
├── Dockerfile               # Definição da imagem do container
├── requirements.txt         # Dependências do Python
├── main.py                  # Ponto de entrada da API (Flask)
├── src/
│   ├── config.py            # Configurações e variáveis de ambiente
│   ├── data_loader.py       # Extração de dados do BigQuery
│   ├── preprocessing.py     # Pipeline de pré-processamento (Scikit-Learn)
│   ├── model_training.py    # Lógica de treinamento e geração de gráficos
│   ├── model_evaluation.py  # Avaliação em dados de teste
│   ├── mlflow_registry.py   # Registro de modelos no MLflow
│   └── models/              # Implementação dos modelos (Strategy Pattern)
│       ├── base.py
│       ├── factory.py
│       ├── random_forest.py
│       └── logistic_regression.py
└── pipeline/
    └── training_pipeline.py # Orquestrador do fluxo de treino

```

## Documentação da API

O serviço expõe um endpoint HTTP para iniciar o pipeline de treinamento.

Pegue isso na nossa URL do serviço train_service, em Cloud Run -> Serviços.

**URL Base:** `https://train-service-SEU-ID.us-central1.run.app`

### POST `/`

Endpoint principal para execução do treinamento.

**Payload (JSON):**

| Parâmetro | Tipo | Obrigatório | Descrição |
| :--- | :--- | :--- | :--- |
| `experiment_name` | String | Não | Nome do experimento no MLflow. |
| `params` | Objeto | Sim | Dicionário contendo a configuração do algoritmo. |
| `params.model_type` | String | Não | Tipo do modelo: `"random_forest"` ou `"logistic_regression"`. |

**Exemplo de Requisição (Random Forest):**

```json
{
    "experiment_name": "Producao_V1",
    "params": {
        "model_type": "random_forest",
        "numTrees": 150,
        "maxDepth": 12,
        "minInstancesPerNode": 5
    }
}

