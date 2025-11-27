FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instala dependencias do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Gunicorn deve ouvir na variavel PORT
# Timeout alto (3600s) para garantir o treino
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 3600 main:app