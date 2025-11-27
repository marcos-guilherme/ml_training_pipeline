FROM python:3.11-slim

RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && \
    apt-get install -y openjdk-21-jre-headless procps curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

WORKDIR /app


RUN mkdir -p /app/jars && \
    curl -o /app/jars/spark-bigquery-with-dependencies.jar \
    https://storage.googleapis.com/hadoop-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.34.0.jar

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec functions-framework --target=train_model