FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    sumo sumo-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONPATH=/app/src

CMD ["bash", "scripts/train.sh", "configs/env_synthetic.yaml"]
