FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    HF_HUB_DISABLE_XET=1 \
    TORCH_NUM_THREADS=4 \
    DATABASE_URL=postgresql+psycopg://invest:invest@db:5432/invest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY web ./web
COPY scripts ./scripts
COPY config ./config
COPY db ./db

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e . \
 && chmod +x scripts/docker_entrypoint.sh \
 && mkdir -p /cache/hf /app/data /app/models /app/reports

EXPOSE 8000
ENTRYPOINT ["./scripts/docker_entrypoint.sh"]
