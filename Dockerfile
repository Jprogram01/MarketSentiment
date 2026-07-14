# Slim image runs the API + LangGraph pipeline on the LLM/FinBERT-optional core.
# The default build does NOT install torch — the pipeline runs on the LLM backend
# or you layer the [finbert] extra in a derived image when you need local weights.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Runtime data dir (mount a volume over this in compose to persist the DB).
RUN mkdir -p /app/data
EXPOSE 8000

CMD ["uvicorn", "marketsentiment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
