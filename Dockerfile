FROM python:3.10-slim-bookworm

ARG PRELOAD_EMBEDDING_MODEL=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/sentence-transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

COPY .streamlit/ ./.streamlit/
COPY app/ ./app/
COPY data/oscal_parsed/ ./data/oscal_parsed/
COPY data/policies_synth_md/ ./data/policies_synth_md/
COPY data/policies_synth_md_v2/ ./data/policies_synth_md_v2/
COPY scripts/ ./scripts/
COPY app.py ./
COPY .env.example ./

RUN mkdir -p /app/data/index /app/data/bm25_index /app/data/uploads_md /app/data/uploads_pdf /opt/huggingface /opt/sentence-transformers \
 && if [ "${PRELOAD_EMBEDDING_MODEL}" = "1" ]; then python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"; fi \
 && useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app /opt/huggingface /opt/sentence-transformers

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=5 CMD ["python", "scripts/container_healthcheck.py"]

CMD ["python", "scripts/start_container.py"]
