FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY app/ app/
COPY data/ data/
COPY static/ static/
COPY evaluation/ evaluation/
COPY pyproject.toml .
COPY run.py .

# Pre-compile Python files for faster startup
RUN python -m compileall -q app/ evaluation/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Pre-download the heavy Nomic embedding model into the container cache 
# so it doesn't download at runtime (which causes OOM or timeouts)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"

EXPOSE ${PORT:-8080}

ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os; import urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8080\")}/health')" || exit 1

# Use python entrypoint to entirely bypass shell escaping issues with $PORT
CMD ["python", "run.py"]
