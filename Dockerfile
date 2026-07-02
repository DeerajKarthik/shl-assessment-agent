FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY app/ app/
COPY data/ data/
COPY evaluation/ evaluation/
COPY pyproject.toml .

# Pre-compile Python files for faster startup
RUN python -m compileall -q app/ evaluation/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE ${PORT:-8080}

ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os; import urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8080\")}/health')" || exit 1

# Use $PORT environment variable so Railway/Docker can override it
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 30"]
