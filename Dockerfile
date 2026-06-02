# Credit Risk Platform — Streamlit dashboard + FastAPI backend (Python 3.11)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_PORT=8501 \
    API_PORT=8000

WORKDIR /app

# System libs for XGBoost / scientific stack wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app.py run_app.sh run_api.sh ./
COPY scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh
COPY src ./src
COPY sql ./sql
COPY .streamlit ./.streamlit

RUN mkdir -p data models documents \
    && chmod +x /app/scripts/docker-entrypoint.sh

# Non-root runtime user
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${API_PORT}/docs" > /dev/null \
    || curl -fsS "http://127.0.0.1:${STREAMLIT_PORT}/_stcore/health" > /dev/null

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["both"]
