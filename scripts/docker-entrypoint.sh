#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-both}"
shift || true

run_api() {
  exec uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port "${API_PORT:-8000}" \
    "$@"
}

run_streamlit() {
  exec streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port="${STREAMLIT_PORT:-8501}" \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    "$@"
}

case "${MODE}" in
  api)
    run_api "$@"
    ;;
  streamlit)
    run_streamlit "$@"
    ;;
  both)
    run_api &
    API_PID=$!
    trap 'kill "${API_PID}" 2>/dev/null || true' EXIT INT TERM
    run_streamlit "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
