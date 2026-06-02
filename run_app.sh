#!/usr/bin/env bash
# Run the Streamlit dashboard with the project virtual environment.
set -e
cd "$(dirname "$0")"

# Prefer venv/ if present, else .venv/
if [ -d "venv" ]; then
  VENV="venv"
elif [ -d ".venv" ]; then
  VENV=".venv"
else
  python3 -m venv venv
  VENV="venv"
fi

"$VENV/bin/pip" install -q -r requirements.txt
exec "$VENV/bin/streamlit" run app.py --server.address=0.0.0.0 "$@"
