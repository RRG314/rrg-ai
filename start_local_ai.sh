#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt
mkdir -p .ai_data

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"
FRONTEND_HOST="127.0.0.1"
FRONTEND_PORT="5173"

echo "Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT}"
uvicorn backend.app:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload > .ai_data/backend.log 2>&1 &
BACKEND_PID=$!

sleep 1

echo "Starting frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT}"
python3 -m http.server "$FRONTEND_PORT" --bind "$FRONTEND_HOST" > .ai_data/frontend.log 2>&1 &
FRONTEND_PID=$!

cleanup() {
  echo "Stopping local AI services..."
  kill "$BACKEND_PID" >/dev/null 2>&1 || true
  kill "$FRONTEND_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if command -v open >/dev/null 2>&1; then
  open "http://${FRONTEND_HOST}:${FRONTEND_PORT}"
fi

echo "RRG AI is running."
echo "- UI:      http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "- Backend: http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "Press Ctrl+C to stop."

wait "$BACKEND_PID"
