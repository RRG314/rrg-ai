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
BACKEND_PORT="${AI_PORT:-8000}"

is_port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1
}

pick_backend_port() {
  local base="$1"
  local port="$base"
  local tries=0
  while is_port_in_use "$port"; do
    tries=$((tries + 1))
    if [[ "$tries" -ge 25 ]]; then
      echo "Unable to find a free port near ${base}" >&2
      exit 1
    fi
    port=$((port + 1))
  done
  echo "$port"
}

BACKEND_PORT="$(pick_backend_port "$BACKEND_PORT")"

echo "Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT}"
uvicorn backend.app:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload > .ai_data/backend.log 2>&1 &
BACKEND_PID=$!

sleep 1
if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
  echo "Backend failed to start. Tail of log:" >&2
  tail -n 80 .ai_data/backend.log >&2 || true
  exit 1
fi

cleanup() {
  echo "Stopping local AI backend..."
  kill "$BACKEND_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

UI_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"

if command -v open >/dev/null 2>&1; then
  open "$UI_URL"
fi

echo "RRG AI is running."
echo "- UI + API: ${UI_URL}"
echo "Press Ctrl+C to stop."

wait "$BACKEND_PID"
