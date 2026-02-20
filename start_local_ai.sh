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

PAIRING_CODE_FILE=".ai_data/pairing_code.txt"
if [[ -n "${AI_BOOTSTRAP_PAIRING_CODE:-}" ]]; then
  PAIRING_CODE="${AI_BOOTSTRAP_PAIRING_CODE}"
elif [[ -s "${PAIRING_CODE_FILE}" ]]; then
  PAIRING_CODE="$(cat "${PAIRING_CODE_FILE}")"
else
  PAIRING_CODE="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(10))
PY
)"
  printf "%s" "${PAIRING_CODE}" > "${PAIRING_CODE_FILE}"
fi
export AI_BOOTSTRAP_PAIRING_CODE="${PAIRING_CODE}"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="${AI_PORT:-8000}"
OLLAMA_HOST="127.0.0.1"
OLLAMA_PORT="11434"
AI_MODEL="${AI_MODEL:-llama3.2:3b}"
export AI_MODEL
AI_RECURSIVE_ADIC_RANKING="${AI_RECURSIVE_ADIC_RANKING:-1}"
AI_RADF_BETA="${AI_RADF_BETA:-0.35}"
AI_RADF_ALPHA="${AI_RADF_ALPHA:-1.5}"
AI_REQUIRE_TOKEN="${AI_REQUIRE_TOKEN:-1}"
AI_REPO_COLLECTION_ROOT="${AI_REPO_COLLECTION_ROOT:-$(cd "${ROOT_DIR}/.." && pwd)}"

if [[ -z "${AI_LEARNING_PDF_PATHS:-}" ]]; then
  PDF_A="$HOME/Downloads/The_Recursive_Adic_Number_Field__Construction__Analysis__and_Recursive_Depth_Transforms (1).pdf"
  PDF_B="$HOME/Downloads/v.pdf"
  PDF_LIST=()
  if [[ -f "${PDF_A}" ]]; then
    PDF_LIST+=("${PDF_A}")
  fi
  if [[ -f "${PDF_B}" ]]; then
    PDF_LIST+=("${PDF_B}")
  fi
  if [[ "${#PDF_LIST[@]}" -gt 0 ]]; then
    AI_LEARNING_PDF_PATHS="$(IFS=:; echo "${PDF_LIST[*]}")"
  fi
fi

export AI_RECURSIVE_ADIC_RANKING AI_RADF_BETA AI_RADF_ALPHA AI_REQUIRE_TOKEN AI_REPO_COLLECTION_ROOT
if [[ -n "${AI_LEARNING_PDF_PATHS:-}" ]]; then
  export AI_LEARNING_PDF_PATHS
fi

OLLAMA_PID=""

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

ollama_reachable() {
  curl -sS -m 2 "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1
}

if command -v ollama >/dev/null 2>&1; then
  if ! ollama_reachable; then
    echo "Starting Ollama on http://${OLLAMA_HOST}:${OLLAMA_PORT}"
    ollama serve > .ai_data/ollama.log 2>&1 &
    OLLAMA_PID=$!
    for _ in {1..20}; do
      if ollama_reachable; then
        break
      fi
      sleep 0.5
    done
  fi

  if ollama_reachable; then
    if ! ollama list | awk 'NR>1 {print $1}' | rg -Fx "${AI_MODEL}" >/dev/null 2>&1; then
      echo "Pulling model ${AI_MODEL} (one-time)..."
      if ! ollama pull "${AI_MODEL}"; then
        echo "Warning: could not pull model ${AI_MODEL}. Backend will run in tool/rules mode." >&2
      fi
    fi
  else
    echo "Warning: Ollama is not reachable; backend will run in tool/rules mode." >&2
  fi
else
  echo "Ollama not found. Install with: brew install ollama" >&2
  echo "Backend will run in tool/rules mode until Ollama is installed." >&2
fi

if ! command -v tesseract >/dev/null 2>&1; then
  echo "Note: tesseract is not installed. Image OCR upload/analyze will be unavailable." >&2
  echo "Install with: brew install tesseract" >&2
fi

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
  if [[ -n "${OLLAMA_PID}" ]]; then
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

UI_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"

if command -v open >/dev/null 2>&1; then
  open "$UI_URL"
fi

echo "RRG AI is running."
echo "- UI + API: ${UI_URL}"
echo "- Model:    ${AI_MODEL}"
echo "- RADF:     enabled=${AI_RECURSIVE_ADIC_RANKING} alpha=${AI_RADF_ALPHA} beta=${AI_RADF_BETA}"
echo "- Sources:  repo-root=${AI_REPO_COLLECTION_ROOT}"
if [[ -n "${AI_LEARNING_PDF_PATHS:-}" ]]; then
  echo "- PDFs:     ${AI_LEARNING_PDF_PATHS}"
fi
echo "- Security: token auth=${AI_REQUIRE_TOKEN} (auto-managed local token)"
echo "- Pairing:  ${PAIRING_CODE} (needed only when connecting from external origin like GitHub Pages)"
echo "Press Ctrl+C to stop."

wait "$BACKEND_PID"
