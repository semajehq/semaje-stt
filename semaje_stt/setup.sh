#!/usr/bin/env bash
# One-time setup for the STT service. Idempotent — safe to re-run.
#
#   services/stt/setup.sh                          # Whisper + Omnilingual, pre-download default models
#   SEMAJE_OMNILINGUAL=0 services/stt/setup.sh     # Whisper only (smaller, no torch)
#   BUILD_WHISPER_CPP=1 services/stt/setup.sh      # also build whisper.cpp (Metal on Apple Silicon)
#
# The venv and models live in $SEMAJE_HOME (default ~/.local/share/semaje), not in
# the repo: cloud-synced folders (iCloud/OneDrive) evict files and break venvs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SEMAJE_HOME="${SEMAJE_HOME:-$HOME/.local/share/semaje}"
VENV="${STT_VENV:-$SEMAJE_HOME/venv-stt}"
export STT_MODEL_DIR="${STT_MODEL_DIR:-$SEMAJE_HOME/models}"
PYTHON="${PYTHON:-python3.11}"

echo "==> venv: $VENV"
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip uv
UV="$VENV/bin/uv pip install --python $VENV/bin/python --quiet"
$UV -r "$HERE/requirements.txt" -r "$HERE/requirements-dev.txt"

if [ "${SEMAJE_OMNILINGUAL:-1}" = "1" ]; then
  echo "==> Omnilingual ASR (1,600+ languages)"
  if ! python3 - <<'PY' 2>/dev/null
import ctypes.util, sys; sys.exit(0 if ctypes.util.find_library("sndfile") else 1)
PY
  then
    echo "   libsndfile missing: install it first (brew install libsndfile | apt install libsndfile1)"; exit 1
  fi
  $UV -r "$HERE/requirements-omnilingual.txt"
fi

echo "==> Pre-downloading models into $STT_MODEL_DIR (so no request waits on a download)"
cd "$ROOT"
"$VENV/bin/python" - <<'PY'
from services.stt import config
from services.stt.models import download_model, is_installed, omnilingual_available
for engine, model in [("faster-whisper", config.MODEL)] + ([("omnilingual", config.OMNILINGUAL_MODEL)] if omnilingual_available() and config.OMNILINGUAL_ENABLED else []):
    if is_installed(engine, model):
        print(f"   {engine}/{model}: already installed")
    else:
        print(f"   {engine}/{model}: downloading…", flush=True)
        download_model(engine, model)
        print(f"   {engine}/{model}: ok")
PY

if [ "${BUILD_WHISPER_CPP:-0}" = "1" ]; then
  echo "==> Building whisper.cpp"
  mkdir -p "$HERE/vendor"
  [ -d "$HERE/vendor/whisper.cpp" ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$HERE/vendor/whisper.cpp"
  (cd "$HERE/vendor/whisper.cpp" && cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --config Release)
fi

echo "Done. Start with: npm run dev:stt   (or services/stt/run.sh)"
