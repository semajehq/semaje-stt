#!/usr/bin/env bash
# One-time setup for semaje-stt. Idempotent — safe to re-run.
#
#   ./setup.sh                          # Whisper + Omnilingual, pre-download default models
#   SEMAJE_OMNILINGUAL=0 ./setup.sh     # Whisper only (smaller, no torch)
#   BUILD_WHISPER_CPP=1 ./setup.sh      # also build whisper.cpp (Metal on Apple Silicon)
#
# The venv and models live in $SEMAJE_HOME (default ~/.local/share/semaje), not in
# the repo: cloud-synced folders (iCloud/OneDrive) evict files and break venvs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SEMAJE_HOME="${SEMAJE_HOME:-$HOME/.local/share/semaje}"
VENV="${STT_VENV:-$SEMAJE_HOME/venv-stt}"
export STT_MODEL_DIR="${STT_MODEL_DIR:-$SEMAJE_HOME/models}"
PYTHON="${PYTHON:-python3.11}"

echo "==> venv: $VENV"
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip uv
UV="$VENV/bin/uv pip install --python $VENV/bin/python --quiet"
$UV -r "$ROOT/requirements.txt" -r "$ROOT/requirements-dev.txt"

if [ "${SEMAJE_OMNILINGUAL:-1}" = "1" ]; then
  echo "==> Omnilingual ASR (1,600+ languages)"
  if ! python3 -c 'import ctypes.util, sys; sys.exit(0 if ctypes.util.find_library("sndfile") else 1)'; then
    echo "   libsndfile missing: install it first (brew install libsndfile | apt install libsndfile1)"; exit 1
  fi
  if [ "$(uname -s)" = "Linux" ]; then
    # fairseq2 publishes Linux wheels (x86-64) on Meta's index, per torch version.
    $UV --extra-index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cpu \
        -r "$ROOT/requirements-omnilingual.txt"
  else
    $UV -r "$ROOT/requirements-omnilingual.txt"
  fi
fi

echo "==> Pre-downloading models into $STT_MODEL_DIR (so no request waits on a download)"
cd "$ROOT"
"$VENV/bin/python" - <<'PY'
from semaje_stt import config
from semaje_stt.models import download_model, is_installed, omnilingual_available
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
  mkdir -p "$ROOT/vendor"
  [ -d "$ROOT/vendor/whisper.cpp" ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$ROOT/vendor/whisper.cpp"
  (cd "$ROOT/vendor/whisper.cpp" && cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --config Release)
fi

echo "Done. Start with: make run"
