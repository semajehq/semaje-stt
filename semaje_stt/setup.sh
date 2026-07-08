#!/usr/bin/env bash
# Sets up the Whisper STT service: Python venv + faster-whisper, and (optionally)
# builds the whisper.cpp backend. Idempotent — safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.11}"
MODEL="${WHISPER_MODEL:-base}"

echo "==> Python venv + deps"
[ -d .venv ] || "$PYTHON" -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt

echo "==> Warming faster-whisper model: $MODEL"
./.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('$MODEL', device='cpu', compute_type='int8'); print('   ok')"

if [ "${BUILD_WHISPER_CPP:-0}" = "1" ]; then
  echo "==> Building whisper.cpp backend"
  mkdir -p vendor
  [ -d vendor/whisper.cpp ] || git clone --depth 1 https://github.com/ggerganov/whisper.cpp vendor/whisper.cpp
  pushd vendor/whisper.cpp >/dev/null
  cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j --config Release
  bash ./models/download-ggml-model.sh "$MODEL"
  popd >/dev/null
  echo "   whisper.cpp ready"
else
  echo "==> Skipping whisper.cpp (set BUILD_WHISPER_CPP=1 to build it)"
fi

echo "Done. Start with: ./run.sh"
