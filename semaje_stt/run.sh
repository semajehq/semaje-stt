#!/usr/bin/env bash
# Starts the Whisper STT service. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."   # project root, so `services.whisper.app` imports cleanly

VENV="services/whisper/.venv"
[ -d "$VENV" ] || { echo "Run services/whisper/setup.sh first."; exit 1; }

exec "$VENV/bin/python" -m services.whisper.app
