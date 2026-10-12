#!/usr/bin/env bash
# Start the STT service from anywhere. Uses the venv created by setup.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
SEMAJE_HOME="${SEMAJE_HOME:-$HOME/.local/share/semaje}"
VENV="${STT_VENV:-$SEMAJE_HOME/venv-stt}"
[ -x "$VENV/bin/python" ] || { echo "Run services/stt/setup.sh first."; exit 1; }
export STT_MODEL_DIR="${STT_MODEL_DIR:-$SEMAJE_HOME/models}"
exec "$VENV/bin/python" -m services.stt.app
