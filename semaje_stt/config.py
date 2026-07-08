"""Whisper service configuration — all knobs come from env vars.

Single source of truth so the API layer and both backends agree on settings.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Which backend to run: "faster-whisper" (default) or "whisper.cpp"
BACKEND = os.getenv("WHISPER_BACKEND", "faster-whisper").strip()

# Model name for faster-whisper: tiny | base | small | medium | large-v3
# Default "base": ~150MB, fast on CPU, solid accuracy. Set large-v3 for best
# accuracy (slower, ~1.5GB), or tiny for fastest/lowest-accuracy.
MODEL = os.getenv("WHISPER_MODEL", "base").strip()

# faster-whisper runtime
DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip()          # cpu | cuda | auto
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip()  # int8 | float16 | float32

# whisper.cpp paths. Defaults point at the vendored build from setup.sh.
WHISPER_CPP_BIN = os.getenv(
    "WHISPER_CPP_BIN",
    str(HERE / "vendor" / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
)
WHISPER_CPP_MODEL = os.getenv(
    "WHISPER_CPP_MODEL",
    str(HERE / "vendor" / "whisper.cpp" / "models" / f"ggml-{MODEL}.bin"),
)

# Number of CPU threads whisper.cpp uses (0 = library default)
WHISPER_CPP_THREADS = os.getenv("WHISPER_CPP_THREADS", "").strip()

HOST = os.getenv("WHISPER_HOST", "0.0.0.0")
PORT = int(os.getenv("WHISPER_PORT", "8011"))
