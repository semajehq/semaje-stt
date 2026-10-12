"""STT service configuration — every knob comes from an env var.

Single source of truth so the API layer, the router, and every engine agree.
Legacy WHISPER_* names are still honoured so existing deployments keep working.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _env(name: str, default: str, *legacy: str) -> str:
    for key in (name, *legacy):
        value = os.getenv(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _int(name: str, default: int, *legacy: str) -> int:
    return int(_env(name, str(default), *legacy))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in ("1", "true", "yes", "on")


# ── Whisper engine ───────────────────────────────────────────────
# Which Whisper runtime: "faster-whisper" (default) or "whisper.cpp".
BACKEND = _env("STT_WHISPER_BACKEND", "faster-whisper", "WHISPER_BACKEND")
# tiny | base | small | medium | large-v3 | large-v3-turbo
MODEL = _env("STT_WHISPER_MODEL", "base", "WHISPER_MODEL")
DEVICE = _env("STT_DEVICE", "cpu", "WHISPER_DEVICE")                    # cpu | cuda | auto
COMPUTE_TYPE = _env("STT_COMPUTE_TYPE", "int8", "WHISPER_COMPUTE_TYPE")  # int8 | float16 | float32

WHISPER_CPP_BIN = _env(
    "WHISPER_CPP_BIN", str(HERE / "vendor" / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
)
WHISPER_CPP_MODEL = _env(
    "WHISPER_CPP_MODEL", str(HERE / "vendor" / "whisper.cpp" / "models" / f"ggml-{MODEL}.bin"),
)
WHISPER_CPP_THREADS = _env("WHISPER_CPP_THREADS", "")

# ── Omnilingual engine (Meta Omnilingual ASR, Apache-2.0, 1600+ languages) ──
# Disabled automatically when the `omnilingual-asr` package is not installed.
OMNILINGUAL_ENABLED = _bool("STT_OMNILINGUAL_ENABLED", True)
OMNILINGUAL_MODEL = _env("STT_OMNILINGUAL_MODEL", "omniASR_CTC_300M_v2")
OMNILINGUAL_DEVICE = _env("STT_OMNILINGUAL_DEVICE", "auto")  # auto | cpu | cuda | mps
OMNILINGUAL_BATCH = _int("STT_OMNILINGUAL_BATCH", 4)
# CTC/LLM model cards reject inputs >= 40s; chunk well below that.
OMNILINGUAL_MAX_CHUNK_SEC = float(_env("STT_OMNILINGUAL_MAX_CHUNK_SEC", "30"))

# ── Routing ──────────────────────────────────────────────────────
# auto        → strong Whisper languages go to Whisper, everything else to Omnilingual
# whisper     → always Whisper
# omnilingual → always Omnilingual (falls back to Whisper if not installed)
ROUTING = _env("STT_ROUTING", "auto")

# ── Audio intake limits ──────────────────────────────────────────
MAX_UPLOAD_MB = _int("STT_MAX_UPLOAD_MB", 2048)
MAX_DURATION_SEC = _int("STT_MAX_DURATION_SEC", 6 * 3600)
FFMPEG_BIN = _env("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = _env("FFPROBE_BIN", "ffprobe")
FFMPEG_TIMEOUT_SEC = _int("STT_FFMPEG_TIMEOUT_SEC", 1800)

# ── Service hardening ────────────────────────────────────────────
# Max transcriptions running at once; extra requests wait up to QUEUE_TIMEOUT_SEC
# and then get 503 + Retry-After instead of piling up and exhausting RAM.
MAX_CONCURRENCY = _int("STT_MAX_CONCURRENCY", 1)
QUEUE_TIMEOUT_SEC = float(_env("STT_QUEUE_TIMEOUT_SEC", "300"))
# Optional shared secret. When set, every route except liveness requires
# `Authorization: Bearer <token>`.
API_TOKEN = _env("STT_API_TOKEN", "")
LOG_LEVEL = _env("STT_LOG_LEVEL", "INFO")

HOST = _env("STT_HOST", "0.0.0.0", "WHISPER_HOST")
PORT = _int("STT_PORT", 8011, "WHISPER_PORT")
