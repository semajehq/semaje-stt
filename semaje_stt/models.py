"""Local model registry, hardware hints, and persistent cache management."""
from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
from urllib.request import urlretrieve

from . import config

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_ROOT = Path(os.getenv("WHISPER_MODEL_DIR", os.getenv("HF_HOME", str(DEFAULT_MODEL_ROOT))))
ACTIVE_FILE = MODEL_ROOT / ".semaje-active.json"

MODEL_REGISTRY = [
    {"id": "tiny", "label": "Tiny", "diskMb": 75, "ramMb": 1000, "vramMb": 1000,
     "speed": "fastest", "accuracy": "basic", "languages": "multilingual"},
    {"id": "base", "label": "Base", "diskMb": 150, "ramMb": 1500, "vramMb": 1500,
     "speed": "very fast", "accuracy": "good", "languages": "multilingual"},
    {"id": "small", "label": "Small", "diskMb": 500, "ramMb": 2500, "vramMb": 2500,
     "speed": "fast", "accuracy": "better", "languages": "multilingual"},
    {"id": "medium", "label": "Medium", "diskMb": 1500, "ramMb": 5000, "vramMb": 5000,
     "speed": "balanced", "accuracy": "high", "languages": "multilingual"},
    {"id": "large-v3", "label": "Large v3", "diskMb": 3100, "ramMb": 10000, "vramMb": 10000,
     "speed": "slow", "accuracy": "highest", "languages": "multilingual"},
]


def _memory_mb() -> int:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(page_size * pages / 1024 / 1024)
    except (ValueError, OSError, AttributeError):
        return 0


def hardware_info() -> dict:
    cuda_visible = bool(os.getenv("NVIDIA_VISIBLE_DEVICES") not in (None, "", "void"))
    device = "cuda" if config.DEVICE == "cuda" or cuda_visible else "cpu"
    ram_mb = _memory_mb()
    if device == "cuda":
        recommended = "medium"
    elif ram_mb >= 12000:
        recommended = "small"
    elif ram_mb >= 6000:
        recommended = "base"
    else:
        recommended = "tiny"
    return {
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "cpuCount": os.cpu_count() or 1,
        "ramMb": ram_mb,
        "device": device,
        "computeType": config.COMPUTE_TYPE,
        "recommendedModel": recommended,
    }


def active_config() -> dict:
    default = {"backend": config.BACKEND, "model": config.MODEL}
    try:
        stored = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
        if stored.get("backend") in ("faster-whisper", "whisper.cpp") and model_exists(stored.get("model")):
            return {"backend": stored["backend"], "model": stored["model"]}
    except (OSError, ValueError, TypeError):
        pass
    return default


def save_active(backend: str, model: str) -> None:
    if backend not in ("faster-whisper", "whisper.cpp") or not model_exists(model):
        raise ValueError("unknown backend or model")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_FILE.write_text(json.dumps({"backend": backend, "model": model}), encoding="utf-8")


def model_exists(model: str | None) -> bool:
    return any(item["id"] == model for item in MODEL_REGISTRY)


def model_path(backend: str, model: str) -> Path:
    if backend == "faster-whisper":
        return MODEL_ROOT / "faster-whisper" / model
    return MODEL_ROOT / "whisper.cpp" / f"ggml-{model}.bin"


def is_installed(backend: str, model: str) -> bool:
    path = model_path(backend, model)
    return (path / "config.json").exists() if backend == "faster-whisper" else path.exists()


def cache_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def list_models() -> list[dict]:
    rows = []
    active = active_config()
    for backend in ("faster-whisper", "whisper.cpp"):
        for item in MODEL_REGISTRY:
            path = model_path(backend, item["id"])
            rows.append({
                **item,
                "backend": backend,
                "installed": is_installed(backend, item["id"]),
                "cachedBytes": cache_size(path),
                "active": active == {"backend": backend, "model": item["id"]},
            })
    return rows


def download_model(backend: str, model: str) -> dict:
    if not model_exists(model):
        raise ValueError(f"unknown model: {model}")
    target = model_path(backend, model)
    target.parent.mkdir(parents=True, exist_ok=True)
    if backend == "faster-whisper":
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=f"Systran/faster-whisper-{model}",
            local_dir=str(target),
        )
    elif backend == "whisper.cpp":
        urlretrieve(
            f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin",
            target,
        )
    else:
        raise ValueError(f"unknown backend: {backend}")
    return {"backend": backend, "model": model, "cachedBytes": cache_size(target)}


def delete_model(backend: str, model: str) -> None:
    if active_config() == {"backend": backend, "model": model}:
        raise ValueError("activate another model before deleting this one")
    target = model_path(backend, model)
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
