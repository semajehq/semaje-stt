"""Local model registry, hardware hints, and persistent cache management.

Every model listed here is open source with its license recorded, so an
operator can see exactly what runs on their hardware and under which terms.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
from urllib.request import urlretrieve

from . import config

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_ROOT = Path(os.getenv("STT_MODEL_DIR", os.getenv("WHISPER_MODEL_DIR", os.getenv("HF_HOME", str(DEFAULT_MODEL_ROOT)))))
ACTIVE_FILE = MODEL_ROOT / ".semaje-active.json"

WHISPER_BACKENDS = ("faster-whisper", "whisper.cpp")
ENGINES = (*WHISPER_BACKENDS, "omnilingual")

WHISPER_LICENSE = {"spdx": "MIT", "url": "https://github.com/openai/whisper/blob/main/LICENSE"}
OMNILINGUAL_LICENSE = {"spdx": "Apache-2.0", "url": "https://github.com/facebookresearch/omnilingual-asr/blob/main/LICENSE"}

WHISPER_MODELS = [
    {"id": "tiny", "label": "Tiny", "diskMb": 75, "ramMb": 1000, "vramMb": 1000,
     "speed": "fastest", "accuracy": "basic"},
    {"id": "base", "label": "Base", "diskMb": 150, "ramMb": 1500, "vramMb": 1500,
     "speed": "very fast", "accuracy": "good"},
    {"id": "small", "label": "Small", "diskMb": 500, "ramMb": 2500, "vramMb": 2500,
     "speed": "fast", "accuracy": "better"},
    {"id": "medium", "label": "Medium", "diskMb": 1500, "ramMb": 5000, "vramMb": 5000,
     "speed": "balanced", "accuracy": "high"},
    {"id": "large-v3-turbo", "label": "Large v3 Turbo", "diskMb": 1600, "ramMb": 6000, "vramMb": 6000,
     "speed": "fast", "accuracy": "very high"},
    {"id": "large-v3", "label": "Large v3", "diskMb": 3100, "ramMb": 10000, "vramMb": 10000,
     "speed": "slow", "accuracy": "highest"},
]

# Hugging Face repos holding CTranslate2 conversions of each Whisper size.
FASTER_WHISPER_REPOS = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# Omnilingual ASR model cards. CTC models are fastest and need no language hint;
# LLM models accept an optional language code and are more accurate.
OMNILINGUAL_MODELS = [
    {"id": "omniASR_CTC_300M_v2", "label": "Omnilingual CTC 300M", "diskMb": 1300, "ramMb": 3000, "vramMb": 2000,
     "speed": "fast", "accuracy": "good"},
    {"id": "omniASR_CTC_1B_v2", "label": "Omnilingual CTC 1B", "diskMb": 4000, "ramMb": 7000, "vramMb": 4000,
     "speed": "balanced", "accuracy": "better"},
    {"id": "omniASR_LLM_300M_v2", "label": "Omnilingual LLM 300M", "diskMb": 1800, "ramMb": 4000, "vramMb": 3000,
     "speed": "balanced", "accuracy": "better"},
    {"id": "omniASR_LLM_1B_v2", "label": "Omnilingual LLM 1B", "diskMb": 5000, "ramMb": 9000, "vramMb": 6000,
     "speed": "slow", "accuracy": "high"},
    {"id": "omniASR_LLM_3B_v2", "label": "Omnilingual LLM 3B", "diskMb": 12000, "ramMb": 20000, "vramMb": 10000,
     "speed": "slow", "accuracy": "very high"},
    {"id": "omniASR_LLM_7B_v2", "label": "Omnilingual LLM 7B", "diskMb": 30000, "ramMb": 40000, "vramMb": 17000,
     "speed": "slowest", "accuracy": "highest"},
]

# Kept for callers that only know about Whisper sizes.
MODEL_REGISTRY = WHISPER_MODELS


def _memory_mb() -> int:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 / 1024)
    except (ValueError, OSError, AttributeError):
        return 0


def hardware_info() -> dict:
    cuda_visible = os.getenv("NVIDIA_VISIBLE_DEVICES") not in (None, "", "void")
    device = "cuda" if config.DEVICE == "cuda" or cuda_visible else "cpu"
    ram_mb = _memory_mb()
    if device == "cuda":
        recommended, omni = "large-v3-turbo", "omniASR_LLM_1B_v2"
    elif ram_mb >= 12000:
        recommended, omni = "small", "omniASR_CTC_1B_v2"
    elif ram_mb >= 6000:
        recommended, omni = "base", "omniASR_CTC_300M_v2"
    else:
        recommended, omni = "tiny", "omniASR_CTC_300M_v2"
    return {
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "cpuCount": os.cpu_count() or 1,
        "ramMb": ram_mb,
        "device": device,
        "computeType": config.COMPUTE_TYPE,
        "recommendedModel": recommended,
        "recommendedOmnilingualModel": omni,
    }


def model_exists(model: str | None, engine: str = "faster-whisper") -> bool:
    registry = OMNILINGUAL_MODELS if engine == "omnilingual" else WHISPER_MODELS
    return any(item["id"] == model for item in registry)


def _default_active() -> dict:
    return {
        "whisper": {"backend": config.BACKEND, "model": config.MODEL},
        "omnilingual": {"model": config.OMNILINGUAL_MODEL},
    }


def active_config() -> dict:
    """{"whisper": {"backend", "model"}, "omnilingual": {"model"}} — persisted choice or env defaults."""
    active = _default_active()
    try:
        stored = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return active
    if not isinstance(stored, dict):
        return active
    # v1 file (Whisper-only service) stored {"backend", "model"} at the top level.
    whisper = stored.get("whisper", stored)
    if isinstance(whisper, dict) and whisper.get("backend") in WHISPER_BACKENDS \
            and model_exists(whisper.get("model")):
        active["whisper"] = {"backend": whisper["backend"], "model": whisper["model"]}
    omni = stored.get("omnilingual")
    if isinstance(omni, dict) and model_exists(omni.get("model"), "omnilingual"):
        active["omnilingual"] = {"model": omni["model"]}
    return active


def save_active(engine: str, model: str) -> dict:
    if engine not in ENGINES or not model_exists(model, engine):
        raise ValueError("unknown engine or model")
    active = active_config()
    if engine == "omnilingual":
        active["omnilingual"] = {"model": model}
    else:
        active["whisper"] = {"backend": engine, "model": model}
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_FILE.write_text(json.dumps(active), encoding="utf-8")
    return active


def model_path(engine: str, model: str) -> Path:
    if engine == "faster-whisper":
        return MODEL_ROOT / "faster-whisper" / model
    if engine == "omnilingual":
        return MODEL_ROOT / "omnilingual"
    return MODEL_ROOT / "whisper.cpp" / f"ggml-{model}.bin"


def omnilingual_marker(model: str) -> Path:
    return MODEL_ROOT / "omnilingual" / f".{model}.ready"


def is_installed(engine: str, model: str) -> bool:
    if engine == "omnilingual":
        return omnilingual_marker(model).exists()
    path = model_path(engine, model)
    return (path / "config.json").exists() if engine == "faster-whisper" else path.exists()


def cache_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def omnilingual_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("omnilingual_asr") is not None
    except (ImportError, ValueError):
        return False


def list_models() -> list[dict]:
    rows = []
    active = active_config()
    for backend in WHISPER_BACKENDS:
        for item in WHISPER_MODELS:
            rows.append({
                **item,
                "backend": backend,
                "engine": backend,
                "family": "whisper",
                "languages": 99,
                "license": WHISPER_LICENSE,
                "installed": is_installed(backend, item["id"]),
                "cachedBytes": cache_size(model_path(backend, item["id"])),
                "active": active["whisper"] == {"backend": backend, "model": item["id"]},
                "available": True,
            })
    omni_ok = omnilingual_available()
    for item in OMNILINGUAL_MODELS:
        rows.append({
            **item,
            "backend": "omnilingual",
            "engine": "omnilingual",
            "family": "omnilingual",
            "languages": 1600,
            "license": OMNILINGUAL_LICENSE,
            "installed": is_installed("omnilingual", item["id"]),
            "cachedBytes": 0,
            "active": active["omnilingual"]["model"] == item["id"],
            "available": omni_ok,
        })
    return rows


def download_model(engine: str, model: str) -> dict:
    if not model_exists(model, engine):
        raise ValueError(f"unknown model: {model}")
    target = model_path(engine, model)
    target.parent.mkdir(parents=True, exist_ok=True)
    if engine == "faster-whisper":
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=FASTER_WHISPER_REPOS.get(model, f"Systran/faster-whisper-{model}"),
            local_dir=str(target),
        )
    elif engine == "whisper.cpp":
        urlretrieve(
            f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin",
            target,
        )
    elif engine == "omnilingual":
        # fairseq2 resolves + caches checkpoints on first load; loading is the download.
        from .engines.omnilingual_engine import OmnilingualEngine
        OmnilingualEngine(model).load()
    else:
        raise ValueError(f"unknown engine: {engine}")
    return {"backend": engine, "model": model, "cachedBytes": cache_size(target)}


def delete_model(engine: str, model: str) -> None:
    active = active_config()
    if engine == "omnilingual":
        if active["omnilingual"]["model"] == model:
            raise ValueError("activate another model before deleting this one")
        omnilingual_marker(model).unlink(missing_ok=True)
        return
    if active["whisper"] == {"backend": engine, "model": model}:
        raise ValueError("activate another model before deleting this one")
    target = model_path(engine, model)
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
