"""Whisper STT HTTP service. See contract.md for the API."""
import os
import tempfile
from threading import Lock

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import config
from .backends import get_backend
from .models import (
    active_config, delete_model, download_model, hardware_info, list_models,
    model_exists, save_active,
)

app = FastAPI(title="semaje Whisper STT", version="1.0.0")

# One backend instance per process. Model loads lazily and can be swapped by an
# authenticated platform admin through the API gateway.
_active = active_config()
_backend = get_backend(_active["backend"], _active["model"])
_backend_lock = Lock()


class ActivateModelRequest(BaseModel):
    backend: str
    model: str


@app.get("/health")
def health():
    return {
        "ok": True,
        "ready": _backend.ready,
        "backend": _backend.name,
        "model": _backend.model,
        "device": config.DEVICE if _backend.name == "faster-whisper" else "metal/cpu",
        "computeType": config.COMPUTE_TYPE,
    }


@app.get("/models")
def models():
    return {
        "models": list_models(),
        "active": {"backend": _backend.name, "model": _backend.model},
        "hardware": hardware_info(),
    }


@app.post("/models/download")
def download(req: ActivateModelRequest):
    try:
        return {"ok": True, **download_model(req.backend, req.model)}
    except (ValueError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.delete("/models/{backend}/{model}")
def remove(backend: str, model: str):
    try:
        delete_model(backend, model)
        return {"ok": True}
    except (ValueError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/models/activate")
def activate(req: ActivateModelRequest):
    global _backend
    if req.backend not in ("faster-whisper", "whisper.cpp") or not model_exists(req.model):
        return JSONResponse({"error": "unknown backend or model"}, status_code=400)
    try:
        replacement = get_backend(req.backend, req.model)
        save_active(req.backend, req.model)
        with _backend_lock:
            _backend = replacement
        return {"ok": True, "active": {"backend": _backend.name, "model": _backend.model}}
    except (ValueError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(""),
    task: str = Form("transcribe"),
):
    if task not in ("transcribe", "translate"):
        return JSONResponse({"error": f"invalid task: {task}"}, status_code=400)

    suffix = os.path.splitext(audio.filename or "")[1] or ".bin"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(await audio.read())
            tmp_path = tf.name

        with _backend_lock:
            backend = _backend
        result = backend.transcribe(tmp_path, language.strip() or None, task)
        return result.to_dict()
    except Exception as e:  # noqa: BLE001 — surface a clean error to the caller
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def main():
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
