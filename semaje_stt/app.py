"""Whisper STT HTTP service. See contract.md for the API."""
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from . import config
from .backends import get_backend

app = FastAPI(title="semaje Whisper STT", version="1.0.0")

# One backend instance per process. Model loads lazily on first transcribe.
_backend = get_backend()


@app.get("/health")
def health():
    return {
        "ok": True,
        "ready": _backend.ready,
        "backend": _backend.name,
        "model": _backend.model,
        "device": config.DEVICE if _backend.name == "faster-whisper" else "metal/cpu",
    }


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

        result = _backend.transcribe(tmp_path, language.strip() or None, task)
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
