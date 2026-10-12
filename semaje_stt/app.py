"""semaje STT service — open-source speech-to-text for any audio, any language.

See contract.md for the API. Engines: Whisper (faster-whisper | whisper.cpp)
and Meta Omnilingual ASR, chosen per request by router.route().
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import config, metrics
from .audio import AudioError, UploadTooLarge, normalize, probe, read_wav
from .diarization import diarize_segments
from .engines import AudioInput, Engine, get_engine
from .models import (
    WHISPER_BACKENDS, ENGINES, active_config, delete_model, download_model, hardware_info,
    list_models, model_exists, omnilingual_available, save_active,
)
from .router import UnknownLanguage, registry, route, whisper_strong

VERSION = "2.0.0"


# ── Logging ──────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "service": "stt",
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


log = logging.getLogger("semaje.stt")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(JsonFormatter())
    log.addHandler(_handler)
    log.setLevel(config.LOG_LEVEL.upper())
    log.propagate = False


def _log(level: int, msg: str, **fields) -> None:
    log.log(level, msg, extra={"fields": fields})


# ── Engine hub ───────────────────────────────────────────────────
class EngineHub:
    """Holds one engine per family; swaps are atomic under a lock."""

    def __init__(self):
        self.lock = threading.Lock()
        active = active_config()
        self.whisper: Engine | None = get_engine(active["whisper"]["backend"], active["whisper"]["model"])
        self.omnilingual: Engine | None = None
        if config.OMNILINGUAL_ENABLED and omnilingual_available():
            self.omnilingual = get_engine("omnilingual", active["omnilingual"]["model"])

    def available(self) -> set[str]:
        with self.lock:
            return {name for name, eng in (("whisper", self.whisper), ("omnilingual", self.omnilingual)) if eng}

    def get(self, family: str) -> Engine:
        with self.lock:
            engine = self.whisper if family == "whisper" else self.omnilingual
        if engine is None:
            raise RuntimeError(f"The {family} engine is not available on this host.")
        return engine

    def disable(self, family: str) -> None:
        with self.lock:
            if family == "omnilingual":
                self.omnilingual = None
            else:
                self.whisper = None

    def swap(self, engine: Engine) -> None:
        with self.lock:
            if engine.family == "omnilingual":
                self.omnilingual = engine
            else:
                self.whisper = engine

    def describe(self) -> dict:
        with self.lock:
            engines = {"whisper": self.whisper, "omnilingual": self.omnilingual}
        return {
            family: (
                {"backend": eng.name, "model": eng.model, "ready": eng.ready} if eng else None
            )
            for family, eng in engines.items()
        }


hub = EngineHub()
_slots = asyncio.Semaphore(config.MAX_CONCURRENCY)
_inflight = 0


def _warmup() -> None:
    """Load (and if needed download) every engine before traffic arrives, so no
    request ever blocks on a multi-GB model download. /health/ready stays false
    until at least one engine is loaded."""
    for family in sorted(hub.available()):
        started = time.perf_counter()
        try:
            hub.get(family).load()
            _log(logging.INFO, "warmup.done", engine=family, seconds=round(time.perf_counter() - started, 1))
        except (ImportError, OSError) as exc:
            # A broken runtime install (e.g. missing libsndfile) will never load: take the
            # engine out of routing so requests degrade to the other engine instead of 500ing.
            if family == "omnilingual":
                hub.disable("omnilingual")
            _log(logging.ERROR, "warmup.engine_disabled" if family == "omnilingual" else "warmup.failed",
                 engine=family, error=str(exc))
        except Exception as exc:  # noqa: BLE001 — transient (network, disk): the request path retries
            _log(logging.WARNING, "warmup.failed", engine=family, error=str(exc))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("STT_WARMUP", "true").lower() == "true":
        threading.Thread(target=_warmup, daemon=True).start()
    _log(logging.INFO, "stt.start", version=VERSION, engines=sorted(hub.available()),
         routing=config.ROUTING, maxConcurrency=config.MAX_CONCURRENCY)
    yield


app = FastAPI(title="semaje STT", version=VERSION, lifespan=lifespan)

PUBLIC_PATHS = {"/health/live", "/health"}


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    if config.API_TOKEN and request.url.path not in PUBLIC_PATHS:
        if request.headers.get("authorization", "") != f"Bearer {config.API_TOKEN}":
            return JSONResponse({"error": "unauthorized", "code": "unauthorized", "requestId": request_id},
                                status_code=401, headers={"x-request-id": request_id})
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    response.headers["x-request-id"] = request_id
    metrics.observe_http(request.url.path, request.method, response.status_code, elapsed)
    if request.url.path not in ("/health/live", "/metrics"):
        _log(logging.INFO, "http", requestId=request_id, method=request.method,
             path=request.url.path, status=response.status_code, ms=round(elapsed * 1000, 1))
    return response


def _error(request: Request, status: int, message: str, code: str, **extra) -> JSONResponse:
    headers = {"x-request-id": request.state.request_id, **extra.pop("headers", {})}
    return JSONResponse({"error": message, "code": code, "requestId": request.state.request_id, **extra},
                        status_code=status, headers=headers)


# ── Health + metrics ─────────────────────────────────────────────
@app.get("/health/live")
def live():
    return {"ok": True}


@app.get("/health/ready")
def ready():
    """Orchestrator readiness: 503 until at least one engine has its model loaded."""
    body = health()
    return body if body["ready"] else JSONResponse(body, status_code=503)


@app.get("/health")
def health():
    engines = hub.describe()
    whisper = engines.get("whisper") or {}
    return {
        "ok": bool(hub.available()),
        "version": VERSION,
        "ready": any(e and e["ready"] for e in engines.values()),
        "engines": engines,
        "routing": config.ROUTING,
        "inflight": _inflight,
        "maxConcurrency": config.MAX_CONCURRENCY,
        # v1 fields, kept so older callers keep working
        "backend": whisper.get("backend"),
        "model": whisper.get("model"),
        "device": config.DEVICE,
        "computeType": config.COMPUTE_TYPE,
    }


@app.get("/metrics")
def prometheus():
    return PlainTextResponse(metrics.render(), media_type=metrics.CONTENT_TYPE)


# ── Languages ────────────────────────────────────────────────────
@app.get("/languages")
def languages(q: str = "", engine: str = ""):
    available = hub.available()
    strong = whisper_strong()
    needle = q.strip().lower()
    rows = []
    for lang in registry().languages:
        if engine == "whisper" and not lang.whisper:
            continue
        if engine == "omnilingual" and not lang.omnilingual:
            continue
        if needle and needle not in lang.name.lower() and needle not in lang.code.lower() \
                and needle != (lang.iso1 or "") and needle != (lang.whisper or ""):
            continue
        engines = []
        if lang.whisper and "whisper" in available:
            engines.append("whisper")
        if lang.omnilingual and "omnilingual" in available:
            engines.append("omnilingual")
        rows.append({
            **lang.to_dict(),
            "engines": engines,
            "whisperStrong": bool(lang.whisper and lang.whisper in strong),
            "translate": bool(lang.whisper and "whisper" in available),
        })
    return {
        "count": len(rows),
        "total": len(registry().languages),
        "available": sorted(available),
        "languages": rows,
    }


# ── Model management ─────────────────────────────────────────────
class ModelRequest(BaseModel):
    backend: str
    model: str


@app.get("/models")
def models():
    whisper = hub.describe()["whisper"] or {}
    return {
        "models": list_models(),
        "active": {"backend": whisper.get("backend"), "model": whisper.get("model")},
        "activeByFamily": active_config(),
        "hardware": hardware_info(),
    }


@app.post("/models/download")
def download(req: ModelRequest, request: Request):
    try:
        return {"ok": True, **download_model(req.backend, req.model)}
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        return _error(request, 400, str(exc), "model_download_failed")


@app.delete("/models/{backend}/{model}")
def remove(backend: str, model: str, request: Request):
    try:
        delete_model(backend, model)
        return {"ok": True}
    except (ValueError, OSError) as exc:
        return _error(request, 400, str(exc), "model_delete_failed")


@app.post("/models/activate")
def activate(req: ModelRequest, request: Request):
    if req.backend not in ENGINES or not model_exists(req.model, req.backend):
        return _error(request, 400, "unknown backend or model", "unknown_model")
    if req.backend == "omnilingual" and not omnilingual_available():
        return _error(request, 400, "omnilingual-asr is not installed on this host", "engine_unavailable")
    try:
        replacement = get_engine(req.backend, req.model)
        save_active(req.backend, req.model)
        hub.swap(replacement)
        return {"ok": True, "active": {"backend": replacement.name, "model": replacement.model}}
    except (ValueError, OSError) as exc:
        return _error(request, 400, str(exc), "model_activate_failed")


# ── Transcription ────────────────────────────────────────────────
async def _save_upload(upload: UploadFile, dest: str) -> int:
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with open(dest, "wb") as out:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                raise UploadTooLarge(f"Upload exceeds {config.MAX_UPLOAD_MB} MB (STT_MAX_UPLOAD_MB).")
            out.write(chunk)
    return written


def _canonical(requested, engine_language: str, family: str):
    """Registry entry for the transcript language: the request, else what the engine reported."""
    if requested is not None:
        return requested
    if not engine_language:
        return None
    if family == "whisper":
        found = registry().from_whisper(engine_language)
        if found:
            return found
    try:
        return registry().resolve(engine_language)
    except UnknownLanguage:
        return None


def _transcribe_blocking(src: str, workdir: str, language: str, task: str, diarize: bool,
                         policy: str, channel: str, enhance: bool) -> dict:
    timings = {}
    t0 = time.perf_counter()
    info = probe(src)
    wav = os.path.join(workdir, "canonical.wav")
    normalize(src, wav, channel=channel, enhance=enhance, layout=info.channel_layout)
    samples = read_wav(wav)
    audio = AudioInput(wav_path=wav, samples=samples)
    timings["decodeMs"] = round((time.perf_counter() - t0) * 1000, 1)

    lang = registry().resolve(language)
    available = hub.available()
    detected = None
    if lang is None and policy == "auto" and available == {"whisper", "omnilingual"} \
            and task == "transcribe" and audio.duration > 0:
        t1 = time.perf_counter()
        detected = hub.get("whisper").detect_language(audio)
        timings["detectMs"] = round((time.perf_counter() - t1) * 1000, 1)

    decision = route(lang, policy, available, detected=detected, task=task, strong=whisper_strong())
    engine = hub.get(decision.engine)

    t2 = time.perf_counter()
    if audio.duration == 0:
        from .engines import TranscriptResult
        result = TranscriptResult(text="", language=decision.language or "", duration=0.0,
                                  backend=engine.name, model=engine.model)
    else:
        result = engine.transcribe(audio, decision.language, task)
    timings["transcribeMs"] = round((time.perf_counter() - t2) * 1000, 1)

    resolved = _canonical(lang, result.language, engine.family)
    payload = None
    if diarize:
        t3 = time.perf_counter()
        result.segments, diarization_backend = diarize_segments(wav, result.segments)
        timings["diarizeMs"] = round((time.perf_counter() - t3) * 1000, 1)
        payload = result.to_dict()
        payload["diarizationBackend"] = diarization_backend
    payload = payload or result.to_dict()
    payload.update({
        "engine": engine.family,
        "engineLanguage": result.language or decision.language,
        "language": (resolved.whisper or resolved.code) if resolved else (result.language or ""),
        "languageCode": resolved.code if resolved else None,
        "languageName": resolved.name if resolved else None,
        "routing": decision.to_dict(),
        "audio": info.to_dict(),
        "timings": timings,
    })
    metrics.observe_transcription(engine.family, engine.model, info.duration,
                                  sum(v for v in timings.values()) / 1000)
    return payload


@app.post("/transcribe")
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
    language: str = Form(""),
    task: str = Form("transcribe"),
    diarize: bool = Form(False),
    engine: str = Form(""),
    channel: str = Form("mix"),
    enhance: bool = Form(False),
):
    global _inflight
    if task not in ("transcribe", "translate"):
        return _error(request, 400, f"invalid task: {task}", "invalid_task")
    policy = (engine or config.ROUTING).strip().lower()
    if policy not in ("auto", "whisper", "omnilingual"):
        return _error(request, 400, f"invalid engine: {engine}", "invalid_engine")

    workdir = tempfile.mkdtemp(prefix="semaje-stt-")
    try:
        suffix = os.path.splitext(audio.filename or "")[1][:12] or ".bin"
        src = os.path.join(workdir, f"upload{suffix}")
        await _save_upload(audio, src)

        try:
            await asyncio.wait_for(_slots.acquire(), timeout=config.QUEUE_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            metrics.REJECTED.inc()
            return _error(request, 503, "STT is at capacity; retry shortly.", "busy",
                          headers={"Retry-After": "30"})
        _inflight += 1
        try:
            payload = await run_in_threadpool(
                _transcribe_blocking, src, workdir, language, task, diarize, policy, channel, enhance,
            )
        finally:
            _inflight -= 1
            _slots.release()
        payload["requestId"] = request.state.request_id
        return payload
    except AudioError as exc:
        metrics.FAILURES.labels(code=exc.code).inc()
        return _error(request, exc.status, exc.message, exc.code)
    except UnknownLanguage as exc:
        metrics.FAILURES.labels(code=exc.code).inc()
        return _error(request, 400, str(exc), exc.code)
    except Exception as exc:  # noqa: BLE001 — surface a clean error to the caller
        metrics.FAILURES.labels(code="internal").inc()
        _log(logging.ERROR, "transcribe.failed", requestId=request.state.request_id, error=str(exc))
        return _error(request, 500, str(exc), "internal")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
