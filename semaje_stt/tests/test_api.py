"""Gate tests: FastAPI surface with fake engines (no model, no network).

Audio goes through the real probe/normalize path, so ffmpeg is required.
"""
import asyncio
import io

import pytest
from fastapi.testclient import TestClient

import services.stt.app as stt_app
from services.stt import config
from services.stt.engines.base import Engine, Segment, TranscriptResult
from .conftest import make_wav, needs_ffmpeg

pytestmark = needs_ffmpeg


class FakeWhisper(Engine):
    name = "faster-whisper"
    family = "whisper"
    model = "fake-base"
    ready = True
    detected = ("en", 0.99)

    def load(self):
        pass

    def detect_language(self, audio):
        return self.detected

    def transcribe(self, audio, language, task):
        text = "translated text" if task == "translate" else "the quick brown fox"
        return TranscriptResult(text=text, language=language or "en", duration=audio.duration,
                                segments=[Segment(0.0, audio.duration, text)], backend=self.name, model=self.model)


class FakeOmni(Engine):
    name = "omnilingual"
    family = "omnilingual"
    model = "fake-ctc"
    ready = True

    def load(self):
        pass

    def transcribe(self, audio, language, task):
        return TranscriptResult(text="habari ya asubuhi", language=language or "", duration=audio.duration,
                                segments=[Segment(0.0, audio.duration, "habari ya asubuhi")],
                                backend=self.name, model=self.model)


@pytest.fixture
def client(monkeypatch):
    stt_app.hub.whisper = FakeWhisper()
    stt_app.hub.omnilingual = FakeOmni()
    monkeypatch.setattr(config, "API_TOKEN", "")
    return TestClient(stt_app.app)


def _audio(data=None, name="clip.wav", mime="audio/wav"):
    return {"audio": (name, io.BytesIO(data if data is not None else make_wav(1.0)), mime)}


def test_liveness_and_readiness(client):
    assert client.get("/health/live").json() == {"ok": True}
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["engines"]["whisper"]["model"] == "fake-base"
    assert body["engines"]["omnilingual"]["model"] == "fake-ctc"
    assert body["backend"] == "faster-whisper"  # v1 field kept


def test_readiness_is_503_until_a_model_is_loaded(client):
    stt_app.hub.whisper.ready = False
    stt_app.hub.omnilingual.ready = False
    assert client.get("/health/ready").status_code == 503
    assert client.get("/health").status_code == 200  # informational endpoint never fails
    stt_app.hub.omnilingual.ready = True
    assert client.get("/health/ready").status_code == 200


def test_transcribe_contract_shape(client):
    r = client.post("/transcribe", files=_audio(), data={"language": "en"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "the quick brown fox"
    assert body["engine"] == "whisper"
    assert body["language"] == "en"
    assert body["languageCode"] == "eng_Latn"
    assert body["languageName"] == "English"
    assert body["segments"][0]["text"] == "the quick brown fox"
    assert body["audio"] == {"container": "wav", "codec": "pcm_s16le", "sampleRate": 16000, "channels": 1,
                             "duration": 1.0, "bitRate": 256000, "hasVideo": False,
                             "channelLayout": ""}
    assert body["routing"]["engine"] == "whisper"
    assert set(body["timings"]) >= {"decodeMs", "transcribeMs"}
    assert body["requestId"] == r.headers["x-request-id"]


def test_low_resource_language_routes_to_omnilingual(client):
    body = client.post("/transcribe", files=_audio(), data={"language": "sw"}).json()
    assert body["engine"] == "omnilingual"
    assert body["engineLanguage"] == "swh_Latn"
    assert body["languageCode"] == "swh_Latn"
    assert body["text"] == "habari ya asubuhi"


def test_auto_detect_uses_whisper_language_id(client):
    stt_app.hub.whisper.detected = ("sw", 0.95)
    body = client.post("/transcribe", files=_audio()).json()
    assert body["engine"] == "omnilingual"
    assert "detectMs" in body["timings"]
    assert body["routing"]["reason"].startswith("detected sw")


def test_engine_override(client):
    body = client.post("/transcribe", files=_audio(), data={"language": "sw", "engine": "whisper"}).json()
    assert body["engine"] == "whisper"


def test_translate_task(client):
    r = client.post("/transcribe", files=_audio(), data={"task": "translate", "language": "sw"})
    assert r.status_code == 200
    assert r.json()["text"] == "translated text"


@pytest.mark.parametrize("field,value,code", [
    ("task", "summarize", "invalid_task"),
    ("engine", "gpt", "invalid_engine"),
    ("language", "klingon", "unknown_language"),
    ("channel", "center", "audio_invalid"),
])
def test_bad_parameters_are_400_class(client, field, value, code):
    r = client.post("/transcribe", files=_audio(), data={field: value})
    assert 400 <= r.status_code < 500
    assert r.json()["code"] == code


def test_non_audio_upload_is_415(client):
    r = client.post("/transcribe", files=_audio(b"this is plain text, not audio" * 40, "notes.txt", "text/plain"))
    assert r.status_code == 415
    assert r.json()["code"] in ("undecodable_audio", "no_audio_stream")


def test_silent_audio_is_200_not_an_error(client):
    r = client.post("/transcribe", files=_audio(make_wav(1.0, freq=0)))
    assert r.status_code == 200


def test_zero_length_audio_is_empty_transcript(client):
    r = client.post("/transcribe", files=_audio(make_wav(0.0)))
    assert r.status_code == 200
    assert r.json()["text"] == ""
    assert r.json()["segments"] == []


def test_upload_size_limit_is_413(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    r = client.post("/transcribe", files=_audio())
    assert r.status_code == 413
    assert r.json()["code"] == "upload_too_large"


def test_duration_limit_is_413(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_DURATION_SEC", 0.5)
    r = client.post("/transcribe", files=_audio(make_wav(1.0)))
    assert r.status_code == 413
    assert r.json()["code"] == "audio_too_long"


def test_bearer_token_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(config, "API_TOKEN", "s3cret")
    assert client.post("/transcribe", files=_audio()).status_code == 401
    assert client.get("/languages").status_code == 401
    assert client.get("/health/live").status_code == 200  # orchestrator probes stay open
    ok = client.post("/transcribe", files=_audio(), headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_capacity_limit_returns_503_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(stt_app, "_slots", asyncio.Semaphore(0))
    monkeypatch.setattr(config, "QUEUE_TIMEOUT_SEC", 0.05)
    r = client.post("/transcribe", files=_audio())
    assert r.status_code == 503
    assert r.headers["retry-after"] == "30"
    assert r.json()["code"] == "busy"


def test_engine_error_is_500_with_request_id(client):
    class Boom(FakeWhisper):
        def transcribe(self, *a, **k):
            raise RuntimeError("model exploded")

    stt_app.hub.whisper = Boom()
    r = client.post("/transcribe", files=_audio(), data={"language": "en"}, headers={"x-request-id": "req-123"})
    assert r.status_code == 500
    assert r.json() == {"error": "model exploded", "code": "internal", "requestId": "req-123"}


def test_omnilingual_missing_degrades_gracefully(client):
    stt_app.hub.omnilingual = None
    body = client.post("/transcribe", files=_audio(), data={"language": "sw"}).json()
    assert body["engine"] == "whisper"
    r = client.post("/transcribe", files=_audio(), data={"language": "kik"})
    assert r.status_code == 400
    assert r.json()["code"] == "unknown_language"


def test_broken_omnilingual_runtime_is_disabled_at_warmup(client):
    class BrokenOmni(FakeOmni):
        def load(self):
            raise OSError("fairseq2 requires libsndfile")

    stt_app.hub.omnilingual = BrokenOmni()
    stt_app._warmup()
    assert stt_app.hub.omnilingual is None
    assert client.get("/health").json()["engines"]["omnilingual"] is None
    body = client.post("/transcribe", files=_audio(), data={"language": "sw"}).json()
    assert body["engine"] == "whisper"  # degraded, not a 500


def test_languages_endpoint_filters(client):
    body = client.get("/languages").json()
    assert body["total"] >= 1600
    assert body["available"] == ["omnilingual", "whisper"]
    sw = next(l for l in body["languages"] if l["code"] == "swh_Latn")
    assert sw["engines"] == ["whisper", "omnilingual"]
    assert sw["whisperStrong"] is False
    q = client.get("/languages", params={"q": "kikuyu"}).json()
    assert [l["code"] for l in q["languages"]] == ["kik_Latn"]
    only_whisper = client.get("/languages", params={"engine": "whisper"}).json()
    assert only_whisper["count"] == 100


def test_models_registry_lists_both_families(client):
    body = client.get("/models").json()
    families = {m["family"] for m in body["models"]}
    assert families == {"whisper", "omnilingual"}
    assert all(m["license"]["spdx"] in ("MIT", "Apache-2.0") for m in body["models"])
    assert body["hardware"]["cpuCount"] >= 1


def test_activate_rejects_unknown_model(client):
    r = client.post("/models/activate", json={"backend": "omnilingual", "model": "gpt-4"})
    assert r.status_code == 400


def test_metrics_exposed(client):
    client.post("/transcribe", files=_audio(), data={"language": "en"})
    text = client.get("/metrics").text
    assert "semaje_stt_audio_seconds_total" in text
    assert 'semaje_stt_http_requests_total{method="POST",path="/transcribe"' in text
