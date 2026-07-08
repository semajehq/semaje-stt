"""Gate test: FastAPI surface with a fake backend (no model, no network)."""
import io
import pytest
from fastapi.testclient import TestClient

import services.whisper.app as whisper_app
from services.whisper.backends.base import Backend, TranscriptResult, Segment


class FakeBackend(Backend):
    name = "fake"
    model = "fake-model"
    ready = True

    def load(self):
        pass

    def transcribe(self, audio_path, language, task):
        if task == "translate":
            text = "translated text"
        else:
            text = "the quick brown fox"
        return TranscriptResult(
            text=text,
            language=language or "en",
            duration=2.0,
            segments=[Segment(0.0, 2.0, text)],
            backend=self.name,
            model=self.model,
        )


@pytest.fixture
def client():
    whisper_app._backend = FakeBackend()
    return TestClient(whisper_app.app)


def _audio():
    return {"audio": ("clip.wav", io.BytesIO(b"RIFFfake"), "audio/wav")}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["backend"] == "fake"
    assert body["model"] == "fake-model"


def test_transcribe_returns_contract_shape(client):
    r = client.post("/transcribe", files=_audio(), data={"language": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "the quick brown fox"
    assert body["language"] == "en"
    assert body["backend"] == "fake"
    assert body["segments"][0] == {"start": 0.0, "end": 2.0, "text": "the quick brown fox"}


def test_translate_task(client):
    r = client.post("/transcribe", files=_audio(), data={"task": "translate"})
    assert r.status_code == 200
    assert r.json()["text"] == "translated text"


def test_invalid_task_400(client):
    r = client.post("/transcribe", files=_audio(), data={"task": "summarize"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_backend_error_returns_500(client):
    class Boom(FakeBackend):
        def transcribe(self, *a, **k):
            raise RuntimeError("model exploded")

    whisper_app._backend = Boom()
    c = TestClient(whisper_app.app)
    r = c.post("/transcribe", files=_audio())
    assert r.status_code == 500
    assert r.json()["error"] == "model exploded"
