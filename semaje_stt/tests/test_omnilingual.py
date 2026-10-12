"""Gate tests: Omnilingual engine chunking/batching with a fake pipeline (no model)."""
import numpy as np
import pytest

from services.stt import config
from services.stt.engines.base import AudioInput
from services.stt.engines.omnilingual_engine import OmnilingualEngine, assemble_segments

SR = 16000


def test_assemble_pairs_chunks_with_text_and_drops_empties():
    segs = assemble_segments([(0, SR * 2), (SR * 2, SR * 5), (SR * 5, SR * 6)], ["habari", "  ", "ya  asubuhi "], SR)
    assert [(s.start, s.end, s.text) for s in segs] == [(0.0, 2.0, "habari"), (5.0, 6.0, "ya asubuhi")]


def test_assemble_rejects_count_mismatch():
    with pytest.raises(ValueError):
        assemble_segments([(0, SR)], [], SR)


class FakePipeline:
    def __init__(self):
        self.calls = []

    def transcribe(self, batch, **kwargs):
        self.calls.append({"n": len(batch), **kwargs})
        for item in batch:
            seconds = len(item["waveform"]) / item["sample_rate"]
            assert seconds < 40, "Omnilingual rejects inputs >= 40 s"
        return [f"chunk{len(self.calls)}-{i}" for i in range(len(batch))]


def _engine(model, monkeypatch, batch=2):
    monkeypatch.setattr(config, "OMNILINGUAL_BATCH", batch)
    monkeypatch.setattr(config, "OMNILINGUAL_MAX_CHUNK_SEC", 30.0)
    engine = OmnilingualEngine(model)
    engine._pipeline = FakePipeline()
    engine.ready = True
    return engine


def _speech(seconds):
    t = np.arange(int(seconds * SR)) / SR
    x = (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    for start in range(25, int(seconds), 25):  # a pause every 25 s
        x[start * SR:(start + 1) * SR] = 0
    return AudioInput(wav_path="unused.wav", samples=x)


def test_long_audio_is_chunked_under_40s_and_batched(monkeypatch):
    engine = _engine("omniASR_CTC_300M_v2", monkeypatch, batch=2)
    result = engine.transcribe(_speech(130), "swh_Latn", "transcribe")
    calls = engine._pipeline.calls
    assert sum(c["n"] for c in calls) == len(result.segments) >= 5
    assert all(c["n"] <= 2 for c in calls)
    assert result.duration == 130.0
    assert result.segments[-1].end <= 130.0
    assert result.backend == "omnilingual"


def test_ctc_models_get_no_language_hint(monkeypatch):
    engine = _engine("omniASR_CTC_300M_v2", monkeypatch)
    engine.transcribe(_speech(5), "swh_Latn", "transcribe")
    assert "lang" not in engine._pipeline.calls[0]


def test_llm_models_get_language_hint_per_item(monkeypatch):
    engine = _engine("omniASR_LLM_300M_v2", monkeypatch)
    engine.transcribe(_speech(5), "kik_Latn", "transcribe")
    assert engine._pipeline.calls[0]["lang"] == ["kik_Latn"]


def test_silence_produces_empty_transcript_without_calling_the_model(monkeypatch):
    engine = _engine("omniASR_CTC_300M_v2", monkeypatch)
    result = engine.transcribe(AudioInput("x.wav", np.zeros(SR * 10, dtype=np.float32)), None, "transcribe")
    assert result.text == "" and result.segments == []
    assert engine._pipeline.calls == []


def test_translate_is_rejected(monkeypatch):
    engine = _engine("omniASR_CTC_300M_v2", monkeypatch)
    with pytest.raises(ValueError, match="only supports task=transcribe"):
        engine.transcribe(_speech(3), None, "translate")
