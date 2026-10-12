"""faster-whisper engine — CTranslate2 Whisper (MIT), CPU/CUDA, no torch."""
import math

from .base import Engine, TranscriptResult, Segment, AudioInput
from .. import config
from ..models import download_model, model_path, is_installed


class FasterWhisperEngine(Engine):
    name = "faster-whisper"
    family = "whisper"
    supports_translate = True
    supports_word_timestamps = True

    def __init__(self, model_name=None):
        self.model_name = model_name or config.MODEL
        self.model = self.model_name
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # lazy: gate tests don't load models
        if not is_installed(self.name, self.model_name):
            download_model(self.name, self.model_name)
        self._model = WhisperModel(
            str(model_path(self.name, self.model_name)),
            device=config.DEVICE,
            compute_type=config.COMPUTE_TYPE,
        )
        self.ready = True

    def detect_language(self, audio: AudioInput):
        self.load()
        # Vote over up to 3 speech windows so a noisy intro can't decide alone.
        language, probability, _ = self._model.detect_language(
            audio.samples, vad_filter=True, language_detection_segments=3,
        )
        return language, float(probability or 0.0)

    def transcribe(self, audio: AudioInput, language, task) -> TranscriptResult:
        self.load()
        segments_iter, info = self._model.transcribe(
            audio.samples,
            language=language or None,
            task=task,
            vad_filter=True,  # skip long silences, cuts hallucinated filler
            word_timestamps=True,
        )
        segments, parts = [], []
        for s in segments_iter:
            txt = s.text.strip()
            if not txt:
                continue
            confidence = max(0.0, min(1.0, math.exp(float(s.avg_logprob or -10))))
            words = [
                {
                    "start": round(float(w.start or 0), 3),
                    "end": round(float(w.end or 0), 3),
                    "word": w.word,
                    "probability": round(float(w.probability or 0), 4),
                }
                for w in (s.words or [])
            ]
            segments.append(Segment(
                start=round(s.start, 3), end=round(s.end, 3), text=txt,
                confidence=round(confidence, 4), words=words,
            ))
            parts.append(txt)

        return TranscriptResult(
            text=" ".join(parts).strip(),
            language=info.language or (language or ""),
            duration=round(info.duration, 3),
            segments=segments,
            backend=self.name,
            model=self.model_name,
        )
