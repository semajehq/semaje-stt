"""faster-whisper backend — CTranslate2, pure Python, no torch."""
import math
from .base import Backend, TranscriptResult, Segment
from .. import config
from ..models import download_model, model_path, is_installed


class FasterWhisperBackend(Backend):
    name = "faster-whisper"

    def __init__(self, model_name=None):
        self.model_name = model_name or config.MODEL
        self.model = self.model_name
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # imported lazily so gate tests don't need it
        if not is_installed(self.name, self.model_name):
            download_model(self.name, self.model_name)
        source = str(model_path(self.name, self.model_name)) \
            if is_installed(self.name, self.model_name) else self.model_name
        self._model = WhisperModel(
            source,
            device=config.DEVICE,
            compute_type=config.COMPUTE_TYPE,
        )
        self.ready = True

    def transcribe(self, audio_path: str, language: str | None, task: str) -> TranscriptResult:
        self.load()
        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language or None,
            task=task,
            vad_filter=True,  # skip long silences, cuts hallucinated filler
            word_timestamps=True,
        )
        segments = []
        parts = []
        for s in segments_iter:
            txt = s.text.strip()
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
