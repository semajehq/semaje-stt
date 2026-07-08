"""faster-whisper backend — CTranslate2, pure Python, no torch."""
from .base import Backend, TranscriptResult, Segment
from .. import config


class FasterWhisperBackend(Backend):
    name = "faster-whisper"

    def __init__(self):
        self.model_name = config.MODEL
        self.model = config.MODEL
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # imported lazily so gate tests don't need it
        self._model = WhisperModel(
            self.model_name,
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
        )
        segments = []
        parts = []
        for s in segments_iter:
            txt = s.text.strip()
            segments.append(Segment(start=round(s.start, 3), end=round(s.end, 3), text=txt))
            parts.append(txt)

        return TranscriptResult(
            text=" ".join(parts).strip(),
            language=info.language or (language or ""),
            duration=round(info.duration, 3),
            segments=segments,
            backend=self.name,
            model=self.model_name,
        )
