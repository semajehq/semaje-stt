"""Shared engine contract + the response shape every engine must return."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    confidence: float | None = None
    words: list = field(default_factory=list)


@dataclass
class TranscriptResult:
    text: str
    language: str
    duration: float
    segments: list = field(default_factory=list)  # list[Segment]
    backend: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        segments = []
        for segment in self.segments:
            item = asdict(segment) if not isinstance(segment, dict) else dict(segment)
            if item.get("speaker") is None:
                item.pop("speaker", None)
            if item.get("confidence") is None:
                item.pop("confidence", None)
            if not item.get("words"):
                item.pop("words", None)
            segments.append(item)
        d["segments"] = segments
        return d


@dataclass
class AudioInput:
    """Canonical input handed to engines: 16 kHz mono PCM, already on disk and in RAM."""

    wav_path: str
    samples: object  # numpy float32 array in [-1, 1]
    sample_rate: int = 16000

    @property
    def duration(self) -> float:
        return round(len(self.samples) / self.sample_rate, 3)


class Engine:
    """Interface every STT engine implements. Keeps the API layer engine-agnostic.

    `language` is already in the engine's own code system (Whisper: ISO 639-1 such
    as "sw"; Omnilingual: "swh_Latn"). The router does that translation.
    """

    name = "base"
    family = "base"          # "whisper" | "omnilingual"
    model = ""
    ready = False
    supports_translate = False
    supports_word_timestamps = False

    def load(self) -> None:
        """Warm/load the model. Safe to call repeatedly."""
        raise NotImplementedError

    def transcribe(self, audio: AudioInput, language: str | None, task: str) -> TranscriptResult:
        """task is 'transcribe' or 'translate'. language=None → engine decides."""
        raise NotImplementedError

    def detect_language(self, audio: AudioInput) -> tuple[str, float] | None:
        """(engine language code, probability) or None when unsupported."""
        return None
