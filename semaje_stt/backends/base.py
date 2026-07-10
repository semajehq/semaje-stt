"""Shared backend contract + the response shape every backend must return."""
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


class Backend:
    """Interface both backends implement. Keeps the API layer backend-agnostic."""

    name = "base"
    model = ""
    ready = False

    def load(self) -> None:
        """Warm/load the model. Safe to call repeatedly."""
        raise NotImplementedError

    def transcribe(self, audio_path: str, language: str | None, task: str) -> TranscriptResult:
        """task is 'transcribe' or 'translate'. language=None → auto-detect."""
        raise NotImplementedError
