"""Shared backend contract + the response shape every backend must return."""
from dataclasses import dataclass, field, asdict


@dataclass
class Segment:
    start: float
    end: float
    text: str


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
        d["segments"] = [asdict(s) if not isinstance(s, dict) else s for s in self.segments]
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
