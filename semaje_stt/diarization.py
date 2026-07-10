"""Optional local pyannote diarization with a deterministic one-speaker fallback."""
from __future__ import annotations

import os
from .backends.base import Segment

_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    model = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return None
    _pipeline = Pipeline.from_pretrained(model, use_auth_token=os.getenv("HF_TOKEN") or None)
    return _pipeline


def diarize_segments(audio_path: str, segments: list[Segment]) -> tuple[list[Segment], str]:
    pipeline = _load_pipeline()
    if pipeline is None:
        for segment in segments:
            segment.speaker = "SPEAKER_00"
        return segments, "single-speaker-fallback"

    result = pipeline(audio_path)
    turns = [(turn.start, turn.end, speaker) for turn, _, speaker in result.itertracks(yield_label=True)]
    for segment in segments:
        midpoint = (segment.start + segment.end) / 2
        segment.speaker = next(
            (speaker for start, end, speaker in turns if start <= midpoint <= end),
            "SPEAKER_00",
        )
    return segments, "pyannote-local"
