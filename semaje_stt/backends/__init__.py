"""Backend factory — picks the implementation from config.WHISPER_BACKEND."""
from .. import config
from .base import Backend, TranscriptResult, Segment


def get_backend(name=None, model=None) -> Backend:
    backend = name or config.BACKEND
    if backend == "whisper.cpp":
        from .whisper_cpp_backend import WhisperCppBackend
        return WhisperCppBackend(model)
    if backend == "faster-whisper":
        from .faster_whisper_backend import FasterWhisperBackend
        return FasterWhisperBackend(model)
    raise ValueError(
        f"Unknown WHISPER_BACKEND={backend!r}. Use 'faster-whisper' or 'whisper.cpp'."
    )


__all__ = ["get_backend", "Backend", "TranscriptResult", "Segment"]
