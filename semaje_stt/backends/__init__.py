"""Backend factory — picks the implementation from config.WHISPER_BACKEND."""
from .. import config
from .base import Backend, TranscriptResult, Segment


def get_backend() -> Backend:
    if config.BACKEND == "whisper.cpp":
        from .whisper_cpp_backend import WhisperCppBackend
        return WhisperCppBackend()
    if config.BACKEND == "faster-whisper":
        from .faster_whisper_backend import FasterWhisperBackend
        return FasterWhisperBackend()
    raise ValueError(
        f"Unknown WHISPER_BACKEND={config.BACKEND!r}. Use 'faster-whisper' or 'whisper.cpp'."
    )


__all__ = ["get_backend", "Backend", "TranscriptResult", "Segment"]
