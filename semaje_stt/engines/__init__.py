"""Engine factory. Engines import their heavy runtimes lazily inside load()."""
from .base import Engine, TranscriptResult, Segment, AudioInput


def get_engine(name: str, model: str | None = None) -> Engine:
    if name == "faster-whisper":
        from .faster_whisper_engine import FasterWhisperEngine
        return FasterWhisperEngine(model)
    if name == "whisper.cpp":
        from .whisper_cpp_engine import WhisperCppEngine
        return WhisperCppEngine(model)
    if name == "omnilingual":
        from .omnilingual_engine import OmnilingualEngine
        return OmnilingualEngine(model)
    raise ValueError(f"Unknown STT engine {name!r}. Use faster-whisper, whisper.cpp, or omnilingual.")


__all__ = ["get_engine", "Engine", "TranscriptResult", "Segment", "AudioInput"]
