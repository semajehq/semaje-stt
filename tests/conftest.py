"""Shared fixtures: synthetic audio generated in-process (no model, no network)."""
import io
import math
import shutil
import struct
import wave

import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


def make_wav(seconds: float = 1.0, sr: int = 16000, channels: int = 1, freq: float = 440.0,
             silence_gaps: list[tuple[float, float]] | None = None) -> bytes:
    """16-bit PCM WAV with a tone; `silence_gaps` are (start, end) seconds zeroed out."""
    gaps = silence_gaps or []
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(int(seconds * sr)):
            t = i / sr
            value = 0 if any(a <= t < b for a, b in gaps) else int(0.5 * 32767 * math.sin(2 * math.pi * freq * t))
            frames += struct.pack("<h", value) * channels
        wf.writeframes(bytes(frames))
    return buf.getvalue()


@pytest.fixture
def wav_bytes():
    return make_wav(1.0)
