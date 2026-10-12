"""whisper.cpp engine (MIT) — shells out to the compiled whisper-cli binary.

The service already hands every engine a 16 kHz mono WAV, which is exactly what
whisper-cli needs. Output is parsed from whisper-cli's `-oj` JSON file.
"""
import json
import os
import subprocess
import tempfile

from .base import Engine, TranscriptResult, Segment, AudioInput
from .. import config
from ..models import model_path


def parse_whisper_cpp_json(data: dict, model_name: str) -> TranscriptResult:
    """Pure parser for whisper-cli `-oj` output. Unit-tested without the binary."""
    items = data.get("transcription", []) or []
    segments = []
    parts = []
    duration = 0.0
    for it in items:
        offsets = it.get("offsets", {}) or {}
        start = (offsets.get("from", 0) or 0) / 1000.0
        end = (offsets.get("to", 0) or 0) / 1000.0
        txt = (it.get("text", "") or "").strip()
        if end > duration:
            duration = end
        if txt:
            segments.append(Segment(start=round(start, 3), end=round(end, 3), text=txt))
            parts.append(txt)

    language = (data.get("result", {}) or {}).get("language", "") or ""
    return TranscriptResult(
        text=" ".join(parts).strip(),
        language=language,
        duration=round(duration, 3),
        segments=segments,
        backend="whisper.cpp",
        model=model_name,
    )


class WhisperCppEngine(Engine):
    name = "whisper.cpp"
    family = "whisper"
    supports_translate = True

    def __init__(self, model_name=None):
        self.bin = config.WHISPER_CPP_BIN
        self.model_name = model_name or config.MODEL
        managed_path = model_path(self.name, self.model_name)
        self.model_path = str(managed_path if managed_path.exists() else config.WHISPER_CPP_MODEL)
        self.model = self.model_name

    def load(self) -> None:
        if not os.path.exists(self.bin):
            raise RuntimeError(
                f"whisper.cpp binary not found at {self.bin}. Run services/stt/setup.sh."
            )
        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"whisper.cpp model not found at {self.model_path}. Run services/stt/setup.sh."
            )
        self.ready = True

    def transcribe(self, audio: AudioInput, language, task) -> TranscriptResult:
        self.load()
        with tempfile.TemporaryDirectory() as tmp:
            out_prefix = os.path.join(tmp, "out")
            cmd = [
                self.bin,
                "-m", self.model_path,
                "-f", audio.wav_path,
                "-oj",                  # JSON output → out.json
                "-of", out_prefix,
                "-l", language or "auto",
            ]
            if task == "translate":
                cmd.append("-tr")
            if config.WHISPER_CPP_THREADS:
                cmd += ["-t", config.WHISPER_CPP_THREADS]

            subprocess.run(cmd, check=True, capture_output=True, timeout=config.FFMPEG_TIMEOUT_SEC)

            with open(out_prefix + ".json", "r", encoding="utf-8") as f:
                data = json.load(f)

        result = parse_whisper_cpp_json(data, self.model)
        result.duration = max(result.duration, audio.duration)
        return result
