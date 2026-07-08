"""whisper.cpp backend — shells out to the compiled whisper-cli binary.

whisper.cpp only accepts 16 kHz mono WAV, so we transcode with ffmpeg first.
Output is parsed from whisper-cli's `-oj` JSON file.
"""
import json
import os
import subprocess
import tempfile

from .base import Backend, TranscriptResult, Segment
from .. import config


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


class WhisperCppBackend(Backend):
    name = "whisper.cpp"

    def __init__(self):
        self.bin = config.WHISPER_CPP_BIN
        self.model_path = config.WHISPER_CPP_MODEL
        self.model = os.path.basename(self.model_path)

    def load(self) -> None:
        if not os.path.exists(self.bin):
            raise RuntimeError(
                f"whisper.cpp binary not found at {self.bin}. Run services/whisper/setup.sh."
            )
        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"whisper.cpp model not found at {self.model_path}. Run services/whisper/setup.sh."
            )
        self.ready = True

    def _to_wav16k(self, audio_path: str, out_path: str) -> None:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", audio_path,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path],
            check=True, capture_output=True,
        )

    def transcribe(self, audio_path: str, language: str | None, task: str) -> TranscriptResult:
        self.load()
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "audio.wav")
            self._to_wav16k(audio_path, wav)

            out_prefix = os.path.join(tmp, "out")
            cmd = [
                self.bin,
                "-m", self.model_path,
                "-f", wav,
                "-oj",                  # JSON output → out.json
                "-of", out_prefix,
                "-l", language or "auto",
            ]
            if task == "translate":
                cmd.append("-tr")
            if config.WHISPER_CPP_THREADS:
                cmd += ["-t", config.WHISPER_CPP_THREADS]

            subprocess.run(cmd, check=True, capture_output=True)

            with open(out_prefix + ".json", "r", encoding="utf-8") as f:
                data = json.load(f)

        return parse_whisper_cpp_json(data, self.model)
