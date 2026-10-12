"""Audio intake: probe anything ffmpeg can open, normalize it, chunk it.

Every engine receives the same canonical input — 16 kHz mono PCM WAV — no matter
what the caller uploaded: MP3, AAC/M4A, Opus/WebM, Ogg/Vorbis, FLAC, ALAC, AMR
(phone voice notes), 8 kHz telephony WAV, 24/32-bit studio WAV, 5.1 surround,
or a video container (MP4, MOV, MKV, AVI) with an audio track.

The pure functions here (parse_probe, build_normalize_args, split_on_silence)
carry the logic and are gate-tested without ffmpeg. The I/O wrappers are thin.
"""
from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass, asdict

from . import config

TARGET_SR = 16000


class AudioError(Exception):
    """Client-facing audio problem. `status` is the HTTP code to return."""

    status = 422
    code = "audio_invalid"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NoAudioStream(AudioError):
    status = 415
    code = "no_audio_stream"


class UndecodableAudio(AudioError):
    status = 415
    code = "undecodable_audio"


class AudioTooLong(AudioError):
    status = 413
    code = "audio_too_long"


class UploadTooLarge(AudioError):
    status = 413
    code = "upload_too_large"


@dataclass
class AudioInfo:
    container: str
    codec: str
    sample_rate: int
    channels: int
    duration: float
    bit_rate: int | None = None
    has_video: bool = False
    channel_layout: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return {
            "container": d["container"], "codec": d["codec"], "sampleRate": d["sample_rate"],
            "channels": d["channels"], "duration": d["duration"], "bitRate": d["bit_rate"],
            "hasVideo": d["has_video"], "channelLayout": d["channel_layout"],
        }


def _float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_probe(data: dict, max_duration: float | None = None) -> AudioInfo:
    """Turn `ffprobe -show_format -show_streams -of json` output into AudioInfo.

    Picks the first audio stream. Duration falls back from stream → format
    because many containers (WebM from MediaRecorder, raw AAC) only report one.
    """
    streams = data.get("streams") or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise NoAudioStream("The file has no audio track.")
    fmt = data.get("format") or {}
    duration = _float(audio.get("duration")) or _float(fmt.get("duration"))
    limit = config.MAX_DURATION_SEC if max_duration is None else max_duration
    if limit and duration > limit:
        raise AudioTooLong(
            f"Audio is {duration / 3600:.2f} h; the limit is {limit / 3600:.2f} h "
            f"(STT_MAX_DURATION_SEC)."
        )
    return AudioInfo(
        container=(fmt.get("format_name") or "unknown").split(",")[0],
        codec=audio.get("codec_name") or "unknown",
        sample_rate=_int_or_none(audio.get("sample_rate")) or 0,
        channels=_int_or_none(audio.get("channels")) or 0,
        duration=round(duration, 3),
        bit_rate=_int_or_none(audio.get("bit_rate")) or _int_or_none(fmt.get("bit_rate")),
        channel_layout=audio.get("channel_layout") or "",
        has_video=any(
            s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")
            for s in streams
        ),
    )


# Light speech enhancement: cut rumble, FFT denoise, even out quiet/loud talkers.
ENHANCE_FILTER = "highpass=f=80,afftdn=nf=-25,dynaudnorm=f=150:g=15"


# Surround layouts that carry a front-center (dialogue) channel.
CENTER_LAYOUTS = {"3.0", "3.1", "4.0", "4.1", "5.0", "5.1", "6.0", "6.1", "7.0", "7.1"}
# Films and broadcast put speech in FC; a flat downmix buries it under music/effects.
DIALOGUE_DOWNMIX = "pan=mono|c0=0.6*FC+0.2*FL+0.2*FR"


def channel_filter(channel: str, layout: str = "") -> str | None:
    """ffmpeg pan filter for a channel choice: mix (default) | left | right | <index>."""
    channel = (channel or "mix").strip().lower()
    if channel in ("", "mix"):
        return DIALOGUE_DOWNMIX if layout.split("(")[0] in CENTER_LAYOUTS else None
    if channel == "left":
        return "pan=mono|c0=c0"
    if channel == "right":
        return "pan=mono|c0=c1"
    if channel.isdigit():
        return f"pan=mono|c0=c{int(channel)}"
    raise AudioError(f"Invalid channel {channel!r}; use mix, left, right, or a channel index.")


def build_normalize_args(src: str, dst: str, channel: str = "mix", enhance: bool = False,
                         layout: str = "") -> list[str]:
    """ffmpeg argv that decodes `src` (any container) to 16 kHz mono s16 WAV."""
    filters = [f for f in (channel_filter(channel, layout), ENHANCE_FILTER if enhance else None) if f]
    args = [config.FFMPEG_BIN, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", src, "-vn", "-sn", "-dn", "-map", "0:a:0"]
    if filters:
        args += ["-af", ",".join(filters)]
    args += ["-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le", "-f", "wav", dst]
    return args


def probe(path: str) -> AudioInfo:
    cmd = [config.FFPROBE_BIN, "-v", "error", "-show_format", "-show_streams", "-of", "json", path]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is not installed on the STT host.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["unknown"]
        raise UndecodableAudio(f"Could not read this file as audio ({detail[0]}).") from exc
    except subprocess.TimeoutExpired as exc:
        raise UndecodableAudio("Timed out while reading the file header.") from exc
    return parse_probe(json.loads(out.stdout or b"{}"))


def normalize(src: str, dst: str, channel: str = "mix", enhance: bool = False, layout: str = "") -> None:
    try:
        subprocess.run(
            build_normalize_args(src, dst, channel, enhance, layout),
            capture_output=True, check=True, timeout=config.FFMPEG_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed on the STT host.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["unknown"]
        raise UndecodableAudio(f"Could not decode the audio ({detail[0]}).") from exc
    except subprocess.TimeoutExpired as exc:
        raise UndecodableAudio("Decoding took too long (STT_FFMPEG_TIMEOUT_SEC).") from exc


def read_wav(path: str):
    """Canonical WAV → float32 numpy array in [-1, 1]."""
    import numpy as np

    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise AudioError("Internal: expected 16-bit mono WAV after normalization.")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def split_on_silence(
    samples,
    sr: int = TARGET_SR,
    max_chunk_sec: float = 30.0,
    frame_ms: int = 30,
    min_chunk_sec: float = 1.0,
    silence_rel_db: float = -35.0,
) -> list[tuple[int, int]]:
    """Cut audio into chunks no longer than `max_chunk_sec`, at the quietest point.

    Deterministic energy-based segmentation (no model): frame RMS in dB, then for
    each window cut at the lowest-energy frame within the last 40% of the window
    so words are not split mid-syllable. Chunks that are pure silence (all frames
    below the relative threshold) are dropped so engines never see empty input.
    Returns (start_sample, end_sample) pairs.
    """
    import numpy as np

    n = len(samples)
    if n == 0:
        return []
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = int(np.ceil(n / frame))
    padded = np.zeros(n_frames * frame, dtype=np.float32)
    padded[:n] = samples
    rms = np.sqrt(np.mean(padded.reshape(n_frames, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    peak = float(db.max())
    if peak < -60:  # digital silence or near-silence throughout
        return []
    voiced = db > (peak + silence_rel_db)

    max_frames = max(1, int(max_chunk_sec * 1000 / frame_ms))
    min_frames = max(1, int(min_chunk_sec * 1000 / frame_ms))
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < n_frames:
        end = min(start + max_frames, n_frames)
        if end < n_frames:
            search_from = start + max(min_frames, int(max_frames * 0.6))
            if search_from < end:
                end = search_from + int(np.argmin(db[search_from:end]))
        if voiced[start:end].any():
            chunks.append((start * frame, min(end * frame, n)))
        start = max(end, start + 1)
    return chunks
