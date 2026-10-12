"""Gate tests: audio intake. Pure logic always; real ffmpeg format matrix when installed."""
import os
import subprocess
import wave

import numpy as np
import pytest

from services.stt import audio
from services.stt.audio import (
    AudioError, AudioTooLong, NoAudioStream, UndecodableAudio,
    build_normalize_args, channel_filter, parse_probe, split_on_silence,
)
from .conftest import make_wav, needs_ffmpeg

SR = 16000


# ── parse_probe ──────────────────────────────────────────────────
def test_parse_probe_picks_first_audio_stream():
    info = parse_probe({
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "12.5", "bit_rate": "128000"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "44100", "channels": 2},
        ],
    })
    assert info.container == "mov"
    assert info.codec == "aac"
    assert info.sample_rate == 44100
    assert info.channels == 2
    assert info.duration == 12.5
    assert info.bit_rate == 128000
    assert info.has_video is True


def test_parse_probe_duration_falls_back_to_stream_then_format():
    stream_only = parse_probe({"format": {}, "streams": [{"codec_type": "audio", "duration": "3.2"}]})
    format_only = parse_probe({"format": {"duration": "7"}, "streams": [{"codec_type": "audio"}]})
    assert stream_only.duration == 3.2
    assert format_only.duration == 7.0


def test_parse_probe_cover_art_is_not_video():
    info = parse_probe({"format": {"duration": "1"}, "streams": [
        {"codec_type": "audio", "codec_name": "mp3"},
        {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}},
    ]})
    assert info.has_video is False


def test_parse_probe_rejects_files_without_audio():
    with pytest.raises(NoAudioStream) as exc:
        parse_probe({"format": {"duration": "5"}, "streams": [{"codec_type": "video"}]})
    assert exc.value.status == 415


def test_parse_probe_enforces_duration_limit():
    with pytest.raises(AudioTooLong) as exc:
        parse_probe({"format": {"duration": "7201"}, "streams": [{"codec_type": "audio"}]}, max_duration=7200)
    assert exc.value.status == 413


def test_parse_probe_tolerates_garbage_numbers():
    info = parse_probe({"format": {"duration": "N/A"}, "streams": [
        {"codec_type": "audio", "sample_rate": "?", "channels": None, "bit_rate": "N/A"},
    ]})
    assert (info.duration, info.sample_rate, info.channels, info.bit_rate) == (0.0, 0, 0, None)


# ── normalize args ───────────────────────────────────────────────
@pytest.mark.parametrize("choice,expected", [
    ("mix", None), ("", None), ("left", "pan=mono|c0=c0"), ("right", "pan=mono|c0=c1"), ("3", "pan=mono|c0=c3"),
])
def test_channel_filter(choice, expected):
    assert channel_filter(choice) == expected


@pytest.mark.parametrize("layout", ["5.1", "5.1(side)", "7.1", "3.0"])
def test_surround_mix_weights_the_dialogue_channel(layout):
    assert channel_filter("mix", layout) == "pan=mono|c0=0.6*FC+0.2*FL+0.2*FR"


@pytest.mark.parametrize("layout", ["", "mono", "stereo", "quad", "2.1"])
def test_non_center_layouts_use_plain_downmix(layout):
    assert channel_filter("mix", layout) is None


def test_channel_filter_rejects_junk():
    with pytest.raises(AudioError):
        channel_filter("center; rm -rf /")


def test_normalize_args_produce_16k_mono_pcm_and_drop_video():
    args = build_normalize_args("in.mp4", "out.wav")
    assert args[args.index("-ar") + 1] == "16000"
    assert args[args.index("-ac") + 1] == "1"
    assert args[args.index("-c:a") + 1] == "pcm_s16le"
    assert "-vn" in args and args[-1] == "out.wav"
    assert "-af" not in args


def test_normalize_args_chain_channel_and_enhance_filters():
    args = build_normalize_args("in.wav", "out.wav", channel="left", enhance=True)
    chain = args[args.index("-af") + 1]
    assert chain.startswith("pan=mono|c0=c0,")
    assert "afftdn" in chain and "dynaudnorm" in chain


# ── split_on_silence ─────────────────────────────────────────────
def _tone(seconds, gaps=()):
    t = np.arange(int(seconds * SR)) / SR
    x = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for a, b in gaps:
        x[int(a * SR):int(b * SR)] = 0
    return x


def test_split_empty_and_silent_audio_yields_no_chunks():
    assert split_on_silence(np.zeros(0, dtype=np.float32)) == []
    assert split_on_silence(np.zeros(SR * 5, dtype=np.float32)) == []


def test_split_short_audio_is_one_chunk():
    chunks = split_on_silence(_tone(4), max_chunk_sec=30)
    assert chunks == [(0, 4 * SR)]


def test_split_long_audio_respects_max_and_cuts_in_silence():
    # 95 s of tone with a 1 s pause every 20 s.
    gaps = [(s, s + 1) for s in range(20, 95, 20)]
    x = _tone(95, gaps)
    chunks = split_on_silence(x, max_chunk_sec=30)
    assert len(chunks) >= 4
    for start, end in chunks:
        assert (end - start) / SR <= 30.0 + 1e-6
    # Every interior cut lands inside a pause, not mid-tone.
    for _, end in chunks[:-1]:
        assert any(a * SR <= end <= b * SR for a, b in gaps), f"cut at {end / SR:.2f}s is not in a pause"
    # Chunks are ordered, non-overlapping, and cover the end of the audio.
    assert all(a[1] <= b[0] for a, b in zip(chunks, chunks[1:]))
    assert chunks[-1][1] == len(x)


def test_split_without_pauses_still_bounds_chunk_length():
    chunks = split_on_silence(_tone(100), max_chunk_sec=30)
    assert all((e - s) / SR <= 30 for s, e in chunks)
    assert sum(e - s for s, e in chunks) == 100 * SR


def test_split_drops_silent_stretches_between_speech():
    x = np.concatenate([_tone(5), np.zeros(SR * 40, dtype=np.float32), _tone(5)])
    chunks = split_on_silence(x, max_chunk_sec=30)
    voiced = sum(e - s for s, e in chunks)
    assert voiced < len(x)  # the 40 s of silence is not all sent to the engine
    assert chunks[0][0] == 0 and chunks[-1][1] == len(x)


# ── Real ffmpeg: every container/codec decodes to canonical WAV ──
FORMATS = [
    ("mp3", ["-c:a", "libmp3lame"]),
    ("m4a", ["-c:a", "aac"]),
    ("aac", ["-c:a", "aac", "-f", "adts"]),
    ("ogg", ["-c:a", "libvorbis"]),
    ("oga", ["-c:a", "vorbis", "-strict", "-2", "-ac", "2"]),  # ffmpeg's native Vorbis encoder
    ("opus", ["-c:a", "libopus"]),
    ("webm", ["-c:a", "libopus"]),
    ("flac", ["-c:a", "flac"]),
    ("wav", ["-c:a", "pcm_s24le", "-ar", "96000"]),        # studio 24-bit / 96 kHz
    ("wav", ["-c:a", "pcm_mulaw", "-ar", "8000"]),          # telephony mu-law 8 kHz
    ("aiff", ["-c:a", "pcm_s16be"]),
    ("wma", ["-c:a", "wmav2"]),                               # Windows Media Audio
    ("caf", ["-c:a", "alac"]),                               # Apple lossless
    ("mka", ["-c:a", "ac3", "-ac", "6"]),                    # 5.1 surround
    ("amr", ["-c:a", "libopencore_amrnb", "-ar", "8000", "-ac", "1"]),  # phone voice notes
    ("mp4", "video"),                                        # video container with audio
    ("mkv", "video"),
]


def _encoder_available(args) -> bool:
    if args == "video":
        return True
    codec = args[args.index("-c:a") + 1]
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return f" {codec} " in out


@needs_ffmpeg
@pytest.mark.parametrize("ext,args", FORMATS, ids=[f"{e}-{i}" for i, (e, _) in enumerate(FORMATS)])
def test_any_format_normalizes_to_16k_mono(tmp_path, ext, args):
    if not _encoder_available(args):
        pytest.skip(f"this ffmpeg build lacks the encoder for {ext}")
    src = tmp_path / f"src.{ext}"
    if args == "video":
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10",
               "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "2",
               "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(src)]
    else:
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
               "sine=frequency=440:sample_rate=48000", "-t", "2", *args, str(src)]
    subprocess.run(cmd, check=True)

    info = audio.probe(str(src))
    assert 1.5 <= info.duration <= 2.5
    if args == "video":
        assert info.has_video

    dst = tmp_path / "out.wav"
    audio.normalize(str(src), str(dst), layout=info.channel_layout)
    with wave.open(str(dst), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert 1.5 <= wf.getnframes() / 16000 <= 2.5
    samples = audio.read_wav(str(dst))
    assert float(np.abs(samples).max()) > 0.05  # real signal survived decoding


@needs_ffmpeg
def test_stereo_channel_selection(tmp_path):
    # Left = tone, right = silence. "right" must come out silent, "left" loud.
    src = tmp_path / "lr.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=440:d=1",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1",
                    "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo", str(src)], check=True)
    left, right = tmp_path / "l.wav", tmp_path / "r.wav"
    audio.normalize(str(src), str(left), channel="left")
    audio.normalize(str(src), str(right), channel="right")
    assert np.abs(audio.read_wav(str(left))).max() > 0.1
    assert np.abs(audio.read_wav(str(right))).max() < 0.01


@needs_ffmpeg
def test_garbage_bytes_are_undecodable(tmp_path):
    bad = tmp_path / "notes.mp3"
    bad.write_bytes(os.urandom(4096))
    with pytest.raises(UndecodableAudio) as exc:
        audio.probe(str(bad))
    assert exc.value.status == 415


@needs_ffmpeg
def test_text_file_is_rejected(tmp_path):
    txt = tmp_path / "readme.txt"
    txt.write_text("hello, this is not audio\n" * 50)
    with pytest.raises(AudioError):
        audio.probe(str(txt))


@needs_ffmpeg
def test_wav_helper_roundtrip(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(make_wav(0.5, sr=22050, channels=2))
    dst = tmp_path / "b.wav"
    audio.normalize(str(src), str(dst))
    assert len(audio.read_wav(str(dst))) == 8000
