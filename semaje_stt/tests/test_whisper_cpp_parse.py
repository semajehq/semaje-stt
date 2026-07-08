"""Gate test: whisper.cpp JSON parser. Pure, no binary, <1ms."""
from services.whisper.backends.whisper_cpp_backend import parse_whisper_cpp_json


def test_parses_segments_text_and_timestamps():
    data = {
        "result": {"language": "en"},
        "transcription": [
            {"offsets": {"from": 0, "to": 4200}, "text": " Hello there"},
            {"offsets": {"from": 4200, "to": 8100}, "text": " general kenobi"},
        ],
    }
    r = parse_whisper_cpp_json(data, "ggml-base.bin")
    assert r.text == "Hello there general kenobi"
    assert r.language == "en"
    assert r.model == "ggml-base.bin"
    assert r.backend == "whisper.cpp"
    assert len(r.segments) == 2
    assert r.segments[0].start == 0.0
    assert r.segments[0].end == 4.2
    assert r.duration == 8.1


def test_skips_empty_segments_and_handles_no_speech():
    data = {"result": {"language": "en"}, "transcription": [{"offsets": {"from": 0, "to": 0}, "text": "   "}]}
    r = parse_whisper_cpp_json(data, "ggml-base.bin")
    assert r.text == ""
    assert r.segments == []


def test_to_dict_shape_matches_contract():
    data = {"result": {"language": "fr"}, "transcription": [{"offsets": {"from": 100, "to": 900}, "text": "bonjour"}]}
    d = parse_whisper_cpp_json(data, "ggml-base.bin").to_dict()
    assert set(d.keys()) == {"text", "language", "duration", "segments", "backend", "model"}
    assert d["segments"][0] == {"start": 0.1, "end": 0.9, "text": "bonjour"}
