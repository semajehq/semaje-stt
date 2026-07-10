# Whisper STT Service — Contract

Self-contained speech-to-text service. The only thing that talks to a Whisper
model in semaje. Every other component (the Node server, the live transcriber)
calls this HTTP contract — never a backend library directly.

Base URL: `WHISPER_URL` (default `http://localhost:8011`).

## `GET /health`

```json
{
  "ok": true,
  "ready": true,
  "backend": "faster-whisper",
  "model": "base",
  "device": "cpu"
}
```

`ready` is `false` until the model finishes loading on first request / warmup.

## `POST /transcribe`

`multipart/form-data`:

| field      | type   | required | notes                                                        |
|------------|--------|----------|--------------------------------------------------------------|
| `audio`    | file   | yes      | any ffmpeg-decodable audio/video container                   |
| `language` | string | no       | ISO code (`en`, `fr`, …). Omit to auto-detect.               |
| `task`     | string | no       | `transcribe` (default) or `translate` (→ English, Whisper)   |
| `diarize`  | bool   | no       | run the optional local speaker-diarization pipeline          |

Response `200`:

```json
{
  "text": "full transcript as one string",
  "language": "en",
  "duration": 12.34,
  "segments": [
    { "start": 0.0, "end": 4.2, "text": "first chunk" },
    { "start": 4.2, "end": 8.1, "text": "second chunk" }
  ],
  "backend": "faster-whisper",
  "model": "base"
}
```

Empty/silent audio returns `text: ""` and `segments: []` (not an error).

Errors return `4xx/5xx` with `{ "error": "message" }`.

## Backends

Selected by `WHISPER_BACKEND`:

- `faster-whisper` (default) — CTranslate2, pure Python, CPU/GPU.
- `whisper.cpp` — shells out to the compiled `whisper-cli` binary (Metal on Apple Silicon).

Both return the identical response shape. Callers never know or care which ran.

Segments may include `speaker`, confidence in the `0..1` range, and word-level
`{start,end,word,probability}` entries. With `diarize=true`, pyannote runs
locally when installed; otherwise the response explicitly uses the
`single-speaker-fallback` backend.
