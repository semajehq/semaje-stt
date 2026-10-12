# STT Service — Contract (v2)

Open-source speech-to-text for any audio format in 1,600+ languages. The only
component that runs a speech-recognition model. Everything else calls this HTTP
contract through `services/stt/client/index.ts`.

Base URL: `STT_URL` (default `http://localhost:8011`). Optional auth: when
`STT_API_TOKEN` is set, every route except `/health` and `/health/live` requires
`Authorization: Bearer <token>`. Every response carries `x-request-id` (echoed
if the caller sent one).

## Engines

| Family | Runtime | License | Languages | Notes |
|---|---|---|---|---|
| `whisper` | faster-whisper (default) or whisper.cpp | MIT | 99 | word timestamps, confidence, translate→English |
| `omnilingual` | Meta Omnilingual ASR (fairseq2) | Apache-2.0 | 1,672 | low-resource coverage; segment-level timestamps |

**Routing** (`router.route`, pure): with `engine=auto` a language Whisper handles
well (`STT_WHISPER_STRONG`, default ≤15% FLEURS WER list) goes to Whisper;
everything else goes to Omnilingual. With no language, Whisper's language ID
decides (detections below p=0.6 go to Omnilingual). `translate` always runs on
Whisper. A missing engine degrades to the other where it can; otherwise 400.

## `POST /transcribe` (multipart)

| field | required | notes |
|---|---|---|
| `audio` | yes | any ffmpeg-decodable audio/video: MP3, AAC/M4A, Opus/WebM, Ogg, FLAC, ALAC, WAV (8–96 kHz, 16/24/32-bit, µ-law), AIFF, AMR, WMA, MP4/MOV/MKV/AVI |
| `language` | no | any registry form: `sw`, `swa`, `swh`, `swh_Latn`, `sw-KE`, `Swahili`. Omit/`auto` to detect |
| `task` | no | `transcribe` (default) or `translate` |
| `engine` | no | `auto` (default `STT_ROUTING`) · `whisper` · `omnilingual` |
| `diarize` | no | local pyannote when installed, else `single-speaker-fallback` |
| `channel` | no | `mix` (default; dialogue-weighted for 5.1/7.1) · `left` · `right` · channel index |
| `enhance` | no | high-pass + FFT denoise + dynamic loudness for noisy/quiet recordings |

Response `200`:

```json
{
  "text": "habari ya asubuhi",
  "language": "sw",
  "languageCode": "swh_Latn",
  "languageName": "Swahili",
  "engineLanguage": "swh_Latn",
  "duration": 9.12,
  "segments": [{ "start": 0.0, "end": 9.12, "text": "habari ya asubuhi" }],
  "engine": "omnilingual",
  "backend": "omnilingual",
  "model": "omniASR_CTC_300M_v2",
  "routing": { "engine": "omnilingual", "language": "swh_Latn", "canonical": "swh_Latn", "reason": "Swahili is better served by Omnilingual" },
  "audio": { "container": "mp3", "codec": "mp3", "sampleRate": 44100, "channels": 2, "duration": 9.12, "bitRate": 128000, "hasVideo": false, "channelLayout": "stereo" },
  "timings": { "decodeMs": 41.0, "transcribeMs": 812.3 },
  "requestId": "…"
}
```

`language` keeps v1 semantics (Whisper code when one exists, else the canonical
code). Whisper segments additionally carry `confidence` (0..1) and `words`
`[{start,end,word,probability}]`. Silent or zero-length audio returns
`text: ""`, `segments: []` with 200.

Errors: `{ "error", "code", "requestId" }`.

| status | code | meaning |
|---|---|---|
| 400 | `invalid_task` · `invalid_engine` · `unknown_language` | fix the request |
| 401 | `unauthorized` | bad/missing bearer token |
| 413 | `upload_too_large` (`STT_MAX_UPLOAD_MB`) · `audio_too_long` (`STT_MAX_DURATION_SEC`) | |
| 415 | `no_audio_stream` · `undecodable_audio` | not audio, corrupt, or DRM |
| 503 | `busy` + `Retry-After` | `STT_MAX_CONCURRENCY` slots full for `STT_QUEUE_TIMEOUT_SEC` |
| 500 | `internal` | engine failure (logged with requestId) |

## Other routes

- `GET /languages?q=&engine=` — registry rows with `engines` (installed engines that cover it), `whisperStrong`, `translate`.
- `GET /models` — Whisper + Omnilingual model registry with license, size, install state, hardware recommendation; `activeByFamily`.
- `POST /models/download` · `POST /models/activate` `{backend, model}` · `DELETE /models/{backend}/{model}`.
- `GET /health` (always 200, informational) · `GET /health/ready` (503 until a model is loaded) · `GET /health/live`.
- `GET /metrics` — Prometheus: requests, latency, audio seconds, real-time factor, failures by code, capacity rejections.

## Guarantees

- Uploads stream to disk (never fully in RAM) and are deleted after the request.
- Transcription runs off the event loop; health checks stay responsive under load.
- `STT_WARMUP=true` (default) loads every engine at startup so no request waits on a model download.
