# Whisper STT Service

Self-contained speech-to-text service for Voxail. The only component that loads a
Whisper model — everything else calls its HTTP contract (`contract.md`).

Two interchangeable backends, same response shape:

| Backend | What it is | Notes |
|---------|-----------|-------|
| `faster-whisper` (default) | CTranslate2, pure Python | No compile step; CPU/GPU |
| `whisper.cpp` | Compiled C++ binary | Metal on Apple Silicon; needs cmake to build |

## Setup

```bash
./setup.sh                        # faster-whisper only
BUILD_WHISPER_CPP=1 ./setup.sh    # also build whisper.cpp
```

## Run

```bash
./run.sh                          # serves on WHISPER_PORT (default 8011)
WHISPER_BACKEND=whisper.cpp ./run.sh
```

## Configuration (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `WHISPER_BACKEND` | `faster-whisper` | `faster-whisper` or `whisper.cpp` |
| `WHISPER_MODEL` | `base` | `tiny`\|`base`\|`small`\|`medium`\|`large-v3` |
| `WHISPER_DEVICE` | `cpu` | faster-whisper device (`cpu`\|`cuda`\|`auto`) |
| `WHISPER_COMPUTE_TYPE` | `int8` | faster-whisper precision |
| `WHISPER_CPP_BIN` | vendored | path to `whisper-cli` |
| `WHISPER_CPP_MODEL` | vendored | path to `ggml-<model>.bin` |
| `WHISPER_PORT` | `8011` | listen port |

## API

`POST /transcribe` (multipart: `audio`, optional `language`, `task`) → transcript
JSON with segment timestamps. `GET /health`. Full spec in `contract.md`.

## Tests & evals

```bash
.venv/bin/python -m pytest tests -q                       # gate: fast, no model
.venv/bin/python -m services.whisper.evals.eval_transcription   # real accuracy (WER)
```

Run the eval from the project root so the `services.whisper` package resolves.
