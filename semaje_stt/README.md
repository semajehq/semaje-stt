# services/stt — speech-to-text

Whisper + Meta Omnilingual ASR behind one HTTP contract ([contract.md](contract.md)).
Any audio or video format in, text with timestamps out, in 1,600+ languages.

```bash
services/stt/setup.sh          # venv + models in ~/.local/share/semaje (outside cloud-synced folders)
npm run dev:stt                # :8011
npm run test:stt               # gate tests (<5 s, no models)
npm run eval:fleurs            # real-speech WER/CER per language on FLEURS (needs the service running)
npm run eval:stt               # English smoke eval (macOS `say`)
```

| file | role |
|---|---|
| `app.py` | FastAPI surface, auth, request ids, concurrency limits, readiness |
| `audio.py` | ffprobe/ffmpeg intake, dialogue-aware downmix, enhancement, silence chunking |
| `router.py` | pure engine routing per language |
| `engines/` | `faster_whisper_engine`, `whisper_cpp_engine`, `omnilingual_engine` |
| `models.py` | model registry (licenses, sizes), downloads, active model |
| `metrics.py` | Prometheus metrics |
| `evals/` | FLEURS multilingual eval + thresholds + dated baselines |

Languages come from the shared registry in [`packages/languages`](../../packages/languages).

## Measured (2026-10-12, Apple M4 CPU, smallest models, FLEURS test, 5 clips)

| language | auto → engine | CER | Whisper-only CER |
|---|---|---|---|
| English | Whisper base | 2.8% | 2.8% |
| Swahili | Omnilingual CTC-300M | 4.9% | 24.3% |
| Amharic | Omnilingual | 11.6% | 148% |
| Yoruba | Omnilingual | 26.1% | 51.5% |
| Luo | Omnilingual | 6.9% | n/a |
| Luganda | Omnilingual | 4.9% | n/a |

All runs faster than real time (RTF 0.09–0.25). Larger models (`omniASR_LLM_1B_v2`,
`large-v3-turbo`) cut error further on GPU.

## Gotchas

- `omnilingual-asr==0.2.0` pins `torch==2.8.0` but not `torchaudio`/`pyarrow`; the
  pins in `requirements-omnilingual.txt` are required or imports fail.
- Omnilingual needs `libsndfile` on the host.
- fairseq2 (Omnilingual's runtime) ships prebuilt wheels for **Linux x86-64** (Meta's
  index `fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/{cpu,cu126,cu128}`) and **macOS
  arm64** (PyPI). Linux arm64 has none: Docker images built on Apple Silicon or ARM
  servers are Whisper-only (the build prints a warning; low-resource languages get a
  clear 400). Use `--platform linux/amd64` or run `setup.sh` natively on a Mac.
- Omnilingual CTC/LLM cards reject inputs ≥40 s; the engine chunks at silences to ≤30 s.
