# semaje-stt

Open-source speech-to-text for any audio, in 1,600+ languages: Whisper + Meta
Omnilingual ASR behind one HTTP contract ([contract.md](contract.md)). Part of
[semaje](https://github.com/semajehq/semaje); runs standalone too.

```bash
docker run -p 8011:8011 -v semaje-models:/models ghcr.io/semajehq/semaje-stt:2.0.0
curl -F audio=@meeting.m4a -F language=sw http://localhost:8011/transcribe
```

Native (needed on Apple Silicon for the Omnilingual engine):

```bash
make setup            # venv + models in ~/.local/share/semaje (outside cloud-synced folders)
make run              # :8011
make test             # gate tests (<5 s, no models; needs ffmpeg)
make eval-fleurs      # real-speech WER/CER per language on FLEURS (service running)
make eval             # English smoke eval (macOS `say`)
```

| file | role |
|---|---|
| `semaje_stt/app.py` | FastAPI surface, auth, request ids, concurrency limits, readiness |
| `semaje_stt/audio.py` | ffprobe/ffmpeg intake, dialogue-aware downmix, enhancement, silence chunking |
| `semaje_stt/router.py` | pure engine routing per language |
| `semaje_stt/engines/` | `faster_whisper_engine`, `whisper_cpp_engine`, `omnilingual_engine` |
| `semaje_stt/models.py` | model registry (licenses, sizes), downloads, active model |
| `semaje_stt/metrics.py` | Prometheus metrics |
| `semaje_stt/evals/` | FLEURS multilingual eval + thresholds + dated baselines |

Languages come from [semaje-languages](https://github.com/semajehq/semaje-languages)
(pinned in `requirements.txt`).

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

## Kikuyu and Dholuo

Measured on Google WAXAL test recordings (8 speakers per language; rows whose
audio does not match the script excluded, including one Kikuyu speaker who
paraphrases instead of reading), identical 40 clips per engine:

| Language | Omnilingual CTC-300M (default) | Omnilingual LLM-300M | MMS-1b-all + adapter (opt-in, non-commercial) |
|---|---|---|---|
| Kikuyu | CER 21.4%, WER 86.1% | CER 23.8%, WER 84.0% | **CER 19.1%, WER 57.1%** |
| Dholuo | CER 10.7%, WER 47.6% | **CER 8.0%, WER 31.9%** | CER 10.0%, WER 37.3% |

Which to run:

- **Dholuo**: the open-licensed LLM card is best. Set
  `STT_OMNILINGUAL_MODEL=omniASR_LLM_300M_v2` (6.5 GB; ~40 s per 10 s clip on a
  laptop CPU in bfloat16, real time on a GPU) and keep `luo` out of
  `STT_MMS_LANGUAGES`.
- **Kikuyu**: the LLM card does not help; MMS-1b-all is clearly better on words
  but is non-commercial (`STT_ALLOW_NONCOMMERCIAL=true`, `STT_MMS_LANGUAGES=kik`).
- Default (CPU, CTC-300M): if you accept the non-commercial license,
  `STT_MMS_LANGUAGES=kik,luo` is best for both.

On a larger 100-clip Dholuo sample the two are even on characters (Omnilingual
10.4%, MMS 11.4% CER) while MMS stays ahead on words (WER 41.0% vs 48.1%).

MMS-1b-all (the ASR C-elo Labs uses for Kikuyu) wins on words because its
language adapter keeps output inside the language's orthography; Omnilingual's
language-agnostic CTC model sometimes writes `ä`/`å` for Kikuyu `ĩ`/`ũ`. Its
weights are CC-BY-NC-4.0, so it is off by default:

```bash
pip install -r requirements-mms.txt
STT_ALLOW_NONCOMMERCIAL=true STT_MMS_LANGUAGES=kik,luo make run
make eval-waxal ARGS="--langs kik,luo --n 40"
```

With the opt-in, requests for `STT_MMS_LANGUAGES` route to MMS (response
`engine: "mms"`, reason in `routing`); everything else is unchanged. MMS is a
1B-parameter model: about 6 s per 10 s clip on a laptop CPU, real time on a GPU.

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
