"""Periodic eval: ASR accuracy on Google WAXAL (CC-BY-4.0 / CC-BY-SA-4.0) test splits.

WAXAL's TTS portion has studio recordings with transcripts for languages FLEURS
lacks (Kikuyu) or covers thinly (Dholuo). Each language has 8 speakers, so the
test split doubles as a speaker-diverse ASR benchmark. Audio is sent as the
original MP3, which also exercises the any-format intake.

Backends:
  service — the RUNNING semaje-stt (whatever engine/model it routes to)
  mms     — Meta MMS-1b-all + language adapter, run locally (CC-BY-NC-4.0;
            the baseline C-elo Labs used for Kikuyu). Needs torch + transformers.

    make eval-waxal ARGS="--langs kik,luo --n 100"
    ... -m semaje_stt.evals.eval_waxal --backend mms --langs kik,luo --n 100

Writes data/observability/stt-waxal-eval.json and fails past waxal_thresholds.json.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from semaje_stt.evals.eval_fleurs import cer, wer

HERE = Path(__file__).resolve().parent
CACHE = Path(os.getenv("WAXAL_CACHE", str(Path.home() / ".cache" / "semaje" / "waxal")))
URL = "https://huggingface.co/datasets/google/WaxalNLP/resolve/main/data/TTS/{lang}/{lang}-test-00000.parquet"
# WAXAL code → what we send to semaje / the MMS adapter id.
LANGS = {"kik": "kik", "luo": "luo", "swa": "sw", "lug": "lg", "ibo": "ig", "hau": "ha", "yor": "yo",
         "aka": "aka", "ewe": "ewe", "wol": "wo", "ful": "ful", "ach": "ach", "nyn": "nyn"}


# WAXAL contains misaligned rows: e.g. a 167 s recording paired with a 330-char
# transcript (2 chars/s) or text far longer than the audio (56 chars/s). Read
# speech runs ~6-16 chars/s. Rows outside these bounds are label errors, not
# model errors, so --clean (default) drops them and the report says how many.
MIN_CHARS_PER_SEC, MAX_CHARS_PER_SEC, MAX_SECONDS = 4.0, 20.0, 40.0


def audio_seconds(data: bytes) -> float:
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(data)
        f.flush()
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", f.name],
                             capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def is_aligned(text: str, seconds: float) -> bool:
    if seconds <= 0 or seconds > MAX_SECONDS:
        return False
    rate = len(" ".join(text.split())) / seconds
    return MIN_CHARS_PER_SEC <= rate <= MAX_CHARS_PER_SEC


# Speakers whose recordings are not verbatim reads of the script. Found by
# inspection: Kikuyu "Female 1" describes the prompt pictures in her own words,
# and every engine (Omnilingual, MMS-1b-all) scores ~90-100% CER on her clips.
NON_VERBATIM = {"kik": {"Female 1"}}


def load_clips(lang: str, n: int, clean: bool = True) -> tuple[list[dict], int]:
    """n clips spread evenly across the test split (all 8 speakers represented).
    Returns (clips, number of misaligned rows dropped)."""
    import pyarrow.parquet as pq

    path = CACHE / f"{lang}-test.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(lang=lang), tmp)
        tmp.replace(path)
    rows = sorted(pq.read_table(path).to_pylist(), key=lambda r: r["id"])
    dropped = 0
    if clean:
        kept = [r for r in rows if r["speaker_id"] not in NON_VERBATIM.get(lang, set())
                and is_aligned(r["text"], audio_seconds(r["audio"]["bytes"]))]
        dropped, rows = len(rows) - len(kept), kept
    step = max(1, len(rows) // n) if n else 1
    picked = rows[::step][:n] if n else rows
    return [{"id": r["id"], "speaker": r["speaker_id"], "text": r["text"], "audio": r["audio"]["bytes"]} for r in picked], dropped


def via_service(url: str, token: str, engine: str):
    def run(clip: dict, lang: str) -> str:
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        for k, v in {"language": LANGS[lang], "engine": engine}.items():
            body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"{clip['id']}.mp3\"\r\n"
                   f"Content-Type: audio/mpeg\r\n\r\n".encode())
        body.write(clip["audio"])
        body.write(f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(f"{url}/transcribe", data=body.getvalue(), method="POST", headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            **({"Authorization": f"Bearer {token}"} if token else {})})
        with urllib.request.urlopen(req, timeout=3600) as res:
            out = json.loads(res.read())
        run.served = f"{out.get('engine')}:{out.get('model')}"
        return out.get("text", "")
    run.served = "?"
    return run


def via_mms():
    """MMS-1b-all with the per-language adapter (CTC, greedy), as C-elo Labs used it."""
    import tempfile
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from semaje_stt.audio import normalize, read_wav

    processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").eval()
    current = {"lang": None}

    def run(clip: dict, lang: str) -> str:
        if current["lang"] != lang:
            processor.tokenizer.set_target_lang(lang)
            model.load_adapter(lang)
            current["lang"] = lang
        with tempfile.TemporaryDirectory() as tmp:
            src, wav = Path(tmp) / "in.mp3", Path(tmp) / "out.wav"
            src.write_bytes(clip["audio"])
            normalize(str(src), str(wav))
            samples = read_wav(str(wav))
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = torch.argmax(model(**inputs).logits, dim=-1)[0]
        return processor.decode(ids)
    run.served = "mms:mms-1b-all"
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="kik,luo")
    ap.add_argument("--n", type=int, default=int(os.getenv("WAXAL_N", "50")))
    ap.add_argument("--backend", choices=["service", "mms"], default="service")
    ap.add_argument("--engine", default="auto", help="service backend: auto | whisper | omnilingual")
    ap.add_argument("--label", default="", help="name for this run in the report (e.g. omni-ctc-300m)")
    ap.add_argument("--no-clean", dest="clean", action="store_false", help="keep misaligned rows")
    ap.add_argument("--report", default=os.getenv("EVAL_REPORT_PATH", "data/observability/stt-waxal-eval.json"))
    args = ap.parse_args(argv)

    run = via_mms() if args.backend == "mms" else via_service(
        os.getenv("STT_URL", "http://localhost:8011").rstrip("/"), os.getenv("STT_API_TOKEN", ""), args.engine)
    label = args.label or (args.backend if args.backend == "mms" else f"service-{args.engine}")
    thresholds = {k: v for k, v in json.loads((HERE / "waxal_thresholds.json").read_text(encoding="utf-8")).items()
                  if not k.startswith("_")}
    rows, failures = [], []
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        clips, dropped = load_clips(lang, args.n, clean=args.clean)
        werr, cerr, per_speaker, secs = [], [], {}, 0.0
        for clip in clips:
            t0 = time.perf_counter()
            hyp = run(clip, lang)
            secs += time.perf_counter() - t0
            c = cer(clip["text"], hyp)
            werr.append(wer(clip["text"], hyp))
            cerr.append(c)
            per_speaker.setdefault(clip["speaker"], []).append(c)
        row = {
            "lang": lang, "run": label, "servedBy": run.served, "clips": len(clips), "misalignedDropped": dropped,
            "wer": round(sum(werr) / len(werr), 4), "cer": round(sum(cerr) / len(cerr), 4),
            "worstSpeakerCer": round(max(sum(v) / len(v) for v in per_speaker.values()), 4),
            "secondsPerClip": round(secs / len(clips), 2),
        }
        limit = thresholds.get(lang, {}).get(label)
        row["thresholdCer"], row["passed"] = limit, limit is None or row["cer"] <= limit
        if not row["passed"]:
            failures.append(row)
        rows.append(row)
        print(f"[{'PASS' if row['passed'] else 'FAIL'}] {lang} {label:18} served={row['servedBy']:32} "
              f"CER={row['cer']:.3f} WER={row['wer']:.3f} worst-speaker CER={row['worstSpeakerCer']:.3f} "
              f"{row['secondsPerClip']}s/clip, {dropped} misaligned/non-verbatim rows dropped (limit {limit})", flush=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"dataset": "google/WaxalNLP TTS test", "results": rows,
                                             "passed": not failures}, indent=2), encoding="utf-8")
    print("EVAL PASS" if not failures else f"EVAL FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
