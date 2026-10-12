"""Periodic eval: real-speech accuracy per language on Google FLEURS (CC-BY-4.0).

Streams the first N test clips per language straight out of FLEURS' tar.gz
(no full download), sends them to the RUNNING STT service as a black box, and
scores WER and CER against FLEURS' normalized transcriptions. Thresholds live
in fleurs_thresholds.json; the run fails if any language regresses past them.

    services/stt/.venv/bin/python -m services.stt.evals.eval_fleurs                  # default set
    ... -m services.stt.evals.eval_fleurs --langs sw_ke,yo_ng --n 10 --engines auto,whisper
    STT_URL=http://gpu-box:8011 ... -m services.stt.evals.eval_fleurs

Writes data/observability/stt-fleurs-eval.json (published by the Signal Room).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tarfile
import time
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path(os.getenv("FLEURS_CACHE", str(Path.home() / ".cache" / "semaje" / "fleurs")))
BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{cfg}"

# FLEURS config → language we send to semaje. A mix of Whisper-strong and
# low-resource languages so both routing paths are always exercised.
FLEURS = {
    "en_us": "en", "fr_fr": "fr", "sw_ke": "sw", "yo_ng": "yo", "am_et": "am", "ha_ng": "ha",
    "lg_ug": "lg", "luo_ke": "luo", "zu_za": "zu", "xh_za": "xh", "wo_sn": "wo", "ig_ng": "ig",
    "so_so": "so", "ny_mw": "ny", "sn_zw": "sn", "om_et": "om", "kam_ke": "kam", "ln_cd": "ln",
}
DEFAULT_LANGS = ["en_us", "sw_ke", "yo_ng", "am_et", "luo_ke", "lg_ug"]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref).split(), normalize(hyp).split()
    return edit_distance(r, h) / max(1, len(r))


def cer(ref: str, hyp: str) -> float:
    r, h = list(normalize(ref).replace(" ", "")), list(normalize(hyp).replace(" ", ""))
    return edit_distance(r, h) / max(1, len(r))


def fetch_clips(cfg: str, n: int) -> list[tuple[Path, str]]:
    """First n FLEURS test clips for a config, cached locally. Streams only what it needs."""
    out_dir = CACHE / cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "test.tsv"
    if not tsv.exists():
        urllib.request.urlretrieve(BASE.format(cfg=cfg) + "/test.tsv", tsv)
    refs = {}
    with open(tsv, encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 4:
                refs[row[1]] = row[3]
    have = sorted(p for p in out_dir.glob("*.wav"))
    if len(have) < n:
        with urllib.request.urlopen(BASE.format(cfg=cfg) + "/audio/test.tar.gz", timeout=120) as res:
            with tarfile.open(fileobj=res, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".wav"):
                        continue
                    name = Path(member.name).name
                    if name in refs and not (out_dir / name).exists():
                        (out_dir / name).write_bytes(tar.extractfile(member).read())
                    if len(list(out_dir.glob("*.wav"))) >= n:
                        break
        have = sorted(out_dir.glob("*.wav"))
    return [(p, refs[p.name]) for p in have[:n] if p.name in refs]


def transcribe(url: str, path: Path, language: str, engine: str, token: str) -> dict:
    boundary = "semajeEvalBoundary"
    fields = {"language": language, "engine": engine}
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"{path.name}\"\r\n"
               f"Content-Type: audio/wav\r\n\r\n".encode())
    body.write(path.read_bytes())
    body.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"{url}/transcribe", data=body.getvalue(), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=3600) as res:
        return json.loads(res.read())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(DEFAULT_LANGS))
    ap.add_argument("--n", type=int, default=int(os.getenv("FLEURS_N", "5")))
    ap.add_argument("--engines", default="auto")
    ap.add_argument("--report", default=os.getenv("EVAL_REPORT_PATH", "data/observability/stt-fleurs-eval.json"))
    args = ap.parse_args(argv)

    url = os.getenv("STT_URL", os.getenv("WHISPER_URL", "http://localhost:8011")).rstrip("/")
    token = os.getenv("STT_API_TOKEN", "")
    thresholds = {k: v for k, v in json.loads((HERE / "fleurs_thresholds.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
    rows, failures = [], []

    for cfg in [c.strip() for c in args.langs.split(",") if c.strip()]:
        if cfg not in FLEURS:
            print(f"skip {cfg}: not in FLEURS map", file=sys.stderr)
            continue
        clips = fetch_clips(cfg, args.n)
        for engine in [e.strip() for e in args.engines.split(",")]:
            werr, cerr, secs, audio = [], [], 0.0, 0.0
            used = set()
            for path, ref in clips:
                t0 = time.perf_counter()
                out = transcribe(url, path, FLEURS[cfg], engine, token)
                secs += time.perf_counter() - t0
                audio += float(out.get("duration") or 0)
                used.add(f"{out.get('engine')}:{out.get('model')}")
                werr.append(wer(ref, out.get("text", "")))
                cerr.append(cer(ref, out.get("text", "")))
            row = {
                "fleurs": cfg, "language": FLEURS[cfg], "engine": engine, "served_by": sorted(used),
                "clips": len(clips), "wer": round(sum(werr) / max(1, len(werr)), 4),
                "cer": round(sum(cerr) / max(1, len(cerr)), 4),
                "rtf": round(secs / max(audio, 1e-6), 3),
            }
            limit = thresholds.get(cfg, {}).get(engine)
            row["threshold_cer"] = limit
            row["passed"] = limit is None or row["cer"] <= limit
            if not row["passed"]:
                failures.append(row)
            rows.append(row)
            print(f"[{'PASS' if row['passed'] else 'FAIL'}] {cfg:7} engine={engine:11} served={','.join(row['served_by']):34} "
                  f"WER={row['wer']:.3f} CER={row['cer']:.3f} RTF={row['rtf']:.2f} (limit CER {limit})")

    # Routing check: wherever both policies ran, auto must not lose to forced Whisper.
    by_lang = {}
    for row in rows:
        by_lang.setdefault(row["fleurs"], {})[row["engine"]] = row
    for cfg, runs in by_lang.items():
        if "auto" in runs and "whisper" in runs and runs["auto"]["cer"] > runs["whisper"]["cer"] + 0.01:
            failures.append({"fleurs": cfg, "check": "routing", "auto": runs["auto"]["cer"], "whisper": runs["whisper"]["cer"]})
            print(f"[FAIL] routing {cfg}: auto CER {runs['auto']['cer']:.3f} > whisper CER {runs['whisper']['cer']:.3f}")

    report = {"dataset": "google/fleurs test", "clipsPerLanguage": args.n, "results": rows, "passed": not failures}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport → {args.report}")
    print("EVAL PASS" if not failures else f"EVAL FAIL ({len(failures)} regressions)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
