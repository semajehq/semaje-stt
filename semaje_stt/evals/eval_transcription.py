"""Periodic eval: English accuracy smoke test with synthetic speech (macOS `say`).

Fast sanity check for the Whisper engine on any Mac; the real multilingual
eval is eval_fleurs.py. Writes data/observability/stt-eval.json.

    npm run eval:stt
"""
import json
import os
import subprocess
import sys
import tempfile

from services.stt import config
from services.stt.audio import normalize, read_wav
from services.stt.engines import AudioInput, get_engine
from services.stt.evals.eval_fleurs import wer

WER_THRESHOLD = float(os.getenv("EVAL_WER_THRESHOLD", "0.20"))
CASES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming how we work and live.",
    "Please transcribe this audio file accurately and completely.",
]


def synth(text: str, out_wav: str) -> None:
    aiff = out_wav + ".aiff"
    subprocess.run(["say", "-o", aiff, text], check=True)
    normalize(aiff, out_wav)
    os.unlink(aiff)


def main() -> int:
    engine = get_engine(config.BACKEND, config.MODEL)
    print(f"Engine: {engine.name}  model: {engine.model}")
    scores = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, ref in enumerate(CASES):
            wav = os.path.join(tmp, f"case{i}.wav")
            synth(ref, wav)
            out = engine.transcribe(AudioInput(wav, read_wav(wav)), "en", "transcribe")
            score = wer(ref, out.text)
            scores.append(score)
            print(f"[{'PASS' if score <= WER_THRESHOLD else 'FAIL'}] WER={score:.2f}  ref={ref!r}  hyp={out.text!r}")
    avg, worst = sum(scores) / len(scores), max(scores)
    ok = worst <= WER_THRESHOLD
    report_path = os.getenv("EVAL_REPORT_PATH", "data/observability/stt-eval.json")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"backend": engine.name, "model": engine.model, "average": avg, "worst": worst,
                   "threshold": WER_THRESHOLD, "passed": ok, "cases": len(scores)}, fh, indent=2)
    print(f"\navg WER={avg:.3f}  worst={worst:.3f}  threshold={WER_THRESHOLD}\n{'EVAL PASS' if ok else 'EVAL FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
