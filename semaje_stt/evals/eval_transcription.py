"""Periodic eval: real Whisper transcription accuracy.

Generates known speech with macOS `say`, transcribes it through the CONFIGURED
backend, and asserts word error rate stays under threshold. Paid/slow (loads the
model) — run before ship and nightly, not on every commit.

Usage:
    services/whisper/.venv/bin/python -m services.whisper.evals.eval_transcription
    WHISPER_BACKEND=whisper.cpp services/whisper/.venv/bin/python -m services.whisper.evals.eval_transcription
"""
import os
import json
import subprocess
import sys
import tempfile

from services.whisper.backends import get_backend

WER_THRESHOLD = float(os.getenv("EVAL_WER_THRESHOLD", "0.20"))

CASES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming how we work and live.",
    "Please transcribe this audio file accurately and completely.",
]


def _normalize(s: str) -> list[str]:
    keep = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in s)
    return keep.split()


def wer(ref: str, hyp: str) -> float:
    r, h = _normalize(ref), _normalize(hyp)
    # Levenshtein distance over word lists
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = prev if r[i - 1] == h[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[len(h)] / max(len(r), 1)


def synth(text: str, out_wav: str) -> None:
    aiff = out_wav + ".aiff"
    subprocess.run(["say", "-o", aiff, text], check=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", aiff, "-ar", "16000", "-ac", "1", out_wav],
        check=True, capture_output=True,
    )
    os.unlink(aiff)


def main() -> int:
    backend = get_backend()
    print(f"Backend: {backend.name}  model: {backend.model}")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, ref in enumerate(CASES):
            wav = os.path.join(tmp, f"case{i}.wav")
            synth(ref, wav)
            out = backend.transcribe(wav, "en", "transcribe")
            score = wer(ref, out.text)
            results.append(score)
            flag = "PASS" if score <= WER_THRESHOLD else "FAIL"
            print(f"[{flag}] WER={score:.2f}  ref={ref!r}  hyp={out.text!r}")

    avg = sum(results) / len(results)
    worst = max(results)
    print(f"\navg WER={avg:.3f}  worst={worst:.3f}  threshold={WER_THRESHOLD}")
    ok = worst <= WER_THRESHOLD
    report_path = os.getenv("EVAL_REPORT_PATH", "data/observability/whisper-eval.json")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report:
        json.dump({
            "backend": backend.name, "model": backend.model, "average": avg,
            "worst": worst, "threshold": WER_THRESHOLD, "passed": ok,
            "cases": len(results),
        }, report, indent=2)
    print("EVAL PASS" if ok else "EVAL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
