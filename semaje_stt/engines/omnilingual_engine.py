"""Omnilingual ASR engine — Meta's 1600+ language recognizer (Apache-2.0).

CTC and LLM model cards only accept inputs shorter than 40 s, so long audio is
cut at silences into <= STT_OMNILINGUAL_MAX_CHUNK_SEC pieces, transcribed in
batches, and stitched back with chunk-level timestamps. The model emits no word
timings or confidences, so segments carry neither.
"""
from __future__ import annotations

import os

from .base import Engine, TranscriptResult, Segment, AudioInput
from .. import config
from ..audio import split_on_silence
from ..models import MODEL_ROOT, omnilingual_marker


def assemble_segments(chunks: list[tuple[int, int]], texts: list[str], sample_rate: int) -> list[Segment]:
    """Pair chunk boundaries with recognized text; drop empty chunks. Pure."""
    if len(chunks) != len(texts):
        raise ValueError(f"chunk/text count mismatch: {len(chunks)} chunks, {len(texts)} texts")
    segments = []
    for (start, end), text in zip(chunks, texts):
        text = " ".join((text or "").split())
        if text:
            segments.append(Segment(
                start=round(start / sample_rate, 3), end=round(end / sample_rate, 3), text=text,
            ))
    return segments


def _resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class OmnilingualEngine(Engine):
    name = "omnilingual"
    family = "omnilingual"

    def __init__(self, model_name=None):
        self.model_name = model_name or config.OMNILINGUAL_MODEL
        self.model = self.model_name
        self._pipeline = None

    @property
    def is_ctc(self) -> bool:
        return "_CTC_" in self.model_name

    def load(self) -> None:
        if self._pipeline is not None:
            return
        cache = MODEL_ROOT / "omnilingual"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("FAIRSEQ2_CACHE_DIR", str(cache))
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        import torch

        device = _resolve_device(config.OMNILINGUAL_DEVICE)
        # The pipeline defaults to bfloat16, which is slow or unsupported on CPU.
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self._pipeline = ASRInferencePipeline(model_card=self.model_name, device=device, dtype=dtype)
        omnilingual_marker(self.model_name).touch()
        self.ready = True

    def _run(self, inputs: list[dict], language: str | None) -> list[str]:
        out: list[str] = []
        size = max(1, config.OMNILINGUAL_BATCH)
        for i in range(0, len(inputs), size):
            batch = inputs[i:i + size]
            kwargs = {"batch_size": len(batch)}
            # CTC models are language-agnostic; LLM models accept an optional hint.
            if language and not self.is_ctc:
                kwargs["lang"] = [language] * len(batch)
            out.extend(str(t) for t in self._pipeline.transcribe(batch, **kwargs))
        return out

    def transcribe(self, audio: AudioInput, language, task) -> TranscriptResult:
        if task != "transcribe":
            raise ValueError("The Omnilingual engine only supports task=transcribe.")
        self.load()
        sr = audio.sample_rate
        chunks = split_on_silence(audio.samples, sr, max_chunk_sec=config.OMNILINGUAL_MAX_CHUNK_SEC)
        inputs = [{"waveform": audio.samples[s:e], "sample_rate": sr} for s, e in chunks]
        texts = self._run(inputs, language) if inputs else []
        segments = assemble_segments(chunks, texts, sr)
        return TranscriptResult(
            text=" ".join(s.text for s in segments),
            language=language or "",
            duration=audio.duration,
            segments=segments,
            backend=self.name,
            model=self.model_name,
        )
