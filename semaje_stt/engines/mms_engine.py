"""MMS-1b-all engine — Meta's 1,162-language wav2vec2 CTC model with per-language
adapters. Weights are CC-BY-NC-4.0 (non-commercial), so it only runs when an
operator sets STT_ALLOW_NONCOMMERCIAL=true, and only for STT_MMS_LANGUAGES.

This is the ASR C-elo Labs uses for Kikuyu. On WAXAL test clips it beats
Omnilingual CTC-300M on words (Kikuyu WER 57% vs 86%, Dholuo 37% vs 48%)
because the language adapter keeps output inside the language's orthography.
"""
from __future__ import annotations

import threading

from .base import Engine, TranscriptResult, AudioInput
from .. import config
from ..audio import split_on_silence
from .omnilingual_engine import assemble_segments


class MmsEngine(Engine):
    name = "mms-1b-all"
    family = "mms"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.MMS_MODEL
        self.model = self.model_name.split("/")[-1]
        self._model = None
        self._processor = None
        self._adapter = None
        self._lock = threading.Lock()  # adapters mutate the shared model

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self._processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=config.MMS_CACHE or None)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name, cache_dir=config.MMS_CACHE or None).eval()
        self.ready = True

    def _use(self, adapter: str) -> None:
        if self._adapter != adapter:
            self._processor.tokenizer.set_target_lang(adapter)
            self._model.load_adapter(adapter)
            self._adapter = adapter

    def transcribe(self, audio: AudioInput, language, task) -> TranscriptResult:
        if task != "transcribe":
            raise ValueError("The MMS engine only supports task=transcribe.")
        if not language:
            raise ValueError("The MMS engine needs a language (an ISO 639-3 adapter id such as 'kik').")
        self.load()
        import torch
        sr = audio.sample_rate
        chunks = split_on_silence(audio.samples, sr, max_chunk_sec=config.OMNILINGUAL_MAX_CHUNK_SEC)
        texts = []
        with self._lock:
            self._use(language)
            for start, end in chunks:
                inputs = self._processor(audio.samples[start:end], sampling_rate=sr, return_tensors="pt")
                with torch.no_grad():
                    ids = torch.argmax(self._model(**inputs).logits, dim=-1)[0]
                texts.append(self._processor.decode(ids))
        segments = assemble_segments(chunks, texts, sr)
        return TranscriptResult(text=" ".join(s.text for s in segments), language=language,
                                duration=audio.duration, segments=segments, backend=self.name, model=self.model)
