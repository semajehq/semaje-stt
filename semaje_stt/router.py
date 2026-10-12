"""Engine routing. Pure and deterministic; languages come from packages/languages.

Two engine families, picked per request:

* Whisper (MIT) — 99 languages, word timestamps, translation to English. Very
  accurate on high-resource languages, unreliable on most others.
* Omnilingual ASR (Apache-2.0) — 1600+ languages including hundreds of
  low-resource African, Asian, and Indigenous languages Whisper never saw.

The router sends each language to the engine that is actually good at it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

from packages.languages import Language, UnknownLanguage, registry

# Languages where Whisper large-v2/v3 reaches roughly <=15% WER/CER on FLEURS
# (Radford et al. 2022, Appendix D). Everything else Whisper "supports" is
# routed to Omnilingual in auto mode. Override with STT_WHISPER_STRONG=en,fr,...
DEFAULT_WHISPER_STRONG = frozenset("""
en es it pt de ja ru pl fr ca nl tr ko uk id sv vi cs fi no da el ro hu ms zh gl sk bg hr
""".split())

# Below this Whisper language-ID probability the detection is not trusted.
DETECT_MIN_PROB = 0.6


@dataclass(frozen=True)
class Decision:
    engine: str                 # "whisper" | "omnilingual"
    language: str | None        # code in the chosen engine's code system
    canonical: str | None       # canonical registry code
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def whisper_strong() -> frozenset:
    override = os.getenv("STT_WHISPER_STRONG", "").strip()
    if override:
        return frozenset(code.strip() for code in override.split(",") if code.strip())
    return DEFAULT_WHISPER_STRONG


def route(
    language: Language | None,
    policy: str,
    available: set[str],
    detected: tuple[str, float] | None = None,
    task: str = "transcribe",
    strong: frozenset | None = None,
) -> Decision:
    """Pick the engine for one request.

    language  — resolved registry entry, or None for auto-detect
    policy    — "auto" | "whisper" | "omnilingual"
    available — engine families that are installed and loadable
    detected  — Whisper language-ID result (code, prob) when language is None
    task      — "translate" can only run on Whisper
    """
    strong = DEFAULT_WHISPER_STRONG if strong is None else strong
    if not available:
        raise RuntimeError("No STT engine is available on this host.")

    def whisper(reason: str, lang: Language | None) -> Decision:
        return Decision("whisper", lang.whisper if lang else None, lang.code if lang else None, reason)

    def omni(reason: str, lang: Language | None) -> Decision:
        return Decision("omnilingual", lang.code if lang and lang.omnilingual else None,
                        lang.code if lang else None, reason)

    has_w, has_o = "whisper" in available, "omnilingual" in available

    if task == "translate":
        if not has_w:
            raise RuntimeError("Translation needs the Whisper engine, which is not available.")
        if language and not language.whisper:
            raise UnknownLanguage(
                f"Translation is only available for Whisper's 99 languages; {language.name} is not one of them."
            )
        return whisper("translation runs on Whisper", language)

    if policy == "whisper" or not has_o:
        if language and not language.whisper:
            if has_o:
                return omni(f"{language.name} is not a Whisper language", language)
            raise UnknownLanguage(
                f"{language.name} needs the Omnilingual engine, which is not installed on this host."
            )
        why = "policy=whisper" if policy == "whisper" else "omnilingual engine not installed"
        return whisper(why, language)

    if policy == "omnilingual" or not has_w:
        if language and not language.omnilingual:
            if has_w and language.whisper:
                return whisper(f"{language.name} is not an Omnilingual language", language)
            raise UnknownLanguage(f"{language.name} is not supported by any installed engine.")
        why = "policy=omnilingual" if policy == "omnilingual" else "whisper engine not installed"
        return omni(why, language)

    # policy == auto, both engines present
    if language is not None:
        if language.whisper in strong:
            return whisper(f"{language.name} is a strong Whisper language", language)
        if language.omnilingual:
            return omni(f"{language.name} is better served by Omnilingual", language)
        return whisper(f"{language.name} is only covered by Whisper", language)

    if detected is not None:
        code, prob = detected
        found = registry().from_whisper(code)
        if code in strong and prob >= DETECT_MIN_PROB:
            return whisper(f"detected {code} (p={prob:.2f}), a strong Whisper language", found)
        return omni(f"detected {code} (p={prob:.2f}); routing to the broader-coverage engine", found)

    return whisper("no language hint and no detection; Whisper auto-detects", None)
