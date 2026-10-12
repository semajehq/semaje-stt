"""Gate tests: engine routing decisions. Pure, deterministic."""
import pytest

from packages.languages import UnknownLanguage, registry
from services.stt.router import DEFAULT_WHISPER_STRONG, route, whisper_strong

BOTH = {"whisper", "omnilingual"}
R = registry()


def lang(code):
    return R.resolve(code)


def test_strong_whisper_language_goes_to_whisper():
    d = route(lang("fr"), "auto", BOTH)
    assert (d.engine, d.language, d.canonical) == ("whisper", "fr", "fra_Latn")


@pytest.mark.parametrize("code,expected_engine_code", [
    ("sw", "swh_Latn"),     # Whisper "supports" Swahili, but poorly → Omnilingual
    ("yo", "yor_Latn"),
    ("am", "amh_Ethi"),
    ("kik", "kik_Latn"),    # Kikuyu: Whisper never saw it
    ("luo", "luo_Latn"),
    ("ig", "ibo_Latn"),
])
def test_low_resource_languages_go_to_omnilingual(code, expected_engine_code):
    d = route(lang(code), "auto", BOTH)
    assert d.engine == "omnilingual"
    assert d.language == expected_engine_code


def test_whisper_only_language_stays_on_whisper_in_auto():
    d = route(lang("nn"), "auto", BOTH)  # Nynorsk: not in Omnilingual
    assert (d.engine, d.language) == ("whisper", "nn")


def test_policy_whisper_forces_whisper_for_whisper_languages():
    d = route(lang("sw"), "whisper", BOTH)
    assert (d.engine, d.language) == ("whisper", "sw")


def test_policy_whisper_still_routes_non_whisper_language_to_omnilingual():
    d = route(lang("kik"), "whisper", BOTH)
    assert d.engine == "omnilingual"


def test_policy_omnilingual_forces_omnilingual():
    d = route(lang("fr"), "omnilingual", BOTH)
    assert (d.engine, d.language) == ("omnilingual", "fra_Latn")


def test_policy_omnilingual_falls_back_for_languages_it_lacks():
    d = route(lang("nn"), "omnilingual", BOTH)
    assert (d.engine, d.language) == ("whisper", "nn")


def test_missing_omnilingual_engine_degrades_to_whisper():
    d = route(lang("sw"), "auto", {"whisper"})
    assert (d.engine, d.language) == ("whisper", "sw")
    assert "not installed" in d.reason


def test_missing_omnilingual_engine_rejects_non_whisper_language():
    with pytest.raises(UnknownLanguage, match="Omnilingual"):
        route(lang("kik"), "auto", {"whisper"})


def test_missing_whisper_engine_uses_omnilingual():
    d = route(lang("en"), "auto", {"omnilingual"})
    assert (d.engine, d.language) == ("omnilingual", "eng_Latn")


def test_translate_always_runs_on_whisper():
    d = route(lang("sw"), "auto", BOTH, task="translate")
    assert (d.engine, d.language) == ("whisper", "sw")


def test_translate_rejects_non_whisper_language():
    with pytest.raises(UnknownLanguage, match="Translation"):
        route(lang("kik"), "auto", BOTH, task="translate")


def test_translate_without_whisper_engine_is_an_error():
    with pytest.raises(RuntimeError):
        route(None, "auto", {"omnilingual"}, task="translate")


def test_auto_detect_confident_strong_language_uses_whisper():
    d = route(None, "auto", BOTH, detected=("de", 0.97))
    assert (d.engine, d.canonical) == ("whisper", "deu_Latn")


def test_auto_detect_low_confidence_goes_to_omnilingual():
    d = route(None, "auto", BOTH, detected=("en", 0.31))
    assert d.engine == "omnilingual"


def test_auto_detect_weak_whisper_language_goes_to_omnilingual_with_hint():
    d = route(None, "auto", BOTH, detected=("sw", 0.9))
    assert (d.engine, d.language) == ("omnilingual", "swh_Latn")


def test_no_language_no_detection_lets_whisper_auto_detect():
    d = route(None, "auto", BOTH)
    assert (d.engine, d.language) == ("whisper", None)


def test_no_engines_is_an_error():
    with pytest.raises(RuntimeError):
        route(lang("en"), "auto", set())


def test_strong_set_override(monkeypatch):
    monkeypatch.setenv("STT_WHISPER_STRONG", "sw, en")
    strong = whisper_strong()
    assert strong == frozenset({"sw", "en"})
    assert route(lang("sw"), "auto", BOTH, strong=strong).engine == "whisper"
    assert route(lang("fr"), "auto", BOTH, strong=strong).engine == "omnilingual"


def test_every_strong_language_is_a_real_whisper_code():
    whisper_codes = {l.whisper for l in R.languages if l.whisper}
    assert DEFAULT_WHISPER_STRONG <= whisper_codes
