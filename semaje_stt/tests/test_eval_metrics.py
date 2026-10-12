"""Gate tests: the eval's metric math (so a scoring bug can't fake a pass)."""
from services.stt.evals.eval_fleurs import cer, normalize, wer


def test_normalize_strips_punctuation_and_case():
    assert normalize("Habari, Yako?  Nzuri!") == "habari yako nzuri"
    assert normalize("ሰላም። እንዴት") == "ሰላም እንዴት"


def test_wer_and_cer():
    assert wer("a b c d", "a b c d") == 0
    assert wer("a b c d", "a x c") == 0.5          # 1 sub + 1 del over 4 words
    assert cer("abcd", "abed") == 0.25
    assert wer("", "anything") == 1.0             # empty ref never divides by zero


def test_metrics_are_unicode_aware():
    assert cer("ẹ̀kọ́", "ẹ̀kọ́") == 0
    assert cer("ẹkọ", "eko") > 0  # Yoruba tone/underdot errors count
