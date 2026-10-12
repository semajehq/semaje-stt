"""Gate tests: the WAXAL misalignment filter (pure)."""
from semaje_stt.evals.eval_waxal import is_aligned


def test_normal_read_speech_is_kept():
    assert is_aligned("a" * 120, 10.0)          # 12 chars/s


def test_audio_far_longer_than_text_is_dropped():
    assert not is_aligned("x" * 330, 167.3)     # 2 chars/s: extra unscripted speech


def test_text_far_longer_than_audio_is_dropped():
    assert not is_aligned("x" * 98, 1.7)        # ~56 chars/s: truncated audio


def test_overlong_and_empty_clips_are_dropped():
    assert not is_aligned("x" * 600, 45.0)
    assert not is_aligned("hello", 0.0)


def test_whitespace_does_not_inflate_rate():
    assert is_aligned("habari   yako\n\nnzuri sana " * 4, 8.0)
