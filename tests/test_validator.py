from vision.validator import quote_supported, validate_window, window_text

SEGS = [
    {"start": 0.0, "end": 8.0, "text": "Thank you very much."},
    {"start": 8.0, "end": 15.0, "text": "Charlie Kirk is doing an incredible job."},
    {"start": 15.0, "end": 25.0, "text": "He is a tremendous advocate."},
]


def test_good_window_is_untouched():
    w = validate_window(8.0, 20.0, 60, 12, [])
    assert (w.start, w.end, w.factor) == (8.0, 20.0, 1.0)


def test_snaps_to_transcript_boundaries():
    w = validate_window(8.4, 24.6, 60, 17, SEGS)
    assert (w.start, w.end, w.factor) == (8.0, 25.0, 1.0)


def test_too_long_is_trimmed():
    w = validate_window(8.0, 40.0, 60, 12, [])
    assert (w.start, w.end, w.factor) == (8.0, 20.0, 0.9)


def test_too_short_is_extended_inside_video():
    w = validate_window(55.0, 58.0, 60, 12, [])
    assert (w.start, w.end, w.factor) == (48.0, 60.0, 0.9)


def test_out_of_bounds_is_clamped():
    w = validate_window(-3.0, 70.0, 60, 12, [])
    assert w.start == 0.0 and w.end == 12.0


def test_video_shorter_than_target_is_penalized():
    w = validate_window(0.0, 5.0, 5, 12, [])
    assert (w.start, w.end, w.factor) == (0.0, 5.0, 0.5)


def test_unusable_windows_are_rejected():
    assert validate_window(20.0, 10.0, 60, 12, []) is None
    assert validate_window(70.0, 80.0, 60, 12, []) is None


def test_quote_must_come_from_the_window():
    text = window_text(SEGS, 8.0, 15.0)
    assert text == "Charlie Kirk is doing an incredible job."
    assert quote_supported("Charlie Kirk is doing an incredible job", text)
    assert not quote_supported("I love what he does for free speech", text)
    assert not quote_supported("", text)
