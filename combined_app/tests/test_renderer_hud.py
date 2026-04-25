import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from combined_app.renderer import draw_clock, draw_label_strip


def _blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_clock_modifies_frame():
    frame = _blank()
    before = frame.copy()
    draw_clock(frame)
    assert not np.array_equal(frame, before), "draw_clock should draw text on frame"


def test_draw_clock_returns_none():
    frame = _blank()
    result = draw_clock(frame)
    assert result is None, "draw_clock modifies frame in place and returns None"


def test_draw_label_strip_empty_labels():
    frame = _blank()
    before = frame.copy()
    draw_label_strip(frame, {})
    # empty labels → nothing drawn, frame unchanged
    assert np.array_equal(frame, before)


def test_draw_label_strip_with_labels():
    frame = _blank()
    before = frame.copy()
    draw_label_strip(frame, {"objects": ["person", "chair"], "gesture": "like", "emotion": "happy"})
    assert not np.array_equal(frame, before), "draw_label_strip should draw text on frame"


def test_draw_label_strip_returns_none():
    frame = _blank()
    result = draw_label_strip(frame, {"gesture": "peace"})
    assert result is None
