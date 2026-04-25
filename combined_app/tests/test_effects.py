import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from combined_app.effects import FlowField


def test_buffer_shape():
    ff = FlowField(480, 640)
    assert ff.buffer.shape == (480, 640, 3)
    assert ff.buffer.dtype == np.uint8


def test_tick_advances_time():
    ff = FlowField(480, 640, step=0.01)
    t0 = ff.t
    ff.tick()
    assert ff.t == pytest.approx(t0 + 0.01)


def test_tick_changes_buffer():
    ff = FlowField(480, 640)
    before = ff.buffer.copy()
    ff.tick()
    # after first tick the buffer should have non-zero content
    assert ff.buffer.sum() > 0


def test_blend_onto_shape():
    ff = FlowField(480, 640)
    ff.tick()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = ff.blend_onto(frame)
    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint8


def test_blend_onto_adds_color():
    ff = FlowField(480, 640, opacity=1.0)
    ff.tick()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = ff.blend_onto(frame)
    # at opacity=1.0, result should equal the effects buffer
    np.testing.assert_array_equal(result, ff.buffer)
