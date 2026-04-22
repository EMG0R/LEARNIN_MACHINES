import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def _hand_mask(h=720, w=1280):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (640, 360), (80, 140), 0, 0, 360, 255, -1)
    return mask

def test_draw_mesh_returns_same_shape():
    from live_app.renderer import draw_mesh
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_mesh(frame.copy(), _hand_mask(), confidence=0.9, gesture_idx=3)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8

def test_draw_mesh_modifies_frame():
    from live_app.renderer import draw_mesh
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_mesh(frame.copy(), _hand_mask(), confidence=0.9, gesture_idx=3)
    assert not np.array_equal(out, frame)

def test_draw_mesh_low_conf_no_change():
    from live_app.renderer import draw_mesh
    from live_app.config import CONF_THRESHOLD
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_mesh(frame.copy(), _hand_mask(), confidence=CONF_THRESHOLD - 0.01, gesture_idx=0)
    assert np.array_equal(out, frame)

def test_draw_mesh_does_not_mutate_input():
    from live_app.renderer import draw_mesh
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    original = frame.copy()
    draw_mesh(frame, _hand_mask(), confidence=0.9, gesture_idx=3)
    assert np.array_equal(frame, original)

def test_draw_ui_returns_same_shape():
    from live_app.renderer import draw_ui
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_ui(frame.copy(), gesture="like", confidence=0.84,
                  second="palm", second_conf=0.10, fps=28.5, present=True)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8

def test_draw_ui_no_hand_renders():
    from live_app.renderer import draw_ui
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = draw_ui(frame.copy(), gesture=None, confidence=0.0,
                  second=None, second_conf=0.0, fps=28.5, present=False)
    assert out.shape == frame.shape

def test_draw_ui_does_not_mutate_input():
    from live_app.renderer import draw_ui
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    original = frame.copy()
    draw_ui(frame, gesture="fist", confidence=0.9,
            second="stop", second_conf=0.05, fps=30.0, present=True)
    assert np.array_equal(frame, original)
