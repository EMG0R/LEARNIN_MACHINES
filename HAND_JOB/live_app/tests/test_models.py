import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_seg_inference_shape():
    from live_app.models import load_models, run_seg
    seg_model, _, _ = load_models()
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = run_seg(fake_frame, seg_model)
    assert mask.shape == (720, 1280)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})

def test_gesture_inference_returns_dict():
    from live_app.models import load_models, run_seg, run_gesture
    seg_model, gest_model, class_names = load_models()
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = run_seg(fake_frame, seg_model)
    result = run_gesture(fake_frame, mask, gest_model, class_names)
    assert "gesture" in result
    assert "confidence" in result
    assert "second" in result
    assert "second_conf" in result
    assert "gesture_idx" in result
    assert 0.0 <= result["confidence"] <= 1.0
