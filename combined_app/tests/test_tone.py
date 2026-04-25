import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from combined_app.tone import build_lut, apply_grade


def test_lut_shape_and_dtype():
    lut = build_lut()
    assert lut.shape == (256,)
    assert lut.dtype == np.uint8


def test_lut_identity_at_gamma_one():
    lut = build_lut(gamma=1.0)
    # at gamma=1, output[i] should equal i (±1 due to float rounding)
    for i in range(256):
        assert abs(int(lut[i]) - i) <= 1


def test_lut_brightens_at_gamma_above_one():
    lut = build_lut(gamma=1.2)
    # gamma > 1 brightens midtones: lut[128] > 128
    assert int(lut[128]) > 128


def test_apply_grade_shape():
    lut = build_lut()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = apply_grade(frame, lut)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8


def test_apply_grade_desaturates():
    lut = build_lut()
    # A fully saturated red frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 2] = 200  # red channel
    result = apply_grade(frame, lut, sat_factor=0.0)
    # sat_factor=0 → all channels equal (greyscale)
    assert np.allclose(result[:,:,0], result[:,:,1], atol=2)
    assert np.allclose(result[:,:,1], result[:,:,2], atol=2)
