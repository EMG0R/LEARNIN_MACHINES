"""Shared pytest fixtures."""
import numpy as np
import pytest
from PIL import Image

@pytest.fixture
def tiny_instance_masks(tmp_path):
    """Three 32×32 PNGs simulating OI instance masks for one image:
    - mask A: top-left 16×16 white (= person, MID '/m/01g317')
    - mask B: bottom-right 16×16 white (= chair, MID '/m/01mzpv')
    - mask C: a 4×4 white square overlapping A (= person again)
    """
    arr = np.zeros((32, 32), dtype=np.uint8)

    a = arr.copy(); a[0:16, 0:16] = 255
    b = arr.copy(); b[16:32, 16:32] = 255
    c = arr.copy(); c[8:12, 8:12] = 255

    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(b).save(tmp_path / "b.png")
    Image.fromarray(c).save(tmp_path / "c.png")
    return [
        (tmp_path / "a.png", "/m/01g317"),  # person
        (tmp_path / "b.png", "/m/01mzpv"),  # chair
        (tmp_path / "c.png", "/m/01g317"),  # person (overlap)
    ]

@pytest.fixture
def fake_class_map():
    return {"/m/01g317": 1, "/m/01mzpv": 14}  # person=1, chair=14
