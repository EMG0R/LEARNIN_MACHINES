import torch
from FACE_JOB.face_det.model import FaceDetector


def test_forward_shapes():
    model = FaceDetector()
    x = torch.randn(2, 3, 320, 320)
    obj, bbox, ctr = model(x)
    # Stride 8 → 40×40 feature map
    assert obj.shape  == (2, 1, 40, 40)
    assert bbox.shape == (2, 4, 40, 40)
    assert ctr.shape  == (2, 1, 40, 40)


def test_param_count_small():
    model = FaceDetector()
    n = sum(p.numel() for p in model.parameters())
    assert n < 3_000_000, f"expected <3M params, got {n}"
