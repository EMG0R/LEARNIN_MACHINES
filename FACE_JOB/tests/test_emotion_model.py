import torch
from FACE_JOB.emotion.model import EmotionWide, NUM_CLASSES


def test_forward_shape():
    model = EmotionWide()
    x = torch.randn(4, 3, 64, 64)
    y = model(x)
    assert y.shape == (4, NUM_CLASSES)
    assert NUM_CLASSES == 7
