import torch
from FACE_JOB.face_parts.model import FacePartsUNet, NUM_CLASSES


def test_forward_shape():
    model = FacePartsUNet().eval()
    x = torch.randn(2, 3, 192, 192)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, NUM_CLASSES, 192, 192)


def test_training_forward_returns_aux():
    model = FacePartsUNet().train()
    x = torch.randn(1, 3, 192, 192)
    main, a3, a2 = model(x)
    assert main.shape == (1, NUM_CLASSES, 192, 192)
    assert a3.shape[1] == NUM_CLASSES
    assert a2.shape[1] == NUM_CLASSES
