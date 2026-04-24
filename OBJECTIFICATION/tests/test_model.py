import torch
from OBJECTIFICATION.seg.model import C3

def test_c3_preserves_spatial_dims():
    x = torch.randn(2, 64, 40, 40)
    block = C3(64, 64, n=2)
    y = block(x)
    assert y.shape == (2, 64, 40, 40)

def test_c3_changes_channels():
    x = torch.randn(2, 64, 40, 40)
    block = C3(64, 128, n=1)
    y = block(x)
    assert y.shape == (2, 128, 40, 40)
