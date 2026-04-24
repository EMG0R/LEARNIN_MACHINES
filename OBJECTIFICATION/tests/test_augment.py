import numpy as np
import torch
from PIL import Image

from OBJECTIFICATION.seg.augment import SegTransform


def _img_mask(size=64):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:size // 2, :, 0] = 255  # red top half
    arr[size // 2:, :, 2] = 255  # blue bottom half
    img = Image.fromarray(arr)
    m = np.zeros((size, size), dtype=np.uint8)
    m[:size // 2, :] = 1
    m[size // 2:, :] = 2
    mask = Image.fromarray(m)
    return img, mask


def test_train_returns_correct_shape_and_dtype():
    img, mask = _img_mask()
    t = SegTransform(img_size=320, mode="train")
    x, y = t(img, mask)
    assert x.shape == (3, 320, 320) and x.dtype == torch.float32
    assert y.shape == (320, 320)    and y.dtype == torch.long


def test_eval_is_deterministic():
    img, mask = _img_mask()
    t = SegTransform(img_size=320, mode="eval")
    x1, y1 = t(img, mask)
    x2, y2 = t(img, mask)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_mask_only_contains_valid_class_ids():
    img, mask = _img_mask()
    t = SegTransform(img_size=320, mode="train")
    _, y = t(img, mask)
    # input mask had values {0, 1, 2}; after rotation/crop only those + ignore=255 may appear
    valid = {0, 1, 2, 255}
    assert set(np.unique(y.numpy()).tolist()).issubset(valid)
