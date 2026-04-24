import numpy as np
import torch
from PIL import Image

from OBJECTIFICATION.seg.augment import SegTransform
from OBJECTIFICATION.seg.dataset import OpenImagesSegDataset


def _seed_split(root, n=4):
    """Create n image+mask pairs in root/{images,masks}/."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    ids = []
    for i in range(n):
        iid = f"img{i:04d}"
        Image.fromarray(np.full((100, 100, 3), i * 30 + 50, dtype=np.uint8)).save(
            root / "images" / f"{iid}.jpg")
        m = np.zeros((100, 100), dtype=np.uint8)
        m[:50, :50] = (i % 23) + 1   # one foreground class per image
        Image.fromarray(m).save(root / "masks" / f"{iid}.png")
        ids.append(iid)
    return ids


def test_dataset_returns_correct_shapes(tmp_path):
    _seed_split(tmp_path)
    ds = OpenImagesSegDataset(tmp_path, transform=SegTransform(img_size=64, mode="eval"))
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    assert y.shape == (64, 64)
    assert y.dtype == torch.long


def test_dataset_skips_image_without_mask(tmp_path):
    _seed_split(tmp_path, n=2)
    # extra image with no matching mask
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(
        tmp_path / "images" / "orphan.jpg")
    ds = OpenImagesSegDataset(tmp_path, transform=SegTransform(img_size=64, mode="eval"))
    assert len(ds) == 2  # orphan is filtered out


def test_class_freq_counts_only_foreground(tmp_path):
    _seed_split(tmp_path, n=3)
    ds = OpenImagesSegDataset(tmp_path, transform=SegTransform(img_size=64, mode="eval"))
    freq = ds.class_freq(num_classes=24)
    assert freq.shape == (24,)
    assert freq[0] == 0  # background NOT counted as a foreground class
    assert freq.sum() > 0
