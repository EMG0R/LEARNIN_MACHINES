"""
Combined emotion dataset: FER+ ∪ RAF-DB ∪ ExpW.

Each sample carries a `source` tag used for:
  - Per-sample loss weighting via `SOURCE_LOSS_WEIGHTS`
  - Balanced batch sampling: each batch draws equally from each source

Images can come from three formats:
  - FER+  → numpy array 48×48 uint8 (grayscale) → upscale + RGB replicate
  - RAF-DB → path to aligned JPG
  - ExpW  → path to JPG
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

SOURCES = ["ferplus", "rafdb", "expw"]
SOURCE_LOSS_WEIGHTS = {"ferplus": 1.0, "rafdb": 1.5, "expw": 0.7}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class EmotionSample:
    image: Union[np.ndarray, Path, Image.Image, None]
    label: int
    source: str


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img).convert("RGB")
        return Image.fromarray(img)
    if isinstance(img, Path):
        return Image.open(img).convert("RGB")
    raise TypeError(f"unsupported image type: {type(img)}")


def _eval_tf(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _train_tf(img_size):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.17)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CombinedEmotionDS(Dataset):
    def __init__(self, samples: list[EmotionSample], img_size: int = 64, mode: str = "train"):
        self.samples = samples
        self.img_size = img_size
        self.mode = mode
        self.tf = _train_tf(img_size) if mode == "train" else _eval_tf(img_size)
        self._source_idx = {s: i for i, s in enumerate(SOURCES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        pil = _to_pil(s.image)
        tensor = self.tf(pil)
        return tensor, s.label, self._source_idx[s.source]


class BalancedSourceSampler(Sampler[int]):
    """Each batch draws batch_size/len(SOURCES) items from each source."""

    def __init__(self, samples: list[EmotionSample], batch_size: int, num_batches: int, seed: int = 42):
        assert batch_size % len(SOURCES) == 0, "batch_size must divide by number of sources"
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.per_source = batch_size // len(SOURCES)
        self.rng = random.Random(seed)
        self._source_indices: dict[str, list[int]] = {src: [] for src in SOURCES}
        for idx, s in enumerate(samples):
            self._source_indices[s.source].append(idx)
        for src, lst in self._source_indices.items():
            if not lst:
                raise ValueError(f"no samples for source {src!r}")

    def __iter__(self):
        for _ in range(self.num_batches):
            batch: list[int] = []
            for src in SOURCES:
                pool = self._source_indices[src]
                batch.extend(self.rng.choices(pool, k=self.per_source))
            self.rng.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.batch_size * self.num_batches
