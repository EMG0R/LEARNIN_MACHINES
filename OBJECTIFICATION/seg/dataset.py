"""OpenImages V7 semantic-segmentation dataset.

Expects a directory tree:
    root/images/{image_id}.jpg
    root/masks/{image_id}.png   (uint8, pixel value = class id 0..23)

Built by OBJECTIFICATION/data_pipeline/{download,prepare_masks}.py.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OpenImagesSegDataset(Dataset):
    def __init__(self, root, transform=None):
        root = Path(root)
        self.root = root
        self.transform = transform
        # Index = image stems that have BOTH an image and a mask file
        masks = {p.stem for p in (root / "masks").glob("*.png")}
        self.ids = sorted([
            p.stem for p in (root / "images").glob("*.jpg") if p.stem in masks
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        iid = self.ids[k]
        img  = Image.open(self.root / "images" / f"{iid}.jpg").convert("RGB")
        mask = Image.open(self.root / "masks"  / f"{iid}.png")
        if self.transform is not None:
            return self.transform(img, mask)
        return img, mask

    def class_freq(self, num_classes: int) -> np.ndarray:
        """Pixel count per class, summed over the whole dataset.
        Index 0 (background) is not counted (kept as 0 to avoid skewing
        loss/sampler weights toward background).
        """
        counts = np.zeros(num_classes, dtype=np.int64)
        for iid in self.ids:
            arr = np.array(Image.open(self.root / "masks" / f"{iid}.png"))
            uniq, c = np.unique(arr, return_counts=True)
            for u, n in zip(uniq, c):
                if 1 <= u < num_classes:
                    counts[u] += int(n)
        return counts

    def sample_weights(self, num_classes: int) -> np.ndarray:
        """Per-image sampling weight: 1 / sqrt(min_class_freq_in_image).
        Background ignored. Images with no foreground get weight 0.
        """
        global_freq = self.class_freq(num_classes).astype(np.float64)
        global_freq[global_freq == 0] = 1.0  # avoid div-by-zero for absent classes
        weights = np.zeros(len(self.ids), dtype=np.float64)
        for i, iid in enumerate(self.ids):
            arr = np.array(Image.open(self.root / "masks" / f"{iid}.png"))
            classes_in_image = [c for c in np.unique(arr) if 1 <= c < num_classes]
            if not classes_in_image:
                continue
            min_freq = min(global_freq[c] for c in classes_in_image)
            weights[i] = 1.0 / np.sqrt(min_freq)
        return weights
