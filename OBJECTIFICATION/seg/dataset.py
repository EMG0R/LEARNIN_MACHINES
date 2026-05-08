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
        # Robust to corrupted/truncated images: on any read error, fall
        # through to the next valid sample instead of crashing the
        # whole training loop. A handful of bad files in a 300K-image
        # mixed-source dataset (OI + COCO) is normal — log first 10 and
        # then go silent.
        n = len(self.ids)
        for off in range(n):
            iid = self.ids[(k + off) % n]
            try:
                img  = Image.open(self.root / "images" / f"{iid}.jpg").convert("RGB")
                # Force "L" mode (8-bit single channel) so values are always
                # uint8 0-255. Some PNGs save as 16-bit or palette mode, which
                # would produce out-of-range pixel values that crash the loss.
                mask = Image.open(self.root / "masks"  / f"{iid}.png").convert("L")
                img.load(); mask.load()  # force decode now so errors surface here
                if self.transform is not None:
                    return self.transform(img, mask)
                return img, mask
            except Exception as e:
                if not hasattr(self, "_bad_seen"):
                    self._bad_seen = set()
                if iid not in self._bad_seen and len(self._bad_seen) < 10:
                    print(f"[dataset] skip corrupted {iid}: {type(e).__name__}", flush=True)
                self._bad_seen.add(iid)
                continue
        raise RuntimeError("all images in dataset are corrupted — abort")

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
