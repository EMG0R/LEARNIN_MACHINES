"""Paired image+mask transforms for semantic segmentation training.

v4: stronger augmentation aligned with YOLOv5/v8 defaults — wider scale
range (0.5-1.5), HSV-based color jitter (instead of RGB-based), larger
rotation. Still avoids the v3-overengineering trap (no MixUp, CutMix,
RandAugment, EMA, AMP — those collapsed gesture training on MPS).
"""
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IGNORE_INDEX  = 255  # used for padding regions when mask doesn't fill the crop


def _hsv_jitter(image: Image.Image, h_jitter=0.015, s_jitter=0.7, v_jitter=0.4):
    """YOLOv5-style HSV jitter. h/s/v perturbations applied multiplicatively
    in HSV space — more domain-realistic than independent RGB channel jitter.
    Defaults match YOLOv5 (h=0.015, s=0.7, v=0.4).
    """
    arr = np.asarray(image.convert("HSV"), dtype=np.int16)
    rh = random.uniform(-h_jitter, h_jitter) * 180
    rs = random.uniform(1 - s_jitter, 1 + s_jitter)
    rv = random.uniform(1 - v_jitter, 1 + v_jitter)

    arr[..., 0] = (arr[..., 0] + int(rh)) % 180
    arr[..., 1] = np.clip(arr[..., 1] * rs, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * rv, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")


class SegTransform:
    def __init__(self, img_size: int, mode: str = "train"):
        assert mode in ("train", "eval")
        self.img_size = img_size
        self.mode = mode

    def __call__(self, image: Image.Image, mask: Image.Image):
        if self.mode == "train":
            image, mask = self._random_scale_crop(image, mask)
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=IGNORE_INDEX)
            image = _hsv_jitter(image)
        else:
            image = TF.resize(image, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask,  [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        x = TF.to_tensor(image)
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        return x, y

    def _random_scale_crop(self, image, mask):
        """Random scale 0.5x..1.5x then random crop to img_size.
        Wider range than v1-v3 (0.8-1.2) — exposes model to more sizes
        per epoch, helps with size invariance and small-object recall.
        """
        s = random.uniform(0.5, 1.5)
        W, H = image.size
        new_W, new_H = max(1, int(W * s)), max(1, int(H * s))
        image = TF.resize(image, [new_H, new_W], interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  [new_H, new_W], interpolation=TF.InterpolationMode.NEAREST)

        # Pad if smaller than target
        pad_w = max(0, self.img_size - new_W)
        pad_h = max(0, self.img_size - new_H)
        if pad_w or pad_h:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask  = TF.pad(mask,  [0, 0, pad_w, pad_h], fill=IGNORE_INDEX)
            new_W += pad_w
            new_H += pad_h

        # Random crop to img_size x img_size
        x0 = random.randint(0, new_W - self.img_size)
        y0 = random.randint(0, new_H - self.img_size)
        image = TF.crop(image, y0, x0, self.img_size, self.img_size)
        mask  = TF.crop(mask,  y0, x0, self.img_size, self.img_size)
        return image, mask
