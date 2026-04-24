"""Paired image+mask transforms for semantic segmentation training.

Light, validated stack only — no MixUp / CutMix / RandAugment / EMA / AMP.
Per feedback_hagrid_v3_overengineering.md, stacking these on MPS collapses
training. HFlip + color jitter + small rotation + scale-and-crop only.
"""
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IGNORE_INDEX  = 255  # used for padding regions when mask doesn't fill the crop


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
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=IGNORE_INDEX)
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
            image = TF.adjust_hue(image,        random.uniform(-0.05, 0.05))
        else:
            image = TF.resize(image, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask,  [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        x = TF.to_tensor(image)
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        return x, y

    def _random_scale_crop(self, image, mask):
        """Random scale 0.8x..1.2x then random crop to img_size."""
        s = random.uniform(0.8, 1.2)
        W, H = image.size
        new_W, new_H = int(W * s), int(H * s)
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
