import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from collections import deque
from live_app.config import (
    SEG_CKPT, GESTURE_CKPT, SEG_THRESHOLD, HAND_MIN_AREA,
    MASK_EMA_ALPHA, MIN_CC_AREA_PX, VOTE_WINDOW,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _conv_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )

class _HandUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = _conv_block(3, 32);  self.d2 = _conv_block(32, 64)
        self.d3 = _conv_block(64, 128); self.d4 = _conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.u3 = _conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128,  64, 2, stride=2); self.u2 = _conv_block(128,  64)
        self.up1 = nn.ConvTranspose2d(64,   32, 2, stride=2); self.u1 = _conv_block(64,   32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.d1(x); c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2)); c4 = self.d4(self.pool(c3))
        u3 = self.u3(torch.cat([self.up3(c4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out(u1)


class _Wide(nn.Module):
    def __init__(self, n=18):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Dropout2d(0.1), nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3,32), block(32,64), block(64,128), block(128,256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, n),
        )
    def forward(self, x): return self.head(self.features(x))


def load_models():
    seg_ckpt = torch.load(SEG_CKPT, map_location="cpu", weights_only=False)
    seg_model = _HandUNet()
    # strict=False: v3 seg ckpt has aux3/aux2 heads used only during training
    seg_model.load_state_dict(seg_ckpt["model_state_dict"], strict=False)
    seg_model.to(device).eval()
    seg_model._img_size = seg_ckpt["img_size"]
    seg_model._mean     = seg_ckpt["normalize_mean"]
    seg_model._std      = seg_ckpt["normalize_std"]

    g_ckpt = torch.load(GESTURE_CKPT, map_location="cpu", weights_only=False)
    class_names = g_ckpt["class_names"]
    gest_model = _Wide(n=len(class_names))
    gest_model.load_state_dict(g_ckpt["model_state_dict"])
    gest_model.to(device).eval()
    gest_model._img_size = g_ckpt["img_size"]
    gest_model._mean     = g_ckpt["normalize_mean"]
    gest_model._std      = g_ckpt["normalize_std"]

    return seg_model, gest_model, class_names


def run_seg_prob(frame_bgr: np.ndarray, seg_model) -> np.ndarray:
    """Return float prob map (0..1) at frame resolution — no thresholding."""
    h, w = frame_bgr.shape[:2]
    pil  = Image.fromarray(frame_bgr[:, :, ::-1])
    pil  = pil.resize((seg_model._img_size, seg_model._img_size), Image.BILINEAR)
    t    = TF.normalize(TF.to_tensor(pil), seg_model._mean, seg_model._std)
    with torch.no_grad():
        logit = seg_model(t.unsqueeze(0).to(device))
    prob = torch.sigmoid(logit).squeeze().cpu().numpy().astype(np.float32)
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def run_seg_prob_batch(crops: list, seg_model) -> list:
    """Run seg on a list of BGR crops in a single batched forward pass.
    Returns a list of prob maps, each at its crop's resolution."""
    sz = seg_model._img_size
    tensors = []
    shapes  = []
    for crop in crops:
        h, w = crop.shape[:2]
        shapes.append((w, h))
        pil = Image.fromarray(crop[:, :, ::-1])
        pil = pil.resize((sz, sz), Image.BILINEAR)
        tensors.append(TF.normalize(TF.to_tensor(pil), seg_model._mean, seg_model._std))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        logits = seg_model(batch)
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
    return [cv2.resize(probs[i], shapes[i], interpolation=cv2.INTER_LINEAR)
            for i in range(len(crops))]


def run_seg(frame_bgr: np.ndarray, seg_model) -> np.ndarray:
    """Single-frame binary mask (legacy, no smoothing)."""
    prob = run_seg_prob(frame_bgr, seg_model)
    return (prob > SEG_THRESHOLD).astype(np.uint8) * 255


def postprocess_mask(prob: np.ndarray, prev_ema: np.ndarray | None):
    """EMA-smooth prob map, threshold, and filter components.
    
    Uses morphological closing to bridge small gaps and keeps all components 
    above MIN_CC_AREA_PX to handle fragmented masks or multiple hands.
    """
    if prev_ema is None:
        ema = prob
    else:
        ema = MASK_EMA_ALPHA * prob + (1.0 - MASK_EMA_ALPHA) * prev_ema

    binary = (ema > SEG_THRESHOLD).astype(np.uint8)
    
    # Morphological closing bridges small gaps in the mask (e.g. between fingers)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary, dtype=np.uint8), ema
    
    # Keep all components that are large enough (noise filter)
    # Keeping multiple helps when a hand is partially occluded or for two hands.
    mask = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_CC_AREA_PX:
            mask[labels == i] = 1
            
    return mask * 255, ema


def hand_present(mask: np.ndarray) -> bool:
    """Gate for Layer 3b: mask must cover at least HAND_MIN_AREA of the frame."""
    if mask is None:
        return False
    h, w = mask.shape
    return (np.count_nonzero(mask) / (h * w)) >= HAND_MIN_AREA


def run_gesture(frame_bgr: np.ndarray, mask: np.ndarray,
                gest_model, class_names: list) -> dict:
    empty = {"gesture": None, "confidence": 0.0, "second": None, "second_conf": 0.0,
             "gesture_idx": 0, "probs": None}
    coords = cv2.findNonZero(mask)
    if coords is None:
        return empty
    x, y, bw, bh = cv2.boundingRect(coords)
    pad = int(max(bw, bh) * 0.15)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(frame_bgr.shape[1], x + bw + pad)
    y1 = min(frame_bgr.shape[0], y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return empty
    pil = Image.fromarray(frame_bgr[y0:y1, x0:x1, ::-1])
    pil = pil.resize((gest_model._img_size, gest_model._img_size), Image.BILINEAR)
    t   = TF.normalize(TF.to_tensor(pil), gest_model._mean, gest_model._std)
    with torch.no_grad():
        logits = gest_model(t.unsqueeze(0).to(device))
    probs = torch.softmax(logits.squeeze(), dim=0).cpu().numpy()
    # merge two_up_inverted → middle_finger
    if "two_up_inverted" in class_names and "middle_finger" in class_names:
        probs[class_names.index("middle_finger")] += probs[class_names.index("two_up_inverted")]
        probs[class_names.index("two_up_inverted")] = 0.0
    top2  = np.argsort(probs)[::-1][:2]
    return {
        "gesture":     class_names[top2[0]],
        "confidence":  float(probs[top2[0]]),
        "second":      class_names[top2[1]],
        "second_conf": float(probs[top2[1]]),
        "gesture_idx": int(top2[0]),
        "probs":       probs.astype(np.float32),
    }


class GestureSmoother:
    """Sums softmax probs over last VOTE_WINDOW frames, emits smoothed top-2.

    Prob-space voting (not mode-of-argmax): a frame that's 60/40 split contributes
    both classes, so close calls don't flicker with one loud wrong frame.
    """
    def __init__(self, class_names, window: int = VOTE_WINDOW):
        self.class_names = class_names
        self.window      = window
        self.buf         = deque(maxlen=window)

    def reset(self):
        self.buf.clear()

    def add(self, probs: np.ndarray):
        if probs is not None:
            if self.buf:
                prev_top = np.argmax(np.mean(np.stack(self.buf, 0), axis=0))
                new_top  = np.argmax(probs)
                if new_top != prev_top and probs[new_top] > 0.7:
                    self.buf.clear()
            self.buf.append(probs)

    def current(self) -> dict:
        if not self.buf:
            return {"gesture": None, "confidence": 0.0, "second": None,
                    "second_conf": 0.0, "gesture_idx": 0}
        frames  = np.stack(self.buf, 0)
        weights = frames.max(axis=1) ** 2          # confidence-squared weighting
        weights /= weights.sum()
        avg  = (frames * weights[:, None]).sum(0)
        top2 = np.argsort(avg)[::-1][:2]
        return {
            "gesture":     self.class_names[top2[0]],
            "confidence":  float(avg[top2[0]]),
            "second":      self.class_names[top2[1]],
            "second_conf": float(avg[top2[1]]),
            "gesture_idx": int(top2[0]),
        }
