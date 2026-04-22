import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from live_app.config import SEG_CKPT, GESTURE_CKPT, SEG_THRESHOLD

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
    seg_model.load_state_dict(seg_ckpt["model_state_dict"])
    seg_model.to(device).eval()
    seg_model._img_size = seg_ckpt["img_size"]
    seg_model._mean     = seg_ckpt["normalize_mean"]
    seg_model._std      = seg_ckpt["normalize_std"]

    g_ckpt = torch.load(GESTURE_CKPT, map_location="cpu", weights_only=False)
    gest_model = _Wide()
    gest_model.load_state_dict(g_ckpt["model_state_dict"])
    gest_model.to(device).eval()
    gest_model._img_size = g_ckpt["img_size"]
    gest_model._mean     = g_ckpt["normalize_mean"]
    gest_model._std      = g_ckpt["normalize_std"]
    class_names          = g_ckpt["class_names"]

    return seg_model, gest_model, class_names


def run_seg(frame_bgr: np.ndarray, seg_model) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    pil  = Image.fromarray(frame_bgr[:, :, ::-1])
    pil  = pil.resize((seg_model._img_size, seg_model._img_size), Image.BILINEAR)
    t    = TF.normalize(TF.to_tensor(pil), seg_model._mean, seg_model._std)
    with torch.no_grad():
        logit = seg_model(t.unsqueeze(0).to(device))
    prob   = torch.sigmoid(logit).squeeze().cpu().numpy()
    binary = (prob > SEG_THRESHOLD).astype(np.uint8) * 255
    return cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)


def run_gesture(frame_bgr: np.ndarray, mask: np.ndarray,
                gest_model, class_names: list) -> dict:
    empty = {"gesture": None, "confidence": 0.0, "second": None, "second_conf": 0.0, "gesture_idx": 0}
    coords = cv2.findNonZero(mask)
    if coords is None:
        return empty
    x, y, bw, bh = cv2.boundingRect(coords)
    pad = int(max(bw, bh) * 0.10)
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
    top2  = np.argsort(probs)[::-1][:2]
    return {
        "gesture":     class_names[top2[0]],
        "confidence":  float(probs[top2[0]]),
        "second":      class_names[top2[1]],
        "second_conf": float(probs[top2[1]]),
        "gesture_idx": int(top2[0]),
    }
