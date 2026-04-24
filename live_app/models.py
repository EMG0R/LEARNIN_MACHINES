import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from collections import deque

_FACE_JOB = str(Path(__file__).parent.parent / "FACE_JOB")
if _FACE_JOB not in sys.path:
    sys.path.insert(0, _FACE_JOB)
from face_det.postprocess import decode, nms
from face_det.model import FaceDetector as _RealFaceDetector
from face_parts.model import FacePartsUNet as _RealFacePartsUNet
from emotion.model import EmotionWide as _RealEmotionWide

from live_app.config import (
    HAND_SEG_CKPT, GESTURE_CKPT, FACE_DET_CKPT, FACE_PARTS_CKPT, EMOTION_CKPT,
    SEG_THRESHOLD, HAND_MIN_AREA, MASK_EMA_ALPHA, MIN_CC_AREA_PX, VOTE_WINDOW,
    FACE_DET_SCORE_THR, FACE_DET_IOU_THR, FACE_DET_IMG,
    FACE_EMA_ALPHA, MAX_FACES,
    FACE_CLASS_SKIN, FACE_CLASS_EYE_L, FACE_CLASS_EYE_R, FACE_CLASS_MOUTH,
    EMOTION_VALENCE, GESTURE_VALENCE,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Model architectures ───────────────────────────────────────────────────────

def _cb(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )

class _HandUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1=_cb(3,32); self.d2=_cb(32,64); self.d3=_cb(64,128); self.d4=_cb(128,256)
        self.pool=nn.MaxPool2d(2)
        self.up3=nn.ConvTranspose2d(256,128,2,2); self.u3=_cb(256,128)
        self.up2=nn.ConvTranspose2d(128,64,2,2);  self.u2=_cb(128,64)
        self.up1=nn.ConvTranspose2d(64,32,2,2);   self.u1=_cb(64,32)
        self.out=nn.Conv2d(32,1,1)
    def forward(self,x):
        c1=self.d1(x); c2=self.d2(self.pool(c1)); c3=self.d3(self.pool(c2)); c4=self.d4(self.pool(c3))
        u3=self.u3(torch.cat([self.up3(c4),c3],1)); u2=self.u2(torch.cat([self.up2(u3),c2],1))
        return self.out(self.u1(torch.cat([self.up1(u2),c1],1)))

class _Wide(nn.Module):
    def __init__(self, n):
        super().__init__()
        def block(ci,co):
            return nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.BatchNorm2d(co),nn.ReLU(),
                                 nn.Conv2d(co,co,3,padding=1),nn.BatchNorm2d(co),nn.ReLU(),
                                 nn.Dropout2d(0.1),nn.MaxPool2d(2))
        self.features=nn.Sequential(block(3,32),block(32,64),block(64,128),block(128,256))
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),
                                nn.Linear(256,512),nn.ReLU(),nn.Dropout(0.4),nn.Linear(512,n))
    def forward(self,x): return self.head(self.features(x))


# ── Load ──────────────────────────────────────────────────────────────────────

def load_all_models():
    print("[models] loading hand seg...")
    seg_ckpt = torch.load(HAND_SEG_CKPT, map_location="cpu", weights_only=False)
    seg = _HandUNet()
    seg.load_state_dict(seg_ckpt["model_state_dict"], strict=False)
    seg.to(device).eval()
    seg._img_size = seg_ckpt["img_size"]
    seg._mean = seg_ckpt["normalize_mean"]
    seg._std  = seg_ckpt["normalize_std"]

    print("[models] loading gesture...")
    g_ckpt = torch.load(GESTURE_CKPT, map_location="cpu", weights_only=False)
    class_names = g_ckpt["class_names"]
    gest = _Wide(n=len(class_names))
    gest.load_state_dict(g_ckpt["model_state_dict"])
    gest.to(device).eval()
    gest._img_size = g_ckpt["img_size"]
    gest._mean = g_ckpt["normalize_mean"]
    gest._std  = g_ckpt["normalize_std"]

    print("[models] loading face detector...")
    fd_ckpt = torch.load(FACE_DET_CKPT, map_location="cpu", weights_only=False)
    fd = _RealFaceDetector()
    fd.load_state_dict(fd_ckpt["model_state_dict"], strict=False)
    fd.to(device).eval()
    fd._img_size = fd_ckpt.get("img_size", FACE_DET_IMG)
    fd._mean = fd_ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
    fd._std  = fd_ckpt.get("normalize_std",  [0.229, 0.224, 0.225])

    print("[models] loading face parts...")
    fp_ckpt = torch.load(FACE_PARTS_CKPT, map_location="cpu", weights_only=False)
    fp = _RealFacePartsUNet()
    fp.load_state_dict(fp_ckpt["model_state_dict"], strict=False)
    fp.to(device).eval()
    fp._img_size = fp_ckpt.get("img_size", 192)
    fp._mean = fp_ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
    fp._std  = fp_ckpt.get("normalize_std",  [0.229, 0.224, 0.225])

    print("[models] loading emotion...")
    em_ckpt = torch.load(EMOTION_CKPT, map_location="cpu", weights_only=False)
    em = _RealEmotionWide()
    em.load_state_dict(em_ckpt["model_state_dict"])
    em.to(device).eval()
    em._img_size = em_ckpt.get("img_size", 64)
    em._mean = em_ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
    em._std  = em_ckpt.get("normalize_std",  [0.229, 0.224, 0.225])
    em._class_names = em_ckpt.get("class_names", ["happy","sad","neutral","surprise","anger","fear","disgust"])

    return dict(seg=seg, gest=gest, class_names=class_names,
                fd=fd, fp=fp, em=em)


# ── Hand inference ────────────────────────────────────────────────────────────

def run_seg_prob_batch(crops, seg_model):
    sz = seg_model._img_size
    tensors, shapes = [], []
    for crop in crops:
        h, w = crop.shape[:2]; shapes.append((w, h))
        pil = Image.fromarray(crop[:, :, ::-1]).resize((sz, sz), Image.BILINEAR)
        tensors.append(TF.normalize(TF.to_tensor(pil), seg_model._mean, seg_model._std))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        logits = seg_model(batch)
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
    return [cv2.resize(probs[i], shapes[i], interpolation=cv2.INTER_LINEAR) for i in range(len(crops))]


def postprocess_hand_mask(prob, prev_ema):
    ema = prob if prev_ema is None else MASK_EMA_ALPHA * prob + (1 - MASK_EMA_ALPHA) * prev_ema
    binary = (ema > SEG_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_CC_AREA_PX:
            mask[labels == i] = 1
    return mask * 255, ema


def hand_present(mask):
    if mask is None: return False
    h, w = mask.shape
    return (np.count_nonzero(mask) / (h * w)) >= HAND_MIN_AREA


def run_gesture(frame_bgr, mask, gest_model, class_names):
    empty = {"gesture": None, "confidence": 0.0, "second": None,
             "second_conf": 0.0, "gesture_idx": 0, "probs": None}
    coords = cv2.findNonZero(mask)
    if coords is None: return empty
    x, y, bw, bh = cv2.boundingRect(coords)
    pad = int(max(bw, bh) * 0.15)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(frame_bgr.shape[1], x + bw + pad); y1 = min(frame_bgr.shape[0], y + bh + pad)
    if x1 <= x0 or y1 <= y0: return empty
    pil = Image.fromarray(frame_bgr[y0:y1, x0:x1, ::-1])
    pil = pil.resize((gest_model._img_size, gest_model._img_size), Image.BILINEAR)
    t = TF.normalize(TF.to_tensor(pil), gest_model._mean, gest_model._std)
    with torch.no_grad():
        logits = gest_model(t.unsqueeze(0).to(device))
    probs = torch.softmax(logits.squeeze(), dim=0).cpu().numpy()
    if "two_up_inverted" in class_names and "middle_finger" in class_names:
        probs[class_names.index("middle_finger")] += probs[class_names.index("two_up_inverted")]
        probs[class_names.index("two_up_inverted")] = 0.0
    top2 = np.argsort(probs)[::-1][:2]
    return {"gesture": class_names[top2[0]], "confidence": float(probs[top2[0]]),
            "second": class_names[top2[1]], "second_conf": float(probs[top2[1]]),
            "gesture_idx": int(top2[0]), "probs": probs.astype(np.float32)}


class GestureSmoother:
    def __init__(self, class_names, window=VOTE_WINDOW):
        self.class_names = class_names
        self.window = window
        self.buf = deque(maxlen=window)

    def reset(self): self.buf.clear()

    def add(self, probs):
        if probs is None: return
        if self.buf:
            prev_top = np.argmax(np.mean(np.stack(self.buf, 0), 0))
            new_top  = np.argmax(probs)
            if new_top != prev_top and probs[new_top] > 0.7:
                self.buf.clear()
        self.buf.append(probs)

    def current(self):
        if not self.buf:
            return {"gesture": None, "confidence": 0.0, "second": None,
                    "second_conf": 0.0, "gesture_idx": 0}
        frames  = np.stack(self.buf, 0)
        weights = frames.max(axis=1) ** 2
        weights /= weights.sum()
        avg = (frames * weights[:, None]).sum(0)
        top2 = np.argsort(avg)[::-1][:2]
        return {"gesture": self.class_names[top2[0]], "confidence": float(avg[top2[0]]),
                "second": self.class_names[top2[1]], "second_conf": float(avg[top2[1]]),
                "gesture_idx": int(top2[0])}


def gesture_valence(result):
    g = result.get("gesture") or ""
    v = GESTURE_VALENCE.get(g, 0.0)
    return v * min(1.0, result.get("confidence", 0.0) * 2.0)


# ── Face inference ────────────────────────────────────────────────────────────

def run_face_det(frame_bgr, fd_model):
    h, w = frame_bgr.shape[:2]
    sz = fd_model._img_size
    pil = Image.fromarray(frame_bgr[:, :, ::-1]).resize((sz, sz), Image.BILINEAR)
    t = TF.normalize(TF.to_tensor(pil), fd_model._mean, fd_model._std)
    with torch.no_grad():
        obj, bbox, ctr = fd_model(t.unsqueeze(0).to(device))
    results = decode(obj, bbox, ctr, stride=8, score_thr=FACE_DET_SCORE_THR)
    boxes, scores = results[0]
    if boxes.numel() == 0:
        return []
    keep = nms(boxes, scores, iou_thr=FACE_DET_IOU_THR)
    boxes  = boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    sx, sy = w / sz, h / sz
    faces = []
    for box, score in zip(boxes, scores):
        x1 = int(max(0, box[0] * sx)); y1 = int(max(0, box[1] * sy))
        x2 = int(min(w, box[2] * sx)); y2 = int(min(h, box[3] * sy))
        if x2 > x1 + 8 and y2 > y1 + 8:
            faces.append(((x1, y1, x2, y2), float(score)))
    # sort left→right for stable slot assignment
    faces.sort(key=lambda f: f[0][0])
    return faces[:MAX_FACES]


def run_face_parts(face_crop_bgr, fp_model):
    """Returns (H, W) uint8 class-index map at crop resolution."""
    h, w = face_crop_bgr.shape[:2]
    sz = fp_model._img_size
    pil = Image.fromarray(face_crop_bgr[:, :, ::-1]).resize((sz, sz), Image.BILINEAR)
    t = TF.normalize(TF.to_tensor(pil), fp_model._mean, fp_model._std)
    with torch.no_grad():
        logits = fp_model(t.unsqueeze(0).to(device))
    pred = logits.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def run_emotion_batch(face_crops, em_model):
    """Batch emotion inference. Returns list of result dicts."""
    if not face_crops:
        return []
    sz = em_model._img_size
    tensors = []
    for crop in face_crops:
        pil = Image.fromarray(crop[:, :, ::-1]).resize((sz, sz), Image.BILINEAR)
        tensors.append(TF.normalize(TF.to_tensor(pil), em_model._mean, em_model._std))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        logits = em_model(batch)
    all_probs = torch.softmax(logits, dim=1).cpu().numpy()
    results = []
    for probs in all_probs:
        top = int(np.argmax(probs))
        results.append({"emotion": em_model._class_names[top],
                        "confidence": float(probs[top]),
                        "probs": probs.astype(np.float32)})
    return results


class EmotionSmoother:
    def __init__(self, class_names, window=8):
        self.class_names = class_names
        self.window = window
        self.buf = deque(maxlen=window)

    def reset(self): self.buf.clear()

    def add(self, probs):
        if probs is not None:
            self.buf.append(probs)

    def current(self):
        if not self.buf:
            return {"emotion": "neutral", "confidence": 0.0}
        avg = np.mean(np.stack(self.buf, 0), 0)
        top = int(np.argmax(avg))
        return {"emotion": self.class_names[top], "confidence": float(avg[top])}


def emotion_valence(result):
    e = result.get("emotion") or "neutral"
    return EMOTION_VALENCE.get(e, 0.0) * min(1.0, result.get("confidence", 0.0) * 2.0)
