"""OBJECTIFICATION (Layer 1) model wrapper for the AI Mirror.

Loads ObjSegNet from OBJECTIFICATION/seg/model.py. Returns None gracefully
if the module or checkpoint is unavailable — the cascade skips Layer 1 in
that case, falling back to the existing hand+face pipeline.

Inference improvements over the naive single-frame argmax (helpful on
small/oblique objects like a phone held close to camera):

  1. **Multi-crop**: 3 square crops (left / center / right) at native
     720x720 → model 320x320 each. Objects on the sides get to occupy
     a larger fraction of the model's view, recovering small-object
     recall lost to the 1280x720 → 320 squash. Mirrors HAND_JOB's
     hand_seg multi-zone strategy.
  2. **Temporal EMA on probability maps**: smooth softmax across frames.
     A consistent ~0.3 prob beats a spiky 0.6/0.1/0.4 single-frame trace.
  3. **Min-prob threshold**: pixels whose top class prob is below
     OBJ_MIN_PROB stay background. Avoids noisy half-confidence flickers.
"""
import sys
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image

# TD bundle layout: RESOURCES/lib/{combined_app,seg}/  — seg/ is already on
# sys.path via td_*.py bootstrap, so we can import seg directly. No external
# OBJECTIFICATION/ folder required.
_LIB = str(Path(__file__).resolve().parent.parent)
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from combined_app.config import OBJ_CKPT, OBJ_SIZE, PERSON_MIN_AREA

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
OBJ_MIN_PROB   = 0.35   # pixel must be at least this confident to leave background
OBJ_EMA_ALPHA  = 0.45   # mix of new prob into EMA: out = a*new + (1-a)*old

try:
    from seg.model import ObjSegNet as _ObjSegNet
    from seg.classes import CLASS_NAMES as _CLASS_NAMES
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False

_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Module-level state for temporal EMA (one prob map per crop).
_prob_ema_state: list = [None, None, None]


def load_objectification():
    """Load ObjSegNet from checkpoint. Returns model or None if unavailable."""
    if not _IMPORT_OK:
        print("[objectification] OBJECTIFICATION module not importable — skipping Layer 1")
        return None
    if not Path(OBJ_CKPT).exists():
        print(f"[objectification] No checkpoint at {OBJ_CKPT} — skipping Layer 1")
        return None
    try:
        ckpt = torch.load(OBJ_CKPT, map_location="cpu", weights_only=False)
        num_classes = ckpt.get("num_classes", 24)
        model = _ObjSegNet(num_classes=num_classes)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(_device).eval()
        model._img_size = ckpt.get("img_size", OBJ_SIZE)
        model._num_classes = num_classes
        # reset EMA state on every fresh load
        global _prob_ema_state
        _prob_ema_state = [None, None, None]
        # Pre-warm: first MPS forward triggers Metal compile (~500ms-1s on M4).
        # Doing it now means the first live frame doesn't stall.
        with torch.inference_mode():
            dummy = torch.zeros(3, 3, model._img_size, model._img_size, device=_device)
            _ = model(dummy)
        print(f"[objectification] Loaded + pre-warmed ObjSegNet ({num_classes} classes)")
        return model
    except Exception as e:
        print(f"[objectification] Failed to load: {e} — skipping Layer 1")
        return None


def _crop_offsets(w: int, h: int) -> list:
    """3 square crops covering the full width: left / center / right."""
    s = h
    return [(0, 0, s), ((w - s) // 2, 0, s), (w - s, 0, s)]


@torch.inference_mode()
def run_objectification(frame_bgr: np.ndarray, model) -> dict:
    """Run Layer 1 inference on a full frame.

    Returns dict with keys:
        enabled (bool)
        class_map (dict[str, np.ndarray]):  class_name -> binary uint8 HxW mask
        person_present (bool)
        active_classes (list[str])
    """
    if model is None:
        return {"enabled": False, "class_map": {}, "person_present": False, "active_classes": []}

    h, w = frame_bgr.shape[:2]
    sz = model._img_size
    C = model._num_classes
    offsets = _crop_offsets(w, h)

    # Build a batch of 3 model-resolution crops.
    tensors = []
    for (x, y, s) in offsets:
        crop = frame_bgr[y:y+s, x:x+s]
        pil = Image.fromarray(crop[:, :, ::-1]).resize((sz, sz), Image.BILINEAR)
        tensors.append(TF.normalize(TF.to_tensor(pil), _IMAGENET_MEAN, _IMAGENET_STD))
    batch = torch.stack(tensors).to(_device)            # (3, 3, sz, sz)
    logits = model(batch)                                # (3, C, sz, sz)
    probs_batch = torch.softmax(logits, dim=1).cpu().numpy()  # (3, C, sz, sz)

    # EMA per crop. Since crop offsets are fixed by frame size, indices map 1:1.
    global _prob_ema_state
    eps_a = OBJ_EMA_ALPHA
    smoothed = []
    for i in range(3):
        cur = probs_batch[i]
        prev = _prob_ema_state[i]
        if prev is None or prev.shape != cur.shape:
            ema = cur
        else:
            ema = eps_a * cur + (1.0 - eps_a) * prev
        _prob_ema_state[i] = ema
        smoothed.append(ema)

    # Stitch crops back into a full-frame class map. Where crops overlap
    # (center crop overlaps left & right) we pick the prediction with the
    # strongest top-class probability — preserves small-object detections.
    full_top_p = np.zeros((h, w), dtype=np.float32)
    full_cls   = np.zeros((h, w), dtype=np.uint8)

    for (x, y, s), ema in zip(offsets, smoothed):
        # Resize ema (C, sz, sz) -> (C, s, s)
        # cv2.resize wants HxWxC layout
        ema_hw = ema.transpose(1, 2, 0)                                  # (sz, sz, C)
        ema_full = cv2.resize(ema_hw, (s, s), interpolation=cv2.INTER_LINEAR)
        if ema_full.ndim == 2:                                           # safety for C==1
            ema_full = ema_full[..., None]
        top_c = ema_full.argmax(axis=2).astype(np.uint8)                 # (s, s)
        top_p = ema_full.max(axis=2).astype(np.float32)                  # (s, s)
        # Clear background where confidence is too low
        top_c[top_p < OBJ_MIN_PROB] = 0
        # Merge into full canvas: the prediction with higher top_p wins on overlap
        canvas_top_p = full_top_p[y:y+s, x:x+s]
        canvas_cls   = full_cls  [y:y+s, x:x+s]
        update = top_p > canvas_top_p
        canvas_top_p[update] = top_p[update]
        canvas_cls  [update] = top_c[update]

    # Build per-class binary masks
    masks = {}
    active = []
    for cls_id, cls_name in enumerate(_CLASS_NAMES):
        if cls_id == 0:
            continue
        m = (full_cls == cls_id).astype(np.uint8) * 255
        if m.any():
            masks[cls_name] = m
            active.append(cls_name)

    person_mask = masks.get("person")
    person_present = (person_mask is not None and
                      np.count_nonzero(person_mask) / (h * w) >= PERSON_MIN_AREA)

    return {
        "enabled": True,
        "class_map": masks,
        "person_present": person_present,
        "active_classes": active,
    }


# ── Object-on-object recursion ────────────────────────────────────────────────
# Carrier classes: things people hold up that often display other objects on
# their surface (a phone with a photo of a cup, a laptop showing a guitar).
# When the model detects one of these covering >REC_AREA_FRAC of the frame,
# we crop to its bbox and re-run inference at full model resolution. The
# secondary inference's view is dominated by whatever's ON the carrier.
RECURSE_CARRIERS    = {"phone", "device"}
REC_AREA_FRAC       = 0.05    # carrier must be at least 5% of frame
REC_TOP_K           = 2       # at most this many carriers per pass
REC_PAD_FRAC        = 0.05    # bbox padding before zooming
REC_MIN_PROB        = 0.45    # secondary needs higher confidence (less noise)
REC_INSET_AREA      = 600     # min area for a secondary blob to render


def _bbox_of_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@torch.inference_mode()
def run_objectification_recurse(frame_bgr: np.ndarray, model, primary: dict) -> list:
    """Detect objects-ON-objects via zoom-in inference on carrier bboxes.

    `primary` is the dict from run_objectification(). Returns a list of
    secondary detection dicts, each:
        {"carrier": <carrier class name>,
         "bbox":    (x1, y1, x2, y2)  in frame coords,
         "class_map": dict[name -> binary HxW mask] of NEW classes detected
                     inside the carrier (excludes the carrier's own class
                     and 'person' which is already known globally)}
    """
    if model is None or not primary.get("enabled"):
        return []
    h, w = frame_bgr.shape[:2]
    sz   = model._img_size
    frame_area = h * w
    out = []

    # Find big-enough carrier blobs
    carriers = []
    for cname in RECURSE_CARRIERS:
        m = primary["class_map"].get(cname)
        if m is None:
            continue
        # Largest connected component of this carrier class
        bb = _bbox_of_mask(m)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        area = (x2 - x1) * (y2 - y1)
        if area / frame_area < REC_AREA_FRAC:
            continue
        carriers.append((area, cname, (x1, y1, x2, y2)))
    if not carriers:
        return []
    carriers.sort(reverse=True)
    carriers = carriers[:REC_TOP_K]

    # Batch the zoom-in inferences
    tensors = []
    metas   = []
    for area, cname, (x1, y1, x2, y2) in carriers:
        bw = x2 - x1; bh = y2 - y1
        px = int(bw * REC_PAD_FRAC); py = int(bh * REC_PAD_FRAC)
        cx1 = max(0, x1 - px); cy1 = max(0, y1 - py)
        cx2 = min(w, x2 + px); cy2 = min(h, y2 + py)
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        pil = Image.fromarray(crop[:, :, ::-1]).resize((sz, sz), Image.BILINEAR)
        tensors.append(TF.normalize(TF.to_tensor(pil), _IMAGENET_MEAN, _IMAGENET_STD))
        metas.append((cname, cx1, cy1, cx2, cy2))
    if not tensors:
        return []

    batch  = torch.stack(tensors).to(_device)
    logits = model(batch)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()    # (N, C, sz, sz)

    for i, (cname, cx1, cy1, cx2, cy2) in enumerate(metas):
        ema_hw = probs[i].transpose(1, 2, 0)
        ch = cy2 - cy1; cw = cx2 - cx1
        ema_full = cv2.resize(ema_hw, (cw, ch), interpolation=cv2.INTER_LINEAR)
        top_c = ema_full.argmax(axis=2).astype(np.uint8)
        top_p = ema_full.max(axis=2).astype(np.float32)
        top_c[top_p < REC_MIN_PROB] = 0
        # Build per-class masks placed back into a full-frame canvas, but
        # ONLY for classes that are different from the carrier itself + person
        sec_masks = {}
        for cls_id, cls_name in enumerate(_CLASS_NAMES):
            if cls_id == 0 or cls_name == cname or cls_name == "person":
                continue
            crop_mask = (top_c == cls_id).astype(np.uint8) * 255
            if crop_mask.sum() // 255 < REC_INSET_AREA:
                continue
            full = np.zeros((h, w), dtype=np.uint8)
            full[cy1:cy2, cx1:cx2] = crop_mask
            sec_masks[cls_name] = full
        if sec_masks:
            out.append({"carrier": cname, "bbox": (cx1, cy1, cx2, cy2),
                        "class_map": sec_masks})
    return out
