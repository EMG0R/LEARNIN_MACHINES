"""OBJECTIFICATION (Layer 1) model wrapper for the AI Mirror.

Loads ObjSegNet from OBJECTIFICATION/seg/model.py. Returns None gracefully
if the module or checkpoint is unavailable — the cascade skips Layer 1 in
that case, falling back to the existing hand+face pipeline.
"""
import sys
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image

_OBJ_SEG = str(Path(__file__).parent.parent / "OBJECTIFICATION")
if _OBJ_SEG not in sys.path:
    sys.path.insert(0, _OBJ_SEG)

from combined_app.config import OBJ_CKPT, OBJ_SIZE, PERSON_MIN_AREA

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

try:
    from seg.model import ObjSegNet as _ObjSegNet
    from seg.classes import CLASS_NAMES as _CLASS_NAMES
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False

_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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
        print(f"[objectification] Loaded ObjSegNet ({num_classes} classes) from {OBJ_CKPT}")
        return model
    except Exception as e:
        print(f"[objectification] Failed to load: {e} — skipping Layer 1")
        return None


def run_objectification(frame_bgr: np.ndarray, model) -> dict:
    """Run Layer 1 inference on a full frame.

    Args:
        frame_bgr: HxWx3 uint8 BGR camera frame
        model: loaded ObjSegNet or None

    Returns dict with keys:
        enabled (bool): False if model is None
        class_map (dict[str, np.ndarray]): class_name → binary uint8 HxW mask
        person_present (bool): True if person pixels >= PERSON_MIN_AREA
        active_classes (list[str]): class names with non-zero pixels
    """
    if model is None:
        return {"enabled": False, "class_map": {}, "person_present": False, "active_classes": []}

    h, w = frame_bgr.shape[:2]
    sz = model._img_size

    pil = Image.fromarray(frame_bgr[:, :, ::-1]).resize((sz, sz), Image.BILINEAR)
    t = TF.normalize(TF.to_tensor(pil), _IMAGENET_MEAN, _IMAGENET_STD)
    with torch.no_grad():
        logits = model(t.unsqueeze(0).to(_device))
    class_map_small = logits.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
    class_map_full = cv2.resize(class_map_small, (w, h), interpolation=cv2.INTER_NEAREST)

    masks = {}
    active = []
    for cls_id, cls_name in enumerate(_CLASS_NAMES):
        if cls_id == 0:
            continue  # skip background
        mask = (class_map_full == cls_id).astype(np.uint8) * 255
        if mask.any():
            masks[cls_name] = mask
            active.append(cls_name)

    person_mask = masks.get("person")
    person_present = False
    if person_mask is not None:
        person_present = (np.count_nonzero(person_mask) / (h * w)) >= PERSON_MIN_AREA

    return {
        "enabled": True,
        "class_map": masks,
        "person_present": person_present,
        "active_classes": active,
    }
