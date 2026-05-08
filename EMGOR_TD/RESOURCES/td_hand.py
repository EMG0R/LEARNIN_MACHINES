"""EMGOR_TD — Hand mesh Script TOP callback.

Inputs : webcam (or any RGB) TOP on input 0.
Outputs: RGBA float32 — rendered mesh on transparent background. NO webcam pixels.
         Composite over the original webcam in TD.

Side-effect: pushes gesture classification to an OSC Out DAT named `oscout1`
inside the same COMP, address `/emgor/hand`. Also writes the latest result to
scriptOp.storage['classification'].

Path resolution + venv bootstrap is fully relative — works on any machine
as long as EMGOR_TD/ is intact. Just run SETUP.command once.
"""
import os
import sys
from pathlib import Path


def _find_resources():
    """Locate EMGOR_TD/RESOURCES/ wherever the user put it. No hardcoded paths."""
    candidates = []
    try:
        proj = Path(project.folder)                            # noqa: F821
        candidates += [
            proj / "RESOURCES",                # .toe lives INSIDE EMGOR_TD/
            proj / "EMGOR_TD" / "RESOURCES",   # .toe sits NEXT TO EMGOR_TD/
        ]
    except Exception:
        pass
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    if os.environ.get("EMGOR_RESOURCES"):
        candidates.append(Path(os.environ["EMGOR_RESOURCES"]))
    for c in candidates:
        if c.exists() and (c / "lib").exists():
            return c
    return Path.cwd() / "RESOURCES"


_HERE = _find_resources()
_LIB  = _HERE / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

# Auto-bootstrap the bundled venv into sys.path so users don't have to set
# TD's "Python 64-bit Module Path" preference. Tries all python3.X folders.
_VENV = _HERE.parent / "venv"
for _site in _VENV.glob("lib/python*/site-packages"):
    if str(_site) not in sys.path:
        sys.path.insert(0, str(_site))

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from combined_app.models import (
    load_all_models, device,
    postprocess_hand_mask, hand_present,
    run_gesture, GestureSmoother,
)
from combined_app.renderer import draw_mesh
from combined_app.config import CONF_THRESHOLD, GESTURE_COLORS_BGR

N_ZONES = 3
MESH_LINES_ONLY = False
OSC_ADDRESS      = "/emgor/hand"
OSC_DAT_NAME     = "oscout1"
INFERENCE_EVERY  = 3   # run ML every N cooks; intermediate cooks re-output cached frame


def _state(scriptOp):
    st = scriptOp.fetch('state', None, search=False)
    if st is None:
        models = load_all_models(face=False)
        st = {
            'seg':           models['seg'],
            'gest':          models['gest'],
            'class_names':   models['class_names'],
            'smoother':      GestureSmoother(models['class_names']),
            'prob_emas':     [None] * N_ZONES,
            'feedback_bufs': {},
            'frame_idx':     0,
            'cached_rgba':   None,
        }
        scriptOp.store('state', st)
    return st


def _bgr_uint8_from_input(scriptOp):
    arr = scriptOp.inputs[0].numpyArray(delayed=False)   # HxWx4 float32 0..1
    rgb = (arr[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _hand_pipeline(frame_bgr, st):
    h, w = frame_bgr.shape[:2]
    s = h
    x_off = [0, (w - s) // 2, w - s]
    crops = [frame_bgr[:, x:x + s] for x in x_off]
    sz = st['seg']._img_size

    tensors = [TF.normalize(TF.to_tensor(
                   Image.fromarray(c[:, :, ::-1]).resize((sz, sz), Image.BILINEAR)),
                   st['seg']._mean, st['seg']._std)
               for c in crops]
    with torch.inference_mode():
        logits = st['seg'](torch.stack(tensors).to(device))
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)

    zone_masks = []
    for i, (p, x) in enumerate(zip(probs, x_off)):
        p_rs = cv2.resize(p, (s, s), interpolation=cv2.INTER_LINEAR)
        prev = st['prob_emas'][i]
        if prev is not None and prev.shape != p_rs.shape:
            prev = None   # reset EMA on resolution change
        m, st['prob_emas'][i] = postprocess_hand_mask(p_rs, prev)
        zone_masks.append((m, x))

    best_idx = max(range(N_ZONES), key=lambda i: np.count_nonzero(zone_masks[i][0]))
    best_m   = zone_masks[best_idx][0]
    full = np.zeros((h, w), dtype=np.uint8)
    for m, x in zone_masks:
        full[:, x:x + s] = np.maximum(full[:, x:x + s], m)
    return full, crops[best_idx], best_m


def _send_osc(scriptOp, gesture, conf, gidx):
    osc = scriptOp.parent().op(OSC_DAT_NAME) if hasattr(scriptOp, 'parent') else None
    if osc is None:
        return
    try:
        osc.sendOSC(OSC_ADDRESS, [str(gesture or "none"), float(conf), int(gidx)])
    except Exception:
        pass


def onCook(scriptOp):
    if not scriptOp.inputs:
        scriptOp.copyNumpyArray(np.zeros((1, 1, 4), dtype=np.float32))
        return

    st = _state(scriptOp)
    st.setdefault('frame_idx', 0)
    st.setdefault('cached_rgba', None)

    # Throttle: only run ML every INFERENCE_EVERY cooks. Between runs, re-emit cached frame.
    st['frame_idx'] += 1
    if st['cached_rgba'] is not None and (st['frame_idx'] % INFERENCE_EVERY) != 0:
        scriptOp.copyNumpyArray(st['cached_rgba'])
        return

    frame_bgr = _bgr_uint8_from_input(scriptOp)

    mask, best_crop, best_mask = _hand_pipeline(frame_bgr, st)

    gesture, conf, gidx = None, 0.0, 0
    if hand_present(mask):
        raw = run_gesture(best_crop, best_mask, st['gest'], st['class_names'])
        st['smoother'].add(raw["probs"])
    else:
        st['smoother'].reset()
    g = st['smoother'].current()
    if g["confidence"] >= CONF_THRESHOLD:
        gesture, conf, gidx = g["gesture"], g["confidence"], g["gesture_idx"]

    color = GESTURE_COLORS_BGR[gidx % len(GESTURE_COLORS_BGR)] if gesture else (60, 200, 255)
    canvas = np.zeros_like(frame_bgr)
    if hand_present(mask):
        canvas = draw_mesh(canvas, mask, color, 1.0,
                           st['feedback_bufs'], 'hand',
                           lines_only=MESH_LINES_ONLY)

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = rgb.max(axis=2, keepdims=True)
    rgba = np.concatenate([rgb, alpha], axis=2)
    st['cached_rgba'] = rgba
    scriptOp.copyNumpyArray(rgba)

    scriptOp.store('classification', {
        "gesture": gesture, "confidence": float(conf), "gesture_idx": int(gidx),
    })
    _send_osc(scriptOp, gesture, conf, gidx)


def onSetupParameters(scriptOp): return
def onPulse(par): return
