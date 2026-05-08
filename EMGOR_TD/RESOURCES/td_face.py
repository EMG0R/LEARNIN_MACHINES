"""EMGOR_TD — Face mesh Script TOP callback.

Inputs : webcam (or any RGB) TOP on input 0.
Outputs: RGBA float32 — face mesh + eye/mouth wireframes on transparent BG.
         NO webcam pixels. Composite over the webcam in TD.

Side-effect: sends emotion classification to an OSC Out DAT named `oscout1`
inside the same COMP, address `/emgor/face`. Also writes latest result to
scriptOp.storage['classification'].
"""
import os
import sys
from pathlib import Path


def _find_resources():
    candidates = []
    try:
        proj = Path(project.folder)                            # noqa: F821
        candidates += [
            proj / "RESOURCES",
            proj / "EMGOR_TD" / "RESOURCES",
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

_VENV = _HERE.parent / "venv"
for _site in _VENV.glob("lib/python*/site-packages"):
    if str(_site) not in sys.path:
        sys.path.insert(0, str(_site))

import numpy as np
import cv2

from combined_app.models import (
    load_all_models, run_face_det, run_face_parts,
    run_emotion_batch, EmotionSmoother,
)
from combined_app.renderer import draw_mesh
from combined_app.config import (
    FACE_CLASS_SKIN, FACE_CLASS_EYE_L, FACE_CLASS_EYE_R, FACE_CLASS_MOUTH,
    FACE_CONF_THR,
)

FACE_DET_INTERVAL = 8
PARTS_INTERVAL    = 2
OSC_ADDRESS       = "/emgor/face"
OSC_DAT_NAME      = "oscout1"
INFERENCE_EVERY   = 3   # run ML every N cooks; intermediate cooks re-output cached frame


def _state(scriptOp):
    st = scriptOp.fetch('state', None, search=False)
    if st is None:
        models = load_all_models(face=True)
        st = {
            'fd': models['fd'], 'fp': models['fp'], 'em': models['em'],
            'em_smoother':   EmotionSmoother(models['em']._class_names),
            'face_box':      None,
            'parts_cache':   None,
            'parts_counter': 0,
            'det_countdown': 0,
            'feedback_bufs': {},
            'frame_idx':     0,
            'cached_rgba':   None,
        }
        scriptOp.store('state', st)
    return st


def _bgr_uint8_from_input(scriptOp):
    arr = scriptOp.inputs[0].numpyArray(delayed=False)
    rgb = (arr[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _face_mesh_data(parts_map, box, fh, fw):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    def to_full(cls_ids):
        region = np.zeros_like(parts_map, dtype=np.uint8)
        for c in cls_ids:
            region[parts_map == c] = 255
        full = np.zeros((fh, fw), dtype=np.uint8)
        full[y1:y2, x1:x2] = cv2.resize(region, (bw, bh), interpolation=cv2.INTER_NEAREST)
        return full

    return to_full([FACE_CLASS_SKIN]), {
        'eye_l': to_full([FACE_CLASS_EYE_L]),
        'eye_r': to_full([FACE_CLASS_EYE_R]),
        'mouth': to_full([FACE_CLASS_MOUTH]),
    }


def _send_osc(scriptOp, emotion, conf):
    osc = scriptOp.parent().op(OSC_DAT_NAME) if hasattr(scriptOp, 'parent') else None
    if osc is None:
        return
    try:
        osc.sendOSC(OSC_ADDRESS, [str(emotion or "none"), float(conf)])
    except Exception:
        pass


def onCook(scriptOp):
    if not scriptOp.inputs:
        scriptOp.copyNumpyArray(np.zeros((1, 1, 4), dtype=np.float32))
        return

    st = _state(scriptOp)
    st.setdefault('frame_idx', 0)
    st.setdefault('cached_rgba', None)

    st['frame_idx'] += 1
    if st['cached_rgba'] is not None and (st['frame_idx'] % INFERENCE_EVERY) != 0:
        scriptOp.copyNumpyArray(st['cached_rgba'])
        return

    frame = _bgr_uint8_from_input(scriptOp)
    h, w = frame.shape[:2]

    st['det_countdown'] -= 1
    if st['det_countdown'] <= 0:
        st['det_countdown'] = FACE_DET_INTERVAL
        faces = run_face_det(frame, st['fd'])
        st['face_box'] = faces[0][0] if faces else None

    skin_mask, region_masks = None, {}
    emotion, conf = None, 0.0

    if st['face_box'] is not None:
        x1, y1, x2, y2 = st['face_box']
        pad_x = int((x2 - x1) * 0.20); pad_y = int((y2 - y1) * 0.25)
        cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x); cy2 = min(h, y2 + pad_y)
        face_crop = frame[cy1:cy2, cx1:cx2]
        if face_crop.size > 0:
            st['parts_counter'] += 1
            if st['parts_counter'] >= PARTS_INTERVAL or st['parts_cache'] is None:
                st['parts_counter'] = 0
                parts_map = run_face_parts(face_crop, st['fp'])
                skin_mask, region_masks = _face_mesh_data(parts_map, (cx1, cy1, cx2, cy2), h, w)
                dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                skin_mask = cv2.dilate(skin_mask, dil, iterations=1)
                st['parts_cache'] = (skin_mask, region_masks)
            else:
                skin_mask, region_masks = st['parts_cache']

            em_raw = run_emotion_batch([frame[y1:y2, x1:x2]], st['em'])
            if em_raw:
                st['em_smoother'].add(em_raw[0]["probs"])
        else:
            st['em_smoother'].reset()
    else:
        st['em_smoother'].reset()

    em = st['em_smoother'].current()
    if em["confidence"] >= FACE_CONF_THR:
        emotion, conf = em["emotion"], em["confidence"]

    canvas = np.zeros_like(frame)
    if skin_mask is not None and skin_mask.any():
        canvas = draw_mesh(canvas, skin_mask, (255, 255, 255), 0.38,
                           st['feedback_bufs'], 'face_skin',
                           lines_only=True, n_contour=20, n_interior=10,
                           pts_update=0, stable_interior=True, base_w_mult=1.2)
        for rid, rmask in region_masks.items():
            if rmask.any():
                canvas = draw_mesh(canvas, rmask, (255, 255, 255), 0.65,
                                   st['feedback_bufs'], f'face_{rid}',
                                   lines_only=True, n_contour=8, n_interior=4,
                                   pts_update=0)

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = rgb.max(axis=2, keepdims=True)
    rgba = np.concatenate([rgb, alpha], axis=2)
    st['cached_rgba'] = rgba
    scriptOp.copyNumpyArray(rgba)

    scriptOp.store('classification', {
        "emotion": emotion, "confidence": float(conf),
    })
    _send_osc(scriptOp, emotion, conf)


def onSetupParameters(scriptOp): return
def onPulse(par): return
