"""EMGOR_TD — hand_and_face_job: streamed hand + face Script TOP.

Mirrors combined_app/app.py's threading EXACTLY:

  combined_app:                           hand_and_face_job (this file):
  ─────────────                           ───────────────────────────────
  capture_loop  thread (read webcam)  ≈   onCook reads scriptOp.inputs[0]
  inference_loop thread (heavy ML)    ≈   _Worker thread: ML only, NO render
  main display loop (render @ webcam) ≈   onCook renders mesh @ TD cook rate

Inference is decoupled from rendering. onCook does the cheap stuff (mesh
draw on numpy canvas) at TD's full cook rate. The worker grinds at its own
rate (~6-15 fps) producing masks. This is why combined_app feels smooth —
display runs at 30 fps even when inference is at 5 fps.

NO general image (OBJECTIFICATION) — hand + face only.
NO text labels (mesh only).
NO mirror — output in raw input orientation, do mirror in TD if you want.

Cascade rules (preserved):
  • Padded face bbox subtracts from hand mask before gesture inference.
  • Hand mask subtracts from face mesh.

OSC out (DAT named `oscout1` in same COMP):
  /emgor/hand  [gesture, conf, gesture_idx]
  /emgor/face  [emotion,  conf]
"""
import os
import sys
import time
import threading
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
import torch
import torchvision.transforms.functional as TF

from combined_app.models import (
    load_all_models, device,
    postprocess_hand_mask, hand_present,
    run_gesture, GestureSmoother,
    run_face_det, run_face_parts, run_emotion_batch, EmotionSmoother,
)
from combined_app.renderer import draw_mesh, draw_face_aura
from combined_app.config import (
    CONF_THRESHOLD, FACE_CONF_THR, GESTURE_COLORS_BGR,
    FACE_CLASS_SKIN, FACE_CLASS_EYE_L, FACE_CLASS_EYE_R, FACE_CLASS_MOUTH,
)

# ── tunables ──────────────────────────────────────────────────────────────────
N_ZONES           = 3
FACE_DET_INTERVAL = 10     # face detector throttle (worker iterations)
FACE_PARTS_EVERY  = 2      # face parts within face cycle
FACE_PAD_FRAC     = 0.18   # padding when subtracting face from hand mask
WORKER_FPS_LOG    = 60     # print worker fps every N iterations
WORKER_TARGET_FPS = 12     # cap worker inference rate — sleeps if faster.
                           # 8 fps is plenty for stable mesh (display still
                           # runs at TD's full cook rate). Lower = less GPU.

_EM_ORDER   = ["happy", "sad", "neutral", "surprise", "anger", "fear", "disgust"]
_EM_TO_GIDX = [0, 3, 6, 9, 12, 15, 2]


# ── face mesh helper ──────────────────────────────────────────────────────────

def _face_mesh_data(parts_map, box, fh, fw):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    def _to_full(cls_ids):
        region = np.zeros_like(parts_map, dtype=np.uint8)
        for c in cls_ids:
            region[parts_map == c] = 255
        full = np.zeros((fh, fw), dtype=np.uint8)
        full[y1:y2, x1:x2] = cv2.resize(region, (bw, bh), interpolation=cv2.INTER_NEAREST)
        return full

    return _to_full([FACE_CLASS_SKIN]), {
        "eye_l": _to_full([FACE_CLASS_EYE_L]),
        "eye_r": _to_full([FACE_CLASS_EYE_R]),
        "mouth": _to_full([FACE_CLASS_MOUTH]),
    }


# ── pipelines (worker-side) ───────────────────────────────────────────────────

def _hand_pipeline(frame_bgr, ws, face_box):
    h, w = frame_bgr.shape[:2]
    s = h
    x_off = [0, (w - s) // 2, w - s]
    crops = [frame_bgr[:, x:x + s] for x in x_off]
    sz = ws['seg']._img_size

    tensors = []
    for c in crops:
        small = cv2.resize(c, (sz, sz), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        t = TF.normalize(t, ws['seg']._mean, ws['seg']._std)
        tensors.append(t)
    with torch.inference_mode():
        logits = ws['seg'](torch.stack(tensors).to(device))
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)

    zone_masks = []
    for i, (p, x) in enumerate(zip(probs, x_off)):
        p_rs = cv2.resize(p, (s, s), interpolation=cv2.INTER_LINEAR)
        prev = ws['prob_emas'][i]
        if prev is not None and prev.shape != p_rs.shape:
            prev = None
        m, ws['prob_emas'][i] = postprocess_hand_mask(p_rs, prev)
        zone_masks.append((m, x))

    pfx1 = pfy1 = pfx2 = pfy2 = None
    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        bw = fx2 - fx1; bh = fy2 - fy1
        px = int(bw * FACE_PAD_FRAC); py = int(bh * FACE_PAD_FRAC)
        pfx1 = max(0, fx1 - px); pfy1 = max(0, fy1 - py)
        pfx2 = min(w, fx2 + px); pfy2 = min(h, fy2 + py)
        for i, (m, x) in enumerate(zone_masks):
            lx1 = max(0, pfx1 - x); lx2 = max(0, pfx2 - x)
            if lx2 > lx1:
                m[pfy1:pfy2, lx1:lx2] = 0
            zone_masks[i] = (m, x)

    # Best zone for gesture inference (matches combined_app exactly:
    # picks the zone with the largest hand-pixel count and passes that
    # crop + mask to run_gesture, NOT the full frame).
    best_idx = max(range(N_ZONES), key=lambda i: np.count_nonzero(zone_masks[i][0]))
    best_m   = zone_masks[best_idx][0]
    best_crop = crops[best_idx]

    full = np.zeros((h, w), dtype=np.uint8)
    for m, x in zone_masks:
        full[:, x:x + s] = np.maximum(full[:, x:x + s], m)
    if pfx1 is not None:
        full[pfy1:pfy2, pfx1:pfx2] = 0
    return full, best_crop, best_m


def _face_pipeline(frame_bgr, ws, hand_mask_full):
    if ws['fd'] is None:
        return None, {}, False, {"emotion": "neutral", "confidence": 0.0}

    h, w = frame_bgr.shape[:2]
    ws['det_countdown'] -= 1
    if ws['det_countdown'] <= 0:
        ws['det_countdown'] = FACE_DET_INTERVAL
        faces = run_face_det(frame_bgr, ws['fd'])
        ws['face_box'] = faces[0][0] if faces else None

    skin_mask, region_masks, face_active = None, {}, False
    em_smoother = ws['em_smoother']
    if ws['face_box'] is not None:
        x1, y1, x2, y2 = ws['face_box']
        pad_x = int((x2 - x1) * 0.20); pad_y = int((y2 - y1) * 0.25)
        cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x); cy2 = min(h, y2 + pad_y)
        face_crop = frame_bgr[cy1:cy2, cx1:cx2]
        if face_crop.size > 0:
            face_active = True
            ws['parts_counter'] += 1
            if ws['parts_counter'] >= FACE_PARTS_EVERY or ws['parts_cache'] is None:
                ws['parts_counter'] = 0
                parts_map = run_face_parts(face_crop, ws['fp'])
                skin_mask, region_masks = _face_mesh_data(parts_map, (cx1, cy1, cx2, cy2), h, w)
                dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                skin_mask = cv2.dilate(skin_mask, dil, iterations=1)
                inv_hand = cv2.bitwise_not(hand_mask_full)
                skin_mask = cv2.bitwise_and(skin_mask, inv_hand)
                region_masks = {k: cv2.bitwise_and(m, inv_hand) for k, m in region_masks.items()}
                ws['parts_cache'] = (skin_mask, region_masks)
            else:
                skin_mask, region_masks = ws['parts_cache']
            em_raw = run_emotion_batch([frame_bgr[y1:y2, x1:x2]], ws['em'])
            if em_raw:
                em_smoother.add(em_raw[0]["probs"])
        else:
            em_smoother.reset()
    else:
        em_smoother.reset()
    em = em_smoother.current()
    return skin_mask, region_masks, face_active, em


# ── background worker (INFERENCE ONLY — no render!) ───────────────────────────

class _Worker:
    def __init__(self):
        models = load_all_models(face=True)
        self.ws = {
            'seg':           models['seg'],
            'gest':          models['gest'],
            'class_names':   models['class_names'],
            'fd':            models.get('fd'),
            'fp':            models.get('fp'),
            'em':            models.get('em'),
            'smoother':      GestureSmoother(models['class_names']),
            'em_smoother':   EmotionSmoother(models['em']._class_names) if models.get('em') else None,
            'prob_emas':     [None] * N_ZONES,
            'face_box':      None,
            'parts_cache':   None,
            'parts_counter': 0,
            'det_countdown': 0,
        }
        self._in_lock  = threading.Lock()
        self._out_lock = threading.Lock()
        self._latest_in: np.ndarray | None = None
        # Latest INFERENCE OUTPUT (masks + classification, no rendered pixels)
        self._latest_out: dict = {
            'hand_mask':    None,
            'g':            {"gesture": None, "confidence": 0.0, "gesture_idx": 0},
            'skin_mask':    None,
            'region_masks': {},
            'face_active':  False,
            'em':           {"emotion": "neutral", "confidence": 0.0},
        }
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="emgor-inference")
        self.thread.start()
        print("[hand_and_face_job] inference worker started", flush=True)

    def push_input(self, frame_bgr):
        with self._in_lock:
            self._latest_in = frame_bgr

    def get_output(self):
        with self._out_lock:
            return dict(self._latest_out)

    def stop(self): self._stop.set()

    def _run(self):
        ws = self.ws
        n_iter = 0
        t_window = time.time()
        min_iter_dt = 1.0 / max(WORKER_TARGET_FPS, 1)
        while not self._stop.is_set():
            t_iter_start = time.time()
            with self._in_lock:
                frame = self._latest_in
            if frame is None:
                time.sleep(0.005)
                continue

            try:
                n_iter += 1
                if n_iter % WORKER_FPS_LOG == 0:
                    dt = time.time() - t_window
                    fps = WORKER_FPS_LOG / max(dt, 1e-6)
                    print(f"[hand_and_face_job] worker {fps:.1f} fps", flush=True)
                    t_window = time.time()

                # Mirror BEFORE inference to match combined_app exactly (its
                # capture loop flips horizontally, so the gesture model sees
                # mirrored hands). HaGRID classes are orientation-sensitive,
                # so feeding raw frames yields different gesture_idx values
                # and therefore wrong colors. We flip masks back below so the
                # rendered output stays in raw (input) orientation for TD.
                frame = cv2.flip(frame, 1)

                hand_mask, best_crop, best_m = _hand_pipeline(frame, ws, ws.get('face_box'))

                # EXACTLY combined_app's gesture call: zone crop + zone mask.
                if hand_present(hand_mask):
                    raw = run_gesture(best_crop, best_m, ws['gest'], ws['class_names'])
                    ws['smoother'].add(raw["probs"])
                else:
                    ws['smoother'].reset()
                g = ws['smoother'].current()

                skin_mask, region_masks, face_active, em = _face_pipeline(frame, ws, hand_mask)

                # Flip masks back to raw input orientation so render aligns
                # with the unmirrored TD feed. Gesture/emotion labels are
                # orientation-invariant — only the masks need flipping.
                if hand_mask is not None:
                    hand_mask = cv2.flip(hand_mask, 1)
                if skin_mask is not None:
                    skin_mask = cv2.flip(skin_mask, 1)
                if region_masks:
                    region_masks = {k: cv2.flip(v, 1) for k, v in region_masks.items()}

                with self._out_lock:
                    self._latest_out = {
                        'hand_mask':    hand_mask,
                        'g':            g,
                        'skin_mask':    skin_mask,
                        'region_masks': region_masks,
                        'face_active':  face_active,
                        'em':           em,
                    }
            except Exception as e:
                print(f"[hand_and_face_job] worker error: {e}", flush=True)
                time.sleep(0.05)

            # Cap worker rate — if this iteration was faster than the target
            # frame budget, sleep the rest. Caps GPU/CPU usage when nothing
            # interesting is happening. Display rate (TD cook) is unaffected.
            elapsed = time.time() - t_iter_start
            slack = min_iter_dt - elapsed
            if slack > 0:
                time.sleep(slack)


# ── render (runs in onCook on TD's main thread) ───────────────────────────────

def _render(canvas, out, fades, feedback_bufs):
    """Mesh-only render — no labels. Uses latest worker output. `fades`
    is a dict carrying mesh_fade / face_fade across calls so the meshes
    ramp smoothly in/out (matches combined_app's display-loop behavior).
    """
    hand_mask    = out['hand_mask']
    g            = out['g']
    skin_mask    = out['skin_mask']
    region_masks = out['region_masks']
    em           = out['em']
    face_active  = out['face_active']

    pres = hand_mask is not None and hand_present(hand_mask)
    fades['mesh'] = min(1.0, fades['mesh'] + 0.15) if pres else max(0.0, fades['mesh'] - 0.025)
    fades['face'] = min(1.0, fades['face'] + 0.20) if face_active else max(0.0, fades['face'] - 0.05)

    if pres and fades['mesh'] > 0.01:
        gidx  = g.get("gesture_idx", 0)
        color = GESTURE_COLORS_BGR[gidx % len(GESTURE_COLORS_BGR)]
        canvas = draw_mesh(canvas, hand_mask, color, fades['mesh'],
                           feedback_bufs, 'hand')

    if face_active and skin_mask is not None and skin_mask.any() and fades['face'] > 0.01:
        canvas = draw_face_aura(canvas, skin_mask, fades['face'])
        canvas = draw_mesh(canvas, skin_mask, (255, 255, 255), fades['face'] * 0.38,
                           feedback_bufs, 'face_skin', lines_only=True,
                           n_contour=20, n_interior=10, pts_update=0,
                           stable_interior=True, base_w_mult=1.2)
        conf = em.get("confidence", 0.0)
        nc   = max(4, int(4 + conf * 36))
        ni   = max(2, int(2 + conf * 18))
        lm_fade = fades['face'] * (0.20 + conf * 0.45)
        for rid, rmask in region_masks.items():
            if rmask.any():
                canvas = draw_mesh(canvas, rmask, (255, 255, 255), lm_fade,
                                   feedback_bufs, f'face_{rid}',
                                   lines_only=True, n_contour=nc, n_interior=ni,
                                   pts_update=0)
    return canvas


# ── state init / cache ────────────────────────────────────────────────────────
_STATE_VERSION = 7


def _state(scriptOp):
    st = scriptOp.fetch('state', None, search=False)
    if st is not None and st.get('_v') != _STATE_VERSION:
        if 'worker' in st:
            try: st['worker'].stop()
            except Exception: pass
        st = None
    if st is None:
        st = {
            '_v':            _STATE_VERSION,
            'worker':        _Worker(),
            'feedback_bufs': {},
            'fades':         {'mesh': 0.0, 'face': 0.0},
        }
        scriptOp.store('state', st)
    return st


def _bgr_uint8_from_input(scriptOp):
    arr = scriptOp.inputs[0].numpyArray(delayed=False)
    rgb = (arr[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _send_osc(scriptOp, g, em, face_active):
    osc = scriptOp.parent().op('oscout1') if hasattr(scriptOp, 'parent') else None
    if osc is None:
        return
    try:
        gesture = g["gesture"] if g["confidence"] >= CONF_THRESHOLD else None
        emotion = em["emotion"] if face_active and em["confidence"] >= FACE_CONF_THR else None
        osc.sendOSC('/emgor/hand', [str(gesture or "none"),
                                    float(g["confidence"]),
                                    int(g.get("gesture_idx", 0))])
        osc.sendOSC('/emgor/face', [str(emotion or "neutral"),
                                    float(em["confidence"])])
    except Exception:
        pass


# ── Script TOP entry ──────────────────────────────────────────────────────────

def onCook(scriptOp):
    if not scriptOp.inputs:
        scriptOp.copyNumpyArray(np.zeros((1, 1, 4), dtype=np.float32))
        return

    st = _state(scriptOp)
    worker = st['worker']

    # Push the latest frame to the worker (for inference) and get the worker's
    # latest masks. Worker runs at its own rate (~6-15 fps); render on this
    # thread runs every cook (~30-60 fps). Same pattern as combined_app's
    # display loop with the inference thread.
    frame_bgr = _bgr_uint8_from_input(scriptOp)
    worker.push_input(frame_bgr)
    out = worker.get_output()

    # Render mesh on a transparent canvas at TD's cook rate.
    canvas = np.zeros_like(frame_bgr)
    canvas = _render(canvas, out, st['fades'], st['feedback_bufs'])

    # BGR → RGBA, alpha from luminance
    rgb   = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = rgb.max(axis=2, keepdims=True)
    rgba  = np.concatenate([rgb, alpha], axis=2)
    scriptOp.copyNumpyArray(rgba)

    # Storage + OSC (cheap, every cook)
    g  = out['g']
    em = out['em']
    face_active = out['face_active']
    gesture = g["gesture"] if g["confidence"] >= CONF_THRESHOLD else None
    emotion = em["emotion"] if face_active and em["confidence"] >= FACE_CONF_THR else None
    scriptOp.store('classification', {
        "hand": {"gesture": gesture, "confidence": float(g["confidence"]),
                 "gesture_idx": int(g.get("gesture_idx", 0))},
        "face": {"emotion": emotion, "confidence": float(em["confidence"])},
    })
    _send_osc(scriptOp, g, em, face_active)


def onSetupParameters(scriptOp): return
def onPulse(par): return
