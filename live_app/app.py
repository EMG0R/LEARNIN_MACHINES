#!/usr/bin/env python3
"""
Integrated real-time vision: 1 hand + 1 face, mesh + labels, dynamic color.

Run from LEARNIN_MACHINES/:
    python3 -m live_app.app

Press Esc to quit.
"""
import time
import os
import numpy as np
import cv2
import torch
import pygame
from live_app.config import (
    WEBCAM_INDEX, FRAME_W, FRAME_H,
    CONF_THRESHOLD, FACE_CONF_THR,
    FACE_CLASS_SKIN, FACE_CLASS_EYE_L, FACE_CLASS_EYE_R, FACE_CLASS_MOUTH,
    FACE_DET_SCORE_THR, FACE_DET_IOU_THR,
    GESTURE_COLORS_BGR,
)
from live_app.models import (
    load_all_models, device,
    run_seg_prob_batch, postprocess_hand_mask, hand_present,
    run_gesture, GestureSmoother,
    run_face_det, run_face_parts, run_emotion_batch, EmotionSmoother,
)
from live_app.renderer import draw_mesh, draw_label, draw_face_aura, label_anchor_raw

from face_det.postprocess import decode, nms
import torchvision.transforms.functional as TF
from PIL import Image

FACE_DET_INTERVAL = 8   # re-run face detector every N frames
N_ZONES = 3


def _prep(crop_bgr, sz, mean, std):
    pil = Image.fromarray(crop_bgr[:, :, ::-1]).resize((sz, sz), Image.BILINEAR)
    return TF.normalize(TF.to_tensor(pil), mean, std)


def _face_mesh_data(parts_map, box, fh, fw):
    x1, y1, x2, y2 = box
    ch, cw = parts_map.shape[:2]
    bw, bh = x2 - x1, y2 - y1

    def _to_full(cls_ids):
        region = np.zeros_like(parts_map, dtype=np.uint8)
        for c in cls_ids:
            region[parts_map == c] = 255
        full = np.zeros((fh, fw), dtype=np.uint8)
        full[y1:y2, x1:x2] = cv2.resize(region, (bw, bh),
                                          interpolation=cv2.INTER_NEAREST)
        return full

    skin_mask = _to_full([FACE_CLASS_SKIN])
    # per-region masks for landmark density rendering
    region_masks = {
        "eye_l": _to_full([FACE_CLASS_EYE_L]),
        "eye_r": _to_full([FACE_CLASS_EYE_R]),
        "mouth": _to_full([FACE_CLASS_MOUTH]),
    }
    return skin_mask, region_masks


def main():
    models      = load_all_models()
    seg_model   = models["seg"]
    gest_model  = models["gest"]
    class_names = models["class_names"]
    fd_model    = models["fd"]
    fp_model    = models["fp"]
    em_model    = models["em"]

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {WEBCAM_INDEX}")

    # pygame handles HiDPI/Retina fullscreen correctly on macOS
    os.environ['SDL_VIDEO_ALLOW_SCREENSAVER'] = '1'
    pygame.init()
    info = pygame.display.Info()
    SCREEN_W, SCREEN_H = info.current_w, info.current_h
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.FULLSCREEN | pygame.NOFRAME)
    pygame.display.set_caption("LEARNIN_MACHINES")
    print(f"[display] pygame fullscreen {SCREEN_W}x{SCREEN_H}")

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40,
                                                 detectShadows=False)

    # hand state
    prob_emas  = [None] * N_ZONES
    smoother   = GestureSmoother(class_names)
    mesh_fade  = 0.0

    # face state
    face_box       = None   # cached face box (x1,y1,x2,y2)
    face_fade      = 0.0
    em_smoother    = EmotionSmoother(em_model._class_names)
    det_countdown  = 0      # run detector when hits 0

    feedback_bufs: dict = {}
    _EM_ORDER = ["happy","sad","neutral","surprise","anger","fear","disgust"]
    LABEL_EMA     = 0.18
    hand_lpos     = None
    face_lpos     = None
    parts_cache   = None   # cached (skin_mask, region_masks) between frames
    parts_counter = 0      # run face_parts every 2 frames
    fps    = 0.0
    t_prev = time.perf_counter()

    # warm up background subtractor silently before showing anything
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            bg_sub.apply(frame)

    print("[live_app] Running. Press Esc to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        fg = bg_sub.apply(frame)

        # ── Hand ──────────────────────────────────────────────────────────────
        s     = h
        x_off = [0, (w - s) // 2, w - s]
        crops = [frame[:, x:x+s] for x in x_off]

        with torch.inference_mode():
            sz = seg_model._img_size
            tensors = [TF.normalize(TF.to_tensor(
                           Image.fromarray(c[:, :, ::-1]).resize((sz, sz), Image.BILINEAR)),
                           seg_model._mean, seg_model._std)
                       for c in crops]
            logits = seg_model(torch.stack(tensors).to(device))
        probs_list = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)

        zone_masks = []
        for i, (prob, x) in enumerate(zip(probs_list, x_off)):
            prob_rs = cv2.resize(prob, (s, s), interpolation=cv2.INTER_LINEAR)
            m, prob_emas[i] = postprocess_hand_mask(prob_rs, prob_emas[i])
            m = cv2.bitwise_and(m, fg[:, x:x+s])
            zone_masks.append((m, x))

        # blank face box out of hand masks so face region is never treated as a hand
        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            for i, (m, x) in enumerate(zone_masks):
                lx1 = max(0, fx1 - x); lx2 = max(0, fx2 - x)
                if lx2 > lx1:
                    m[fy1:fy2, lx1:lx2] = 0
                zone_masks[i] = (m, x)

        # pick zone with largest hand mask
        best_idx = max(range(N_ZONES), key=lambda i: np.count_nonzero(zone_masks[i][0]))
        best_m, best_x = zone_masks[best_idx]

        # full-frame hand mask (union of all zones)
        hand_mask_full = np.zeros((h, w), dtype=np.uint8)
        for m, x in zone_masks:
            hand_mask_full[:, x:x+s] = np.maximum(hand_mask_full[:, x:x+s], m)

        pres = hand_present(hand_mask_full)
        mesh_fade = min(1.0, mesh_fade + 0.15) if pres else max(0.0, mesh_fade - 0.025)

        if pres:
            raw = run_gesture(crops[best_idx], best_m, gest_model, class_names)
            smoother.add(raw["probs"])
        else:
            smoother.reset()
        g = smoother.current()

        # ── Face ──────────────────────────────────────────────────────────────
        det_countdown -= 1
        if det_countdown <= 0:
            det_countdown = FACE_DET_INTERVAL
            fh_img, fw_img = h, w
            sz_fd = fd_model._img_size
            with torch.inference_mode():
                t_fd = _prep(frame, sz_fd, fd_model._mean, fd_model._std)
                obj, bbox, ctr = fd_model(t_fd.unsqueeze(0).to(device))
            results = decode(obj, bbox, ctr, stride=8, score_thr=FACE_DET_SCORE_THR)
            boxes, scores = results[0]
            face_box = None
            if boxes.numel() > 0:
                keep = nms(boxes, scores, iou_thr=FACE_DET_IOU_THR)
                boxes_np  = boxes[keep].cpu().numpy()
                scores_np = scores[keep].cpu().numpy()
                sx, sy = fw_img / sz_fd, fh_img / sz_fd
                best_score = -1
                for box, score in zip(boxes_np, scores_np):
                    x1 = int(max(0, box[0]*sx)); y1 = int(max(0, box[1]*sy))
                    x2 = int(min(fw_img, box[2]*sx)); y2 = int(min(fh_img, box[3]*sy))
                    area = (x2-x1) * (y2-y1)
                    if area > 400 and score > best_score:
                        best_score = score
                        face_box = (x1, y1, x2, y2)

        # run parts + emotion on cached face box every frame
        skin_mask, region_masks = None, {}
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            # pad crop so model sees forehead/chin/sides
            pad_x = int((x2 - x1) * 0.20)
            pad_y = int((y2 - y1) * 0.25)
            cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x); cy2 = min(h, y2 + pad_y)
            face_crop = frame[cy1:cy2, cx1:cx2]
            padded_box = (cx1, cy1, cx2, cy2)
            if face_crop.size > 0:
                parts_counter += 1
                if parts_counter >= 2 or parts_cache is None:
                    parts_counter = 0
                    parts_map = run_face_parts(face_crop, fp_model)
                    skin_mask, region_masks = _face_mesh_data(parts_map, padded_box, h, w)
                    # dilate skin mask to fill model gaps
                    dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                    skin_mask = cv2.dilate(skin_mask, dil, iterations=1)
                    inv_hand = cv2.bitwise_not(hand_mask_full)
                    skin_mask = cv2.bitwise_and(skin_mask, inv_hand)
                    region_masks = {k: cv2.bitwise_and(m, inv_hand)
                                    for k, m in region_masks.items()}
                    parts_cache = (skin_mask, region_masks)
                else:
                    skin_mask, region_masks = parts_cache
                em_raw = run_emotion_batch([frame[y1:y2, x1:x2]], em_model)
                if em_raw:
                    em_smoother.add(em_raw[0]["probs"])
                face_fade = min(1.0, face_fade + 0.2)
            else:
                face_fade = max(0.0, face_fade - 0.05)
        else:
            face_fade = max(0.0, face_fade - 0.05)
            em_smoother.reset()

        em = em_smoother.current()

        # ── Render ────────────────────────────────────────────────────────────
        rendered = frame.copy()

        if mesh_fade > 0.01:
            gidx  = g.get("gesture_idx", 0)
            color = GESTURE_COLORS_BGR[gidx % len(GESTURE_COLORS_BGR)]
            rendered = draw_mesh(rendered, hand_mask_full, color, mesh_fade,
                                 feedback_bufs, "hand")
            if pres and g["gesture"] and g["confidence"] >= CONF_THRESHOLD:
                raw_pos = label_anchor_raw(hand_mask_full, h, w)
                if raw_pos:
                    hand_lpos = (raw_pos if hand_lpos is None else
                                 (hand_lpos[0] + LABEL_EMA * (raw_pos[0] - hand_lpos[0]),
                                  hand_lpos[1] + LABEL_EMA * (raw_pos[1] - hand_lpos[1])))
                rendered = draw_label(rendered, hand_lpos,
                                      g["gesture"], color, g["confidence"])

        _EM_TO_GIDX = [0, 3, 6, 9, 12, 15, 2]  # happy sad neutral surprise anger fear disgust

        if face_fade > 0.01 and skin_mask is not None and skin_mask.any():
            eidx   = _EM_ORDER.index(em["emotion"]) if em.get("emotion") in _EM_ORDER else 2
            fcolor = GESTURE_COLORS_BGR[_EM_TO_GIDX[eidx]]

            # dark purple aura behind everything
            rendered = draw_face_aura(rendered, skin_mask, face_fade)
            # white skin wireframe — denser structure lines
            rendered = draw_mesh(rendered, skin_mask, (255, 255, 255), face_fade * 0.35,
                                 feedback_bufs, "face_skin", lines_only=True,
                                 n_contour=40, n_interior=20)

            # landmark regions: white, density scales 4→40 points with confidence
            conf = em.get("confidence", 0.0)
            nc = max(4, int(4 + conf * 36))   # 4 at 0% → 40 at 100%
            ni = max(2, int(2 + conf * 18))   # 2 at 0% → 20 at 100%
            lm_fade = face_fade * (0.3 + conf * 0.65)
            for rid, rmask in region_masks.items():
                if rmask.any():
                    rendered = draw_mesh(rendered, rmask, (255, 255, 255), lm_fade,
                                         feedback_bufs, f"face_{rid}",
                                         lines_only=True, n_contour=nc, n_interior=ni)

            if em["confidence"] >= FACE_CONF_THR:
                raw_pos = label_anchor_raw(skin_mask, h, w)
                if raw_pos:
                    face_lpos = (raw_pos if face_lpos is None else
                                 (face_lpos[0] + LABEL_EMA * (raw_pos[0] - face_lpos[0]),
                                  face_lpos[1] + LABEL_EMA * (raw_pos[1] - face_lpos[1])))
                rendered = draw_label(rendered, face_lpos,
                                      em["emotion"], fcolor, em["confidence"])

        t_now  = time.perf_counter()
        fps    = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        cv2.putText(rendered, f"{fps:.0f}", (w-44, h-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)

        # pygame display — stretches to fill screen, no letterboxing
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surf, (SCREEN_W, SCREEN_H))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release(); pygame.quit(); return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                cap.release(); pygame.quit(); return

    cap.release()
    pygame.quit()
    print("[live_app] Stopped.")


if __name__ == "__main__":
    main()
