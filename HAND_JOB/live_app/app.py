#!/usr/bin/env python3
"""
Real-time hand seg + gesture inference: fullscreen display, NDI video, OSC data.

Run from HAND_JOB/:
    python3 -m live_app.app

Press Esc to quit. OSC → 127.0.0.1:9000. NDI source: LEARNIN_MACHINES.

Pipeline (3 threads):
  capture    → shared latest_frame  (lock-protected, always newest)
  inference  → shared latest_result (lock-protected, always newest)
  main       → reads both at full camera fps, overlays last known result
"""
import time
import threading
import cv2
import numpy as np
from live_app.config   import WEBCAM_INDEX, FRAME_W, FRAME_H
from live_app.models   import (
    load_models, run_seg_prob_batch, run_gesture, hand_present,
    postprocess_mask, GestureSmoother,
)
from live_app.renderer import draw_mesh, draw_ui, draw_blobs
from live_app.osc_sender import OSCSender
from live_app.ndi_sender import NDISender


def layer1_object_detection(frame): return None
def layer2a_face(crop): return None
def layer2c_body_pose(crop): return None
def layer3a_emotion(face_crop): return None


def _capture_loop(cap, bg_sub, shared, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        fg = bg_sub.apply(frame)
        with shared["frame_lock"]:
            shared["frame"] = frame
            shared["fg"]    = fg


def _inference_loop(seg_model, gest_model, class_names, shared, stop_event):
    prob_emas = [None, None, None]
    smoother  = GestureSmoother(class_names)

    while not stop_event.is_set():
        with shared["frame_lock"]:
            frame = shared["frame"]
            fg    = shared["fg"]
        if frame is None:
            time.sleep(0.005)
            continue

        h, w  = frame.shape[:2]
        s     = h
        x_off = [0, (w - s) // 2, w - s]
        crops = [frame[:, x:x+s] for x in x_off]

        probs = run_seg_prob_batch(crops, seg_model)

        mask_disp  = np.zeros((h, w), dtype=np.uint8)
        crop_masks = []
        for i, (prob, x) in enumerate(zip(probs, x_off)):
            m, prob_emas[i] = postprocess_mask(prob, prob_emas[i])
            crop_masks.append((m, x))
            mask_disp[:, x:x+s] = np.maximum(mask_disp[:, x:x+s], m)

        present = hand_present(mask_disp)
        if present:
            best_crop, best_x = max(crop_masks, key=lambda t: np.count_nonzero(t[0]))
            raw = run_gesture(crops[x_off.index(best_x)], best_crop, gest_model, class_names)
            smoother.add(raw["probs"])
        else:
            smoother.reset()
        g_result = smoother.current()

        with shared["result_lock"]:
            shared["mask_disp"] = mask_disp
            shared["fg_disp"]   = fg
            shared["present"]   = present
            shared["g_result"]  = g_result
            shared["inf_frame"] = frame


def main():
    print("[live_app] Loading models...")
    seg_model, gest_model, class_names = load_models()
    print("[live_app] Models loaded.")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {WEBCAM_INDEX}")

    osc    = OSCSender()
    ndi    = NDISender()

    shared = {
        "frame_lock":  threading.Lock(),
        "result_lock": threading.Lock(),
        "frame": None, "fg": None,
        "mask_disp": None, "fg_disp": None,
        "present": False, "g_result": None, "inf_frame": None,
    }
    stop_event = threading.Event()
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40,
                                                 detectShadows=False)

    threading.Thread(target=_capture_loop,
                     args=(cap, bg_sub, shared, stop_event),
                     daemon=True).start()
    threading.Thread(target=_inference_loop,
                     args=(seg_model, gest_model, class_names, shared, stop_event),
                     daemon=True).start()

    cv2.namedWindow("LEARNIN_MACHINES", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("LEARNIN_MACHINES", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    fps       = 0.0
    t_prev    = time.perf_counter()
    mesh_fade = 0.0
    print("[live_app] Running. Press Esc to quit.")

    while True:
        # always grab the latest frame for display
        with shared["frame_lock"]:
            frame = shared["frame"]
            fg    = shared["fg"]
        if frame is None:
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        # grab latest inference result (may be from a slightly older frame — that's fine)
        with shared["result_lock"]:
            mask_disp = shared["mask_disp"]
            fg_disp   = shared["fg_disp"]
            present   = shared["present"]
            g_result  = shared["g_result"]

        if mask_disp is None:
            mask_disp = np.zeros(frame.shape[:2], dtype=np.uint8)
        if fg_disp is None:
            fg_disp = np.zeros(frame.shape[:2], dtype=np.uint8)
        if g_result is None:
            g_result = {"gesture": None, "confidence": 0.0, "second": None,
                        "second_conf": 0.0, "gesture_idx": 0}

        mesh_fade = min(1.0, mesh_fade + 0.15) if present else max(0.0, mesh_fade - 0.025)

        t_now  = time.perf_counter()
        fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        rendered = draw_blobs(frame, fg_disp, mask_disp)
        rendered = draw_mesh(rendered, mask_disp,
                             confidence=g_result["confidence"],
                             gesture_idx=g_result.get("gesture_idx", 0),
                             fade=mesh_fade)
        rendered = draw_ui(rendered,
                           gesture=g_result["gesture"],
                           confidence=g_result["confidence"],
                           second=g_result["second"],
                           second_conf=g_result["second_conf"],
                           fps=fps, present=present)

        osc.send({
            "present": present, "fps": fps,
            "mask": mask_disp if present else None,
            "gesture": g_result["gesture"],
            "confidence": g_result["confidence"],
            "second": g_result["second"],
            "second_conf": g_result["second_conf"],
            "gesture_idx": g_result.get("gesture_idx", 0),
        })
        ndi.send(rendered)

        cv2.imshow("LEARNIN_MACHINES", rendered)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    stop_event.set()
    cap.release()
    ndi.destroy()
    cv2.destroyAllWindows()
    print("[live_app] Stopped.")


if __name__ == "__main__":
    main()
