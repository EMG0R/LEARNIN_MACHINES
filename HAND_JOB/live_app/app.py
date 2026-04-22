#!/usr/bin/env python3
"""
Real-time hand seg + gesture inference: fullscreen display, NDI video, OSC data.

Run from HAND_JOB/:
    python3 -m live_app.app

Press Q to quit. OSC → 127.0.0.1:9000. NDI source: LEARNIN_MACHINES.
"""
import time
import cv2
from live_app.config   import WEBCAM_INDEX, FRAME_W, FRAME_H
from live_app.models   import load_models, run_seg, run_gesture
from live_app.renderer import draw_mesh, draw_ui
from live_app.osc_sender import OSCSender
from live_app.ndi_sender import NDISender


def layer1_object_detection(frame): return None
def layer2a_face(crop): return None
def layer2c_body_pose(crop): return None
def layer3a_emotion(face_crop): return None


def main():
    print("[live_app] Loading models...")
    seg_model, gest_model, class_names = load_models()
    print("[live_app] Models loaded.")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {WEBCAM_INDEX}")

    osc = OSCSender()
    ndi = NDISender()

    cv2.namedWindow("LEARNIN_MACHINES", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("LEARNIN_MACHINES", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    fps    = 0.0
    t_prev = time.perf_counter()
    print("[live_app] Running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        mask     = run_seg(frame, seg_model)
        g_result = run_gesture(frame, mask, gest_model, class_names)
        present  = g_result["gesture"] is not None

        t_now  = time.perf_counter()
        fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        rendered = draw_mesh(frame, mask,
                             confidence=g_result["confidence"],
                             gesture_idx=g_result.get("gesture_idx", 0))
        rendered = draw_ui(rendered,
                           gesture=g_result["gesture"],
                           confidence=g_result["confidence"],
                           second=g_result["second"],
                           second_conf=g_result["second_conf"],
                           fps=fps, present=present)

        osc.send({
            "present": present, "fps": fps,
            "mask": mask if present else None,
            "gesture": g_result["gesture"],
            "confidence": g_result["confidence"],
            "second": g_result["second"],
            "second_conf": g_result["second_conf"],
            "gesture_idx": g_result.get("gesture_idx", 0),
        })
        ndi.send(rendered)

        cv2.imshow("LEARNIN_MACHINES", rendered)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    ndi.destroy()
    cv2.destroyAllWindows()
    print("[live_app] Stopped.")


if __name__ == "__main__":
    main()
