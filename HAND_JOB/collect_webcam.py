"""
Webcam gesture data collection — auto-labeled, auto-cropped.

Cycles through all 18 gesture classes. For each:
  - 4-second countdown while you get into position
  - 20-second recording at 5 fps → ~100 frames saved
  - Segmentation model auto-crops your hand (same format as HaGRID)

Saves to: data/webcam/{gesture_name}/frame_{NNNN}.jpg

Run from HAND_JOB/:
    python3 collect_webcam.py

Flags:
    --gestures call fist palm     # only collect these (default: all 18)
    --fps 5                       # capture rate (default: 5)
    --duration 20                 # seconds per gesture (default: 20)
    --round 1                     # append round suffix to avoid overwriting (default: 0)
    --skip-seg                    # save full frame instead of seg-cropped hand
"""
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from live_app.models import load_models

CLASS_NAMES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted", "middle_finger",
]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SAVE_SIZE = 384  # same resolution as HaGRID


def draw_overlay(frame, text_lines, progress=None, color=(255, 255, 255)):
    out = frame.copy()
    h, w = out.shape[:2]
    # dim background
    cv2.rectangle(out, (0, 0), (w, 120 + 30 * len(text_lines)), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.35, out, 0.65, 0, out)
    for i, line in enumerate(text_lines):
        cv2.putText(out, line, (20, 50 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    if progress is not None:
        bar_w = int((w - 40) * progress)
        cv2.rectangle(out, (20, h - 30), (w - 20, h - 10), (60, 60, 60), -1)
        cv2.rectangle(out, (20, h - 30), (20 + bar_w, h - 10), (0, 200, 80), -1)
    return out


def seg_crop(frame_bgr, seg_model, pad_frac=0.18):
    """Return hand-cropped image using seg model, or None if no hand found."""
    from live_app.models import run_seg_prob
    from live_app.config import SEG_THRESHOLD, HAND_MIN_AREA
    prob = run_seg_prob(frame_bgr, seg_model)
    mask = (prob > SEG_THRESHOLD).astype(np.uint8)
    h, w = mask.shape
    if np.count_nonzero(mask) / (h * w) < HAND_MIN_AREA:
        return None
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, bw, bh = cv2.boundingRect(coords)
    pad = int(max(bw, bh) * pad_frac)
    x0 = max(0, x - pad);      y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return frame_bgr[y0:y1, x0:x1]


def collect_gesture(cap, seg_model, gesture, out_dir, fps, duration, round_id, skip_seg):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.jpg")))

    # --- countdown ---
    countdown = 4
    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        remaining = countdown - int(time.time() - t_start)
        if remaining <= 0:
            break
        display = draw_overlay(frame,
            [f"Next: {gesture.upper()}", f"Get ready... {remaining}"],
            color=(255, 220, 50))
        cv2.imshow("Collect", display)
        cv2.waitKey(1)

    # --- recording ---
    interval = 1.0 / fps
    saved = 0
    t_start = time.time()
    t_next = t_start
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        elapsed = time.time() - t_start
        if elapsed >= duration:
            break
        progress = elapsed / duration

        now = time.time()
        if now >= t_next:
            t_next += interval
            crop = None if skip_seg else seg_crop(frame, seg_model)
            if crop is not None:
                save_img = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
                indicator = (0, 200, 80)  # green — hand found
            else:
                # fallback: center square crop of full frame
                h, w = frame.shape[:2]
                side = min(h, w)
                x0 = (w - side) // 2; y0 = (h - side) // 2
                save_img = cv2.resize(frame[y0:y0+side, x0:x0+side],
                                      (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
                indicator = (50, 50, 255)  # red — no hand, full frame saved

            fname = out_dir / f"r{round_id}_{existing + saved:05d}.jpg"
            cv2.imwrite(str(fname), save_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        status = "RECORDING" if True else ""
        display = draw_overlay(frame,
            [f"RECORDING: {gesture.upper()}",
             f"Saved: {saved}  |  {duration - elapsed:.1f}s left"],
            progress=progress, color=(100, 255, 100))
        # draw hand outline if seg found it
        cv2.imshow("Collect", display)
        cv2.waitKey(1)

    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gestures", nargs="+", default=CLASS_NAMES)
    ap.add_argument("--fps",      type=int,   default=5)
    ap.add_argument("--duration", type=int,   default=20)
    ap.add_argument("--round",    type=int,   default=0)
    ap.add_argument("--skip-seg", action="store_true")
    args = ap.parse_args()

    # Validate gesture names
    for g in args.gestures:
        if g not in CLASS_NAMES:
            print(f"Unknown gesture: {g}. Valid: {CLASS_NAMES}")
            sys.exit(1)

    print("[collect] Loading segmentation model...", flush=True)
    seg_model = None
    if not args.skip_seg:
        try:
            seg_model, _, _ = load_models()
            print("[collect] Seg model ready.", flush=True)
        except Exception as e:
            print(f"[collect] Seg model failed ({e}), falling back to center crop.", flush=True)
            args.skip_seg = True

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[collect] Cannot open webcam.")
        sys.exit(1)

    cv2.namedWindow("Collect", cv2.WINDOW_NORMAL)

    out_root = Path("data/webcam")
    total_saved = 0

    for i, gesture in enumerate(args.gestures):
        out_dir = out_root / gesture
        print(f"[collect] [{i+1}/{len(args.gestures)}] {gesture} → {out_dir}", flush=True)

        # Show "press any key to start" screen
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            existing = len(list(out_dir.glob("*.jpg"))) if out_dir.exists() else 0
            display = draw_overlay(frame, [
                f"[{i+1}/{len(args.gestures)}] {gesture.upper()}",
                f"Existing frames: {existing}",
                "Press SPACE to start, Q to quit",
            ], color=(200, 200, 255))
            cv2.imshow("Collect", display)
            k = cv2.waitKey(30) & 0xFF
            if k == ord(" "):
                break
            if k == ord("q"):
                print("[collect] Quit.")
                cap.release()
                cv2.destroyAllWindows()
                return

        n = collect_gesture(cap, seg_model, gesture, out_dir,
                            args.fps, args.duration, args.round, args.skip_seg)
        total_saved += n
        print(f"[collect] {gesture}: saved {n} frames.", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    print(f"[collect] Done. Total saved: {total_saved} frames across {len(args.gestures)} gestures.")
    print(f"[collect] Data at: {out_root.resolve()}")


if __name__ == "__main__":
    main()
