"""
Webcam gesture data collection — unlimited captures, auto-labeled by class.

Controls:
  SPACE      — capture (unlimited, green flash = hand cropped, yellow = full frame)
  BACKSPACE  — delete last capture (thumbnail shown in corner for 3s after each capture)
  N          — next class
  P          — previous class
  Q          — quit

Frames saved to: data/webcam/{class_name}/
The class name on screen IS the label — you never type anything.

Run from HAND_JOB/:
    python3 collect.py                          # all classes
    python3 collect.py --only call fist rock    # specific classes
    python3 collect.py --start-at middle_finger # resume from a class
"""
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from live_app.models import load_models, run_seg_prob
from live_app.config  import SEG_THRESHOLD, HAND_MIN_AREA

SAVE_SIZE  = 384
SAVE_ROOT  = Path("data/webcam")
THUMB_SHOW = 3.0   # seconds to show thumbnail after capture

CLASSES = [
    "call", "dislike", "fist", "four", "like", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted",
    "middle_finger",
    "background",
]

C_GREEN  = (80,  220, 80)
C_YELLOW = (50,  220, 220)
C_DARK   = (20,  20,  20)
C_GREY   = (120, 120, 120)
C_WHITE  = (255, 255, 255)
C_RED    = (60,  60,  220)


# ─── HAND CROP ────────────────────────────────────────────────────────────────
def hand_crop(frame, seg_model, pad=0.18):
    prob  = run_seg_prob(frame, seg_model)
    mask  = (prob > SEG_THRESHOLD).astype(np.uint8)
    h, w  = mask.shape
    if np.count_nonzero(mask) / (h * w) < HAND_MIN_AREA:
        return None
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, bw, bh = cv2.boundingRect(coords)
    p  = int(max(bw, bh) * pad)
    x0 = max(0, x - p);      y0 = max(0, y - p)
    x1 = min(w, x + bw + p); y1 = min(h, y + bh + p)
    return frame[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None


def center_square(frame):
    h, w  = frame.shape[:2]
    s     = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return frame[y0:y0 + s, x0:x0 + s]


def count_existing(cls_name):
    d = SAVE_ROOT / cls_name
    return len(list(d.glob("*.jpg"))) if d.exists() else 0


def last_saved_path(cls_name):
    d = SAVE_ROOT / cls_name
    if not d.exists():
        return None
    files = sorted(d.glob("*.jpg"))
    return files[-1] if files else None


# ─── RENDER ───────────────────────────────────────────────────────────────────
def render(frame, cls_name, cls_idx, cls_total, n_this_class,
           flash_color, flash_alpha, thumb, thumb_born):
    out = frame.copy()
    h, w = out.shape[:2]

    # flash overlay
    if flash_color is not None and flash_alpha > 0:
        ov = np.full_like(out, flash_color)
        cv2.addWeighted(ov, flash_alpha, out, 1 - flash_alpha, 0, out)

    # top bar
    cv2.rectangle(out, (0, 0), (w, 115), C_DARK, -1)
    cv2.addWeighted(frame[:115], 0.15, out[:115], 0.85, 0, out[:115])

    # class name
    label = cls_name.upper().replace("_", " ")
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.7, 3)
    cv2.putText(out, label, ((w - tw) // 2, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.7, C_GREEN, 3, cv2.LINE_AA)

    # progress  +  count
    cv2.putText(out, f"{cls_idx + 1}/{cls_total}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_GREY, 1, cv2.LINE_AA)
    count_str = f"{n_this_class} captured"
    (cw, _), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
    cv2.putText(out, count_str, (w - cw - 20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_WHITE, 1, cv2.LINE_AA)

    # bottom bar
    cv2.rectangle(out, (0, h - 50), (w, h), C_DARK, -1)
    cv2.addWeighted(frame[h - 50:], 0.15, out[h - 50:], 0.85, 0, out[h - 50:])
    hints = "SPACE=capture   BKSP=undo last   N=next   P=prev   Q=quit"
    cv2.putText(out, hints, (20, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GREY, 1, cv2.LINE_AA)

    # thumbnail — bottom-right corner, 10% of frame width, shown for THUMB_SHOW sec
    if thumb is not None and (time.time() - thumb_born) < THUMB_SHOW:
        age   = time.time() - thumb_born
        alpha = max(0.0, 1.0 - (age / THUMB_SHOW) * 0.5)  # gentle fade
        tw_px = max(60, w // 10)
        th_px = tw_px
        small = cv2.resize(thumb, (tw_px, th_px), interpolation=cv2.INTER_AREA)
        x0t = w - tw_px - 10
        y0t = h - th_px - 60   # just above bottom bar
        roi  = out[y0t:y0t + th_px, x0t:x0t + tw_px]
        blended = cv2.addWeighted(small, alpha, roi, 1 - alpha, 0)
        out[y0t:y0t + th_px, x0t:x0t + tw_px] = blended
        # border
        cv2.rectangle(out, (x0t - 1, y0t - 1), (x0t + tw_px, y0t + th_px),
                      C_GREY, 1)
        cv2.putText(out, "BKSP=undo", (x0t, y0t - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1, cv2.LINE_AA)

    return out


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only",     nargs="+", default=None)
    ap.add_argument("--start-at", default=None)
    ap.add_argument("--skip-seg", action="store_true")
    args = ap.parse_args()

    classes = args.only if args.only else CLASSES
    for c in classes:
        if c not in CLASSES:
            print(f"Unknown class '{c}'. Valid: {CLASSES}"); sys.exit(1)

    start_idx = 0
    if args.start_at:
        if args.start_at not in classes:
            print(f"--start-at '{args.start_at}' not in selected classes"); sys.exit(1)
        start_idx = classes.index(args.start_at)

    print("[collect] Loading seg model...", flush=True)
    seg_model = None
    if not args.skip_seg:
        try:
            seg_model, _, _ = load_models()
            print("[collect] Seg model ready.", flush=True)
        except Exception as e:
            print(f"[collect] Seg model unavailable ({e}). Saving full frames.", flush=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[collect] Cannot open webcam."); sys.exit(1)

    cv2.namedWindow("Collect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Collect", 1280, 720)

    cls_idx    = start_idx
    flash_color = None
    flash_start = 0.0
    FLASH_DUR   = 0.15

    thumb       = None   # last captured image (BGR, SAVE_SIZE)
    thumb_born  = 0.0

    while cls_idx < len(classes):
        cls_name = classes[cls_idx]
        (SAVE_ROOT / cls_name).mkdir(parents=True, exist_ok=True)

        ret, frame = cap.read()
        if not ret:
            continue

        n_this = count_existing(cls_name)

        # compute flash alpha
        flash_alpha = max(0.0, 1.0 - (time.time() - flash_start) / FLASH_DUR)

        display = render(frame, cls_name, cls_idx, len(classes), n_this,
                         flash_color, flash_alpha, thumb, thumb_born)
        cv2.imshow("Collect", display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("n"):
            cls_idx = min(len(classes) - 1, cls_idx + 1)
            thumb = None

        elif key == ord("p"):
            cls_idx = max(0, cls_idx - 1)
            thumb = None

        elif key == 8 or key == 127:  # BACKSPACE / DELETE
            last = last_saved_path(cls_name)
            if last:
                last.unlink()
                print(f"[collect] Deleted {last.name}", flush=True)
                thumb = None

        elif key == ord(" "):
            # capture
            if cls_name == "background" or seg_model is None or args.skip_seg:
                crop       = cv2.resize(center_square(frame), (SAVE_SIZE, SAVE_SIZE),
                                        interpolation=cv2.INTER_AREA)
                hand_found = False
            else:
                raw = hand_crop(frame, seg_model)
                if raw is not None:
                    crop       = cv2.resize(raw, (SAVE_SIZE, SAVE_SIZE),
                                            interpolation=cv2.INTER_AREA)
                    hand_found = True
                else:
                    crop       = cv2.resize(center_square(frame), (SAVE_SIZE, SAVE_SIZE),
                                            interpolation=cv2.INTER_AREA)
                    hand_found = False

            existing = count_existing(cls_name)
            fname    = SAVE_ROOT / cls_name / f"{existing:05d}.jpg"
            cv2.imwrite(str(fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            thumb      = crop.copy()
            thumb_born = time.time()
            flash_color = C_GREEN if hand_found else C_YELLOW
            flash_start = time.time()

            status = "hand cropped" if hand_found else "full frame"
            print(f"[collect] {cls_name} #{existing + 1}  ({status})", flush=True)

    cap.release()
    cv2.destroyAllWindows()

    print("\n[collect] Done. Summary:")
    total = 0
    for c in classes:
        n = count_existing(c)
        if n:
            print(f"  {c:25s} {n:4d}")
            total += n
    print(f"  {'TOTAL':25s} {total:4d}")
    print(f"\nData saved to: {SAVE_ROOT.resolve()}")
    print("Next: python3 train_all.py")


if __name__ == "__main__":
    main()
