"""
Browse and delete webcam captures before training.

Controls:
  LEFT / RIGHT  — previous / next image
  D or DELETE   — delete current image
  A / Z         — jump back / forward 10 images
  N             — next class
  P             — previous class
  Q             — quit

Run from HAND_JOB/:
    python3 review.py              # all classes with captures
    python3 review.py rock palm    # specific classes only
"""
import sys
from pathlib import Path

import cv2
import numpy as np

SAVE_ROOT = Path("data/webcam")
CLASSES = [
    "call","dislike","fist","four","like","ok","one","palm","peace",
    "peace_inverted","rock","stop","stop_inverted","three","three2","two_up",
    "two_up_inverted","middle_finger","background",
]

C_DARK  = (20, 20, 20)
C_GREEN = (80, 220, 80)
C_RED   = (60, 60, 220)
C_GREY  = (130, 130, 130)
C_WHITE = (255, 255, 255)


def load_files(cls_name):
    d = SAVE_ROOT / cls_name
    if not d.exists():
        return []
    return sorted(d.glob("*.jpg"))


def render(img_bgr, cls_name, idx, total, cls_idx, cls_total):
    h, w = img_bgr.shape[:2]
    out  = img_bgr.copy()

    # top bar
    cv2.rectangle(out, (0, 0), (w, 80), C_DARK, -1)
    cv2.addWeighted(img_bgr[:80], 0.1, out[:80], 0.9, 0, out[:80])

    label = cls_name.upper().replace("_", " ")
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2)
    cv2.putText(out, label, ((w - tw) // 2, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, C_GREEN, 2, cv2.LINE_AA)

    # top-left: class progress
    cv2.putText(out, f"class {cls_idx+1}/{cls_total}",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GREY, 1, cv2.LINE_AA)

    # top-right: image index
    cv2.putText(out, f"{idx+1} / {total}",
                (w - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1, cv2.LINE_AA)

    # bottom bar
    cv2.rectangle(out, (0, h - 44), (w, h), C_DARK, -1)
    cv2.addWeighted(img_bgr[h-44:], 0.1, out[h-44:], 0.9, 0, out[h-44:])
    hints = "←/→ navigate   D=delete   A/Z=jump 10   N=next class   P=prev class   Q=quit"
    (hw, _), _ = cv2.getTextSize(hints, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(out, hints, ((w - hw) // 2, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GREY, 1, cv2.LINE_AA)

    return out


def show_deleted(w, h):
    """Flash a red DELETED banner for one frame."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w, h), (30, 30, 80), -1)
    txt = "DELETED"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 4)
    cv2.putText(img, txt, ((w-tw)//2, (h+th)//2),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, C_RED, 4, cv2.LINE_AA)
    return img


def review_class(cls_name, cls_idx, cls_total):
    """Returns: 'next', 'prev', or 'quit'."""
    files = load_files(cls_name)
    if not files:
        return "next"

    idx = 0
    WIN = "Review"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 860)

    while True:
        files = load_files(cls_name)  # refresh after deletes
        if not files:
            return "next"
        idx = min(idx, len(files) - 1)

        img = cv2.imread(str(files[idx]))
        if img is None:
            idx = min(idx + 1, len(files) - 1)
            continue

        # upscale small images for comfortable viewing
        h, w = img.shape[:2]
        if w < 600:
            scale = 600 // w
            img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        display = render(img, cls_name, idx, len(files), cls_idx, cls_total)
        cv2.imshow(WIN, display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            return "quit"
        elif key == ord("n"):
            return "next"
        elif key == ord("p"):
            return "prev"
        elif key == 81 or key == ord("a"):   # left arrow or A
            idx = max(0, idx - 1)
        elif key == 83 or key == ord("z"):   # right arrow or Z
            idx = min(len(files) - 1, idx + 1)
        elif key == 82:                       # up arrow — jump back 10
            idx = max(0, idx - 10)
        elif key == 84:                       # down arrow — jump forward 10
            idx = min(len(files) - 1, idx + 10)
        elif key in (ord("d"), 127, 8):      # D, DELETE, BACKSPACE
            path = files[idx]
            h2, w2 = display.shape[:2]
            cv2.imshow(WIN, show_deleted(w2, h2))
            cv2.waitKey(200)
            path.unlink()
            print(f"[review] Deleted {path.name}")
            files = load_files(cls_name)
            if not files:
                return "next"
            idx = min(idx, len(files) - 1)
        # ← / → on Mac send 2-byte escape sequences; handle common case
        elif key == 255:
            pass  # ignore unknown keys


def main():
    filter_classes = sys.argv[1:] if len(sys.argv) > 1 else None

    classes_with_data = [
        c for c in CLASSES
        if (SAVE_ROOT / c).exists() and len(list((SAVE_ROOT / c).glob("*.jpg"))) > 0
    ]
    if filter_classes:
        classes_with_data = [c for c in classes_with_data if c in filter_classes]

    if not classes_with_data:
        print("No webcam captures found. Run collect.py first.")
        return

    print(f"[review] {len(classes_with_data)} classes to review:")
    for c in classes_with_data:
        print(f"  {c:25s} {len(load_files(c)):4d} images")
    print()

    cls_idx = 0
    while 0 <= cls_idx < len(classes_with_data):
        result = review_class(classes_with_data[cls_idx], cls_idx, len(classes_with_data))
        if result == "quit":
            break
        elif result == "next":
            cls_idx += 1
        elif result == "prev":
            cls_idx = max(0, cls_idx - 1)

    cv2.destroyAllWindows()
    print("\n[review] Final counts:")
    total = 0
    for c in classes_with_data:
        n = len(load_files(c))
        print(f"  {c:25s} {n:4d}")
        total += n
    print(f"  {'TOTAL':25s} {total:4d}")
    print("\nReady to train: python3 train_all.py")


if __name__ == "__main__":
    main()
