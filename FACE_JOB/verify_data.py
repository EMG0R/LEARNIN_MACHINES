"""
Pre-flight data check. Run before train_all.py to confirm all required files exist.

    python3 FACE_JOB/verify_data.py
"""
from pathlib import Path
import sys

DATA = Path(__file__).parent / "data"

OK   = "\033[32m  ✓\033[0m"
FAIL = "\033[31m  ✗\033[0m"
WARN = "\033[33m  !\033[0m"

failures = []
warnings = []


def check(label: str, path: Path, required: bool = True):
    if path.exists():
        count = ""
        if path.is_dir():
            n = sum(1 for _ in path.rglob("*") if _.is_file())
            count = f"  ({n} files)"
        print(f"{OK} {label}{count}")
        return True
    else:
        if required:
            failures.append(label)
            print(f"{FAIL} {label}  ← MISSING: {path}")
        else:
            warnings.append(label)
            print(f"{WARN} {label}  ← optional, not found: {path}")
        return False


print("\n── WIDER FACE (face detector) ──────────────────────────────────")
check("WIDER_train images",    DATA / "wider_face/WIDER_train/images")
check("WIDER_val images",      DATA / "wider_face/WIDER_val/images")
check("train annotations",     DATA / "wider_face/wider_face_split/wider_face_train_bbx_gt.txt")
check("val annotations",       DATA / "wider_face/wider_face_split/wider_face_val_bbx_gt.txt")

print("\n── CelebAMask-HQ (face parts U-Net) ────────────────────────────")
check("CelebA-HQ-img",         DATA / "celeba_mask/CelebAMask-HQ/CelebA-HQ-img")
check("mask-anno dir",         DATA / "celeba_mask/CelebAMask-HQ/CelebAMask-HQ-mask-anno")

print("\n── FER+ / FER2013 (emotion classifier) ─────────────────────────")
check("fer2013.csv (pixels)",  DATA / "fer/fer2013.csv")
check("fer2013new.csv (FER+)", DATA / "fer/fer2013new.csv")

print("\n── RAF-DB (emotion, optional) ───────────────────────────────────")
check("list_patition_label",   DATA / "rafdb/list_patition_label.txt", required=False)
check("Image/aligned",         DATA / "rafdb/Image/aligned",           required=False)

print("\n── ExpW (emotion, optional) ─────────────────────────────────────")
check("label.lst",             DATA / "expw/label.lst",  required=False)
check("image dir",             DATA / "expw/image",      required=False)

print()
if failures:
    print(f"\033[31mFAILED: {len(failures)} required dataset(s) missing:\033[0m")
    for f in failures:
        print(f"  - {f}")
    print("\nRun:  python3 FACE_JOB/download_data.py\n")
    sys.exit(1)
else:
    active = ["ferplus"]
    if (DATA / "rafdb/list_patition_label.txt").exists():
        active.append("rafdb")
    if (DATA / "expw/label.lst").exists():
        active.append("expw")

    if warnings:
        print(f"\033[33mOptional datasets not found: {[w for w in warnings]}\033[0m")
        print(f"Emotion will train on: DATASETS={','.join(active)}")

    print("\033[32mAll required data present. Ready to train.\033[0m")
    print(f"\n  sudo python3 FACE_JOB/train_all.py\n")
    if active != ["ferplus", "rafdb", "expw"]:
        print(f"  (emotion will use: DATASETS={','.join(active)})")
        print(f"  To set explicitly: DATASETS={','.join(active)} sudo python3 FACE_JOB/train_all.py\n")
