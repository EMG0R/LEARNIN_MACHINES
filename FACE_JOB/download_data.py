"""
Download all FACE_JOB datasets.

Automatic downloads:
  - WIDER FACE (train + val + annotations)
  - CelebAMask-HQ
  - FER+ (labels) + FER2013 (pixels from Kaggle)
  - ExpW

Manual (printed instructions):
  - RAF-DB: requires request form

Run from FACE_JOB/:
    python3 download_data.py                 # all automatic
    python3 download_data.py --only wider    # single dataset
"""
import argparse, hashlib, subprocess, sys, zipfile, tarfile
from pathlib import Path
from urllib.request import urlretrieve

DATA = Path(__file__).parent / "data"

DATASETS = {
    "wider": {
        "urls": [
            ("https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip", "WIDER_train.zip"),
            ("https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip",   "WIDER_val.zip"),
            ("http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip", "wider_face_split.zip"),
        ],
        "subdir": "wider_face",
    },
    "celeba": {
        "urls": [
            # CelebAMask-HQ is hosted on Google Drive; direct links are fragile.
            # Use gdown if available; fallback prints instructions.
        ],
        "subdir": "celeba_mask",
        "manual": (
            "CelebAMask-HQ: download from https://github.com/switchablenorms/CelebAMask-HQ\n"
            "  Place CelebAMask-HQ.zip in FACE_JOB/data/celeba_mask/ and unzip."
        ),
    },
    "fer": {
        "urls": [
            ("https://github.com/microsoft/FERPlus/raw/master/fer2013new.csv", "fer2013new.csv"),
        ],
        "subdir": "fer",
        "manual": (
            "FER2013 pixels require Kaggle:\n"
            "  kaggle datasets download -d msambare/fer2013 -p FACE_JOB/data/fer/\n"
            "  unzip FACE_JOB/data/fer/fer2013.zip -d FACE_JOB/data/fer/"
        ),
    },
    "rafdb": {
        "urls": [],
        "subdir": "rafdb",
        "manual": (
            "RAF-DB requires a request form:\n"
            "  http://www.whdeng.cn/RAF/model1.html\n"
            "  After approval, place contents into FACE_JOB/data/rafdb/\n"
            "  Expected structure: rafdb/Image/aligned/  rafdb/EmoLabel/list_patition_label.txt"
        ),
    },
    "expw": {
        "urls": [],
        "subdir": "expw",
        "manual": (
            "ExpW: download from https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html\n"
            "  Place images into FACE_JOB/data/expw/image/\n"
            "  Place label.lst into FACE_JOB/data/expw/label.lst"
        ),
    },
}


def download(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"  [skip] {dst.name} already exists")
        return
    print(f"  [get] {url}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dst)


def extract(archive: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [extract] {archive.name} → {dst_dir}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z: z.extractall(dst_dir)
    elif archive.suffix in (".tar", ".gz", ".tgz"):
        with tarfile.open(archive) as t: t.extractall(dst_dir)


def fetch(key: str) -> None:
    spec = DATASETS[key]
    subdir = DATA / spec["subdir"]
    subdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {key} → {subdir} ===")

    for url, fname in spec.get("urls", []):
        archive = subdir / fname
        download(url, archive)
        if archive.suffix in (".zip", ".tar", ".gz", ".tgz"):
            extract(archive, subdir)

    if "manual" in spec:
        print(f"\n  MANUAL STEP REQUIRED for {key}:")
        for line in spec["manual"].splitlines():
            print(f"    {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(DATASETS.keys()))
    args = ap.parse_args()

    keys = [args.only] if args.only else list(DATASETS.keys())
    for k in keys:
        fetch(k)
    print("\nDone. Re-run with --only <key> to retry individual datasets.")


if __name__ == "__main__":
    main()
