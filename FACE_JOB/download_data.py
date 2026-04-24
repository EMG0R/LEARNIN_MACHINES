"""
Download all FACE_JOB datasets.

Automatic:
  - WIDER FACE (train + val + annotations)       via huggingface_hub
  - CelebAMask-HQ                                via huggingface_hub
  - FER+ labels                                  direct URL
  - FER2013 pixels                               via Kaggle CLI

Manual (printed instructions):
  - RAF-DB: requires request form
  - ExpW:   requires CUHK download

Run from repo root:
    python3 FACE_JOB/download_data.py
    python3 FACE_JOB/download_data.py --only wider
    python3 FACE_JOB/download_data.py --only celeba
    python3 FACE_JOB/download_data.py --only fer
"""
import argparse, subprocess, sys, zipfile, tarfile
from pathlib import Path
from urllib.request import urlretrieve

DATA = Path(__file__).parent / "data"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ensure_hf():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("  [pip] installing huggingface_hub...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "--break-system-packages", "huggingface_hub",
        ])


def is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            z.infolist()
        return True
    except Exception:
        return False


def download_url(url: str, dst: Path) -> None:
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
        with zipfile.ZipFile(archive) as z:
            z.extractall(dst_dir)
    elif archive.suffix in (".tar", ".gz", ".tgz"):
        with tarfile.open(archive) as t:
            t.extractall(dst_dir)


def hf_download_file(repo_id: str, filename: str, dst: Path) -> None:
    """Download a single file from a HuggingFace dataset repo."""
    _ensure_hf()
    from huggingface_hub import hf_hub_download

    if dst.exists() and (dst.suffix != ".zip" or is_valid_zip(dst)):
        print(f"  [skip] {dst.name} already valid")
        return
    if dst.exists():
        print(f"  [re-download] {dst.name} was corrupt, removing")
        dst.unlink()

    print(f"  [hf] {repo_id} / {filename}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset",
        local_dir=dst.parent,
    )
    tmp_path = Path(tmp)
    if tmp_path != dst:
        tmp_path.rename(dst)


# ─── dataset fetchers ─────────────────────────────────────────────────────────

def fetch_wider():
    subdir = DATA / "wider_face"
    subdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== wider_face → {subdir} ===")

    for hf_file, local_name in [
        ("data/WIDER_train.zip", "WIDER_train.zip"),
        ("data/WIDER_val.zip",   "WIDER_val.zip"),
    ]:
        dst = subdir / local_name
        split = local_name.replace(".zip", "")
        img_dir = subdir / split / "images"
        if img_dir.exists() and any(img_dir.iterdir()):
            print(f"  [skip] {split}/images already extracted")
            continue
        hf_download_file("wider_face", hf_file, dst)
        extract(dst, subdir)

    # annotations
    anno_dir = subdir / "wider_face_split"
    if not (anno_dir / "wider_face_train_bbx_gt.txt").exists():
        anno_zip = subdir / "wider_face_split.zip"
        download_url(
            "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip",
            anno_zip,
        )
        extract(anno_zip, subdir)
    else:
        print("  [skip] wider_face_split annotations already present")


def fetch_celeba():
    subdir = DATA / "celeba_mask"
    subdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== celeba_mask → {subdir} ===")

    hq_dir = subdir / "CelebAMask-HQ"
    img_dir = hq_dir / "CelebA-HQ-img"
    mask_dir = hq_dir / "CelebAMask-HQ-mask-anno"

    if img_dir.exists() and mask_dir.exists() and any(img_dir.iterdir()):
        print("  [skip] CelebAMask-HQ already extracted")
        return

    dst = subdir / "CelebAMask-HQ.zip"
    hf_download_file("liusq/CelebAMask-HQ", "CelebAMask-HQ.zip", dst)
    extract(dst, subdir)

    if not hq_dir.exists():
        candidates = [d for d in subdir.iterdir() if d.is_dir()]
        print(f"  [warn] expected {hq_dir}, found: {[c.name for c in candidates]}")


def fetch_fer():
    subdir = DATA / "fer"
    subdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== fer → {subdir} ===")

    ferplus_csv = subdir / "fer2013new.csv"
    download_url(
        "https://github.com/microsoft/FERPlus/raw/master/fer2013new.csv",
        ferplus_csv,
    )

    fer_csv = subdir / "fer2013.csv"
    if fer_csv.exists():
        print(f"  [skip] fer2013.csv already exists")
        return

    print("  [kaggle] downloading msambare/fer2013...")
    try:
        subprocess.check_call([
            "kaggle", "datasets", "download",
            "-d", "deadskull7/fer2013",
            "-p", str(subdir),
            "--unzip",
        ])
    except FileNotFoundError:
        print(
            "\n  ERROR: kaggle CLI not found.\n"
            "  Install it:  pip install kaggle\n"
            "  Then put your API token at ~/.kaggle/kaggle.json\n"
            "  Then re-run:  python3 FACE_JOB/download_data.py --only fer"
        )
        return
    except subprocess.CalledProcessError as e:
        print(
            f"\n  ERROR: kaggle download failed (exit {e.returncode}).\n"
            "  Make sure ~/.kaggle/kaggle.json is valid.\n"
            "  Manual alternative:\n"
            "    kaggle datasets download -d msambare/fer2013 -p FACE_JOB/data/fer/ --unzip"
        )


def fetch_rafdb():
    print("\n=== rafdb (MANUAL) ===")
    print(
        "  RAF-DB requires a request form:\n"
        "    http://www.whdeng.cn/RAF/model1.html\n"
        "  After approval, place contents into FACE_JOB/data/rafdb/ so that:\n"
        "    FACE_JOB/data/rafdb/Image/aligned/   (aligned face images)\n"
        "    FACE_JOB/data/rafdb/list_patition_label.txt"
    )


def fetch_expw():
    print("\n=== expw (MANUAL) ===")
    print(
        "  ExpW requires manual download from:\n"
        "    https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html\n"
        "  Place files so that:\n"
        "    FACE_JOB/data/expw/image/   (face images)\n"
        "    FACE_JOB/data/expw/label.lst"
    )


# ─── main ─────────────────────────────────────────────────────────────────────

FETCHERS = {
    "wider":  fetch_wider,
    "celeba": fetch_celeba,
    "fer":    fetch_fer,
    "rafdb":  fetch_rafdb,
    "expw":   fetch_expw,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FETCHERS.keys()))
    args = ap.parse_args()

    keys = [args.only] if args.only else list(FETCHERS.keys())
    for k in keys:
        FETCHERS[k]()

    print("\n=== Done. Run python3 FACE_JOB/verify_data.py to confirm readiness. ===")


if __name__ == "__main__":
    main()
