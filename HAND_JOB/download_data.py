"""
Download all training data and wire it into the training pipeline.

Downloads:
  1. HaGRID 500k (gesture) — 509k images, 16GB, same format as your 30k
  2. EgoHands (seg)         — 4,800 real webcam hand frames with masks, 1.3GB

Total: ~17GB. Run from HAND_JOB/:
    python3 download_data.py

After this completes, run:
    python3 train_all.py
"""
import io, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# ─── STEP 0: install requirements ────────────────────────────────────────────
def pip_install(*packages):
    missing = []
    for pkg in packages:
        mod = pkg.split("[")[0].replace("-", "_")
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[setup] Installing: {' '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

pip_install("huggingface_hub", "scipy", "requests")


import requests
from huggingface_hub import snapshot_download
import scipy.io
import numpy as np
from PIL import Image, ImageDraw


# ─── CANONICAL GESTURE CLASS LIST ────────────────────────────────────────────
# This is THE single source of truth. All datasets map to these names.
GESTURE_CLASSES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted",
    "middle_finger",   # webcam-only
]
# Write this to a shared config file so train_v7.py and collect.py stay in sync
CLASSES_PATH = ROOT / "gesture_classes.json"
CLASSES_PATH.write_text(json.dumps(GESTURE_CLASSES, indent=2))
print(f"[setup] Canonical class list written → {CLASSES_PATH}")


# ─── UTIL ─────────────────────────────────────────────────────────────────────
def download_file(url, dest: Path, desc=""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {desc or dest.name} already exists")
        return dest
    print(f"[download] {desc or dest.name} ← {url}")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    mb = done / 1e6
                    print(f"\r  {pct:5.1f}%  {mb:6.0f} MB / {total/1e6:.0f} MB  "
                          f"({time.time()-t0:.0f}s)", end="", flush=True)
    print()
    return dest


def unzip(src: Path, dest: Path, desc=""):
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip] {desc or dest.name} already unzipped")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {src.name} → {dest}")
    with zipfile.ZipFile(src) as z:
        z.extractall(dest)
    print(f"[unzip] done")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HAGRID 500k
# ═══════════════════════════════════════════════════════════════════════════════
HAGRID_500K_REPO = "cj-mills/hagrid-sample-500k-384p"
HAGRID_500K_DIR  = DATA / "hagrid-sample-500k-384p"

def download_hagrid_500k():
    if HAGRID_500K_DIR.exists() and any(HAGRID_500K_DIR.iterdir()):
        print(f"[skip] HaGRID 500k already at {HAGRID_500K_DIR}")
        return

    print("\n" + "="*60)
    print("DOWNLOADING: HaGRID 500k (~16 GB)")
    print("="*60)
    print("This is a large download. Estimated time: 15-45 min depending on connection.")
    print(f"Destination: {HAGRID_500K_DIR}\n")

    try:
        snapshot_download(
            repo_id=HAGRID_500K_REPO,
            repo_type="dataset",
            local_dir=str(HAGRID_500K_DIR),
            ignore_patterns=["*.parquet"],   # skip parquet if any, we want raw files
        )
        print(f"\n[hagrid-500k] Download complete → {HAGRID_500K_DIR}")
    except Exception as e:
        print(f"\n[hagrid-500k] Download failed: {e}")
        print("You can retry manually: pip install huggingface_hub && "
              f"python3 -c \"from huggingface_hub import snapshot_download; "
              f"snapshot_download('{HAGRID_500K_REPO}', repo_type='dataset', "
              f"local_dir='data/hagrid-sample-500k-384p')\"")
        return

    _inspect_and_adapt_hagrid(HAGRID_500K_DIR, "hagrid_500k")


def _inspect_and_adapt_hagrid(root: Path, img_subdir: str):
    """
    Verify folder structure matches what train_v7.py expects.
    Expected:
      root/ann_train_val/{class}.json
      root/{img_subdir}/train_val_{class}/*.jpg

    If the download used a different layout, adapt it in-place.
    """
    ann_dir = root / "ann_train_val"
    img_dir = root / img_subdir

    if ann_dir.exists() and img_dir.exists():
        n_ann = len(list(ann_dir.glob("*.json")))
        n_img = sum(len(list(d.glob("*.jpg"))) for d in img_dir.iterdir() if d.is_dir())
        print(f"[hagrid] Structure OK: {n_ann} annotation files, {n_img} images")
        return

    # Structure differs — inspect what we got and try to adapt
    print(f"[hagrid] Unexpected structure in {root}, inspecting...")
    for p in sorted(root.iterdir())[:20]:
        print(f"  {p.name}/")
        if p.is_dir():
            for pp in sorted(p.iterdir())[:5]:
                print(f"    {pp.name}")

    # Common alternative: images stored in class subdirectories (ImageFolder format)
    # Check for pattern: root/{class}/{image_id}.jpg
    class_dirs = [d for d in root.iterdir() if d.is_dir() and d.name in GESTURE_CLASSES]
    if class_dirs:
        print(f"[hagrid] Found ImageFolder layout — converting to expected format...")
        ann_dir.mkdir(exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        for cls_dir in class_dirs:
            cls = cls_dir.name
            dest = img_dir / f"train_val_{cls}"
            if not dest.exists():
                shutil.copytree(cls_dir, dest)
            # Synthesize minimal annotation JSON (bbox = full image, label = folder name)
            ann_path = ann_dir / f"{cls}.json"
            if not ann_path.exists():
                ann = {}
                for img_path in cls_dir.glob("*.jpg"):
                    img_id = img_path.stem
                    ann[img_id] = {
                        "bboxes": [[0.0, 0.0, 1.0, 1.0]],
                        "labels": [cls],
                        "user_id": f"hf_{img_id[:8]}",
                    }
                ann_path.write_text(json.dumps(ann))
        print(f"[hagrid] Conversion done.")

    # Check for zip files that need extraction
    zips = list(root.glob("*.zip"))
    if zips:
        print(f"[hagrid] Found {len(zips)} zip files, extracting...")
        for z in zips:
            unzip(z, root / z.stem, z.stem)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EGOHANDS (segmentation)
# ═══════════════════════════════════════════════════════════════════════════════
EGOHANDS_URL    = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
EGOHANDS_ZIP    = DATA / "egohands_data.zip"
EGOHANDS_RAW    = DATA / "egohands_raw"
EGOHANDS_OUT    = DATA / "egohands"          # processed: rgb/ + mask/
MASK_SIZE       = 224
EGOHANDS_TARGET = 5000                       # max frames to process (4,800 available)

def process_egohands():
    if EGOHANDS_OUT.exists() and len(list((EGOHANDS_OUT / "rgb").glob("*.jpg"))) > 100:
        n = len(list((EGOHANDS_OUT / "rgb").glob("*.jpg")))
        print(f"[skip] EgoHands already processed: {n} frames at {EGOHANDS_OUT}")
        return

    download_file(EGOHANDS_URL, EGOHANDS_ZIP, "EgoHands (1.3 GB)")
    unzip(EGOHANDS_ZIP, EGOHANDS_RAW, "EgoHands")

    # Locate the _LABELLED_SAMPLES folder
    labelled = None
    for p in EGOHANDS_RAW.rglob("_LABELLED_SAMPLES"):
        labelled = p; break
    if labelled is None:
        # Alternative: might be in EgoHands/_LABELLED_SAMPLES
        candidates = list(EGOHANDS_RAW.rglob("*.mat"))
        if not candidates:
            print("[egohands] Cannot find annotation .mat files. Skipping.")
            return
        labelled = candidates[0].parent.parent

    (EGOHANDS_OUT / "rgb").mkdir(parents=True, exist_ok=True)
    (EGOHANDS_OUT / "mask").mkdir(parents=True, exist_ok=True)

    print(f"\n[egohands] Converting frames to seg format...")
    saved = 0
    errors = 0

    for video_dir in sorted(labelled.iterdir()):
        if not video_dir.is_dir():
            continue
        mat_files = list(video_dir.glob("*.mat"))
        if not mat_files:
            continue
        mat_path = mat_files[0]

        try:
            mat = scipy.io.loadmat(str(mat_path))
        except Exception as e:
            errors += 1
            continue

        # EgoHands .mat has a 'polygons' variable:
        # shape (100, 4) — 100 frames, 4 possible hands
        # Each entry is (N,2) polygon or empty
        try:
            polygons_data = mat.get("polygons", mat.get("POLYGONS", None))
            if polygons_data is None:
                continue
        except Exception:
            continue

        # Find frame images in the same folder
        frames = sorted(video_dir.glob("frame_*.jpg"))
        if not frames:
            # Try looking one level up
            frames = sorted(video_dir.parent.glob("frame_*.jpg"))

        n_frames = min(len(frames), polygons_data.shape[0])

        for fi in range(n_frames):
            if saved >= EGOHANDS_TARGET:
                break
            frame_path = frames[fi] if fi < len(frames) else None
            if frame_path is None or not frame_path.exists():
                continue

            try:
                img = Image.open(frame_path).convert("RGB")
                W, H = img.size

                # Build binary mask from all hand polygons in this frame
                mask_arr = Image.new("L", (W, H), 0)
                draw = ImageDraw.Draw(mask_arr)
                has_hand = False

                for hand_idx in range(polygons_data.shape[1]):
                    poly = polygons_data[fi, hand_idx]
                    if poly is None or not hasattr(poly, "shape"):
                        continue
                    poly = np.array(poly, dtype=np.float32)
                    if poly.size < 6:   # need at least 3 points
                        continue
                    poly = poly.reshape(-1, 2)
                    pts = [(float(p[0]), float(p[1])) for p in poly]
                    if len(pts) >= 3:
                        draw.polygon(pts, fill=255)
                        has_hand = True

                if not has_hand:
                    continue

                # Resize and save
                img_r  = img.resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR)
                mask_r = mask_arr.resize((MASK_SIZE, MASK_SIZE), Image.NEAREST)

                fname = f"ego_{saved:06d}"
                img_r.save(EGOHANDS_OUT  / "rgb"  / f"{fname}.jpg", quality=95)
                mask_r.save(EGOHANDS_OUT / "mask" / f"{fname}.jpg")
                saved += 1

            except Exception:
                errors += 1
                continue

        if saved >= EGOHANDS_TARGET:
            break

    print(f"[egohands] Saved {saved} frames ({errors} errors) → {EGOHANDS_OUT}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PATCH TRAINING SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════════

def patch_gesture_training():
    src = (ROOT / "gesture" / "train_v7.py").read_text()
    if "HAGRID_ROOTS" in src:
        print("[patch] gesture/train_v7.py already supports multiple roots ✓")
    else:
        print("[patch] WARNING: gesture/train_v7.py missing multi-root support — re-run setup")


def patch_seg_training():
    """
    Update hand_seg/train.py to also load EgoHands frames alongside FreiHAND.
    """
    train_path = ROOT / "hand_seg" / "train.py"
    src = train_path.read_text()

    if "egohands" in src.lower():
        print("[patch] hand_seg/train.py already includes EgoHands")
        return

    egohands_dir = ROOT / "data" / "egohands"
    if not egohands_dir.exists() or not any((egohands_dir / "rgb").glob("*.jpg")):
        print("[patch] EgoHands not downloaded yet — skipping seg patch")
        return

    # Count EgoHands frames
    n_ego = len(list((egohands_dir / "rgb").glob("*.jpg")))

    # Inject EgoHands into HandSegDataset by subclassing the index
    old_main_start = '''def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    BG_EXTS = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}
    bg_paths = [p for p in BG_ROOT.rglob('*') if p.suffix in BG_EXTS]
    assert len(bg_paths) >= 50, f'need backgrounds in {BG_ROOT}'

    all_idx = list(range(N_BASE))'''

    new_main_start = f'''EGOHANDS_RGB  = Path('../data/egohands/rgb')
EGOHANDS_MASK = Path('../data/egohands/mask')
N_EGO = {n_ego}

class EgoHandsDataset(Dataset):
    """EgoHands frames — real webcam hands with auto-generated polygon masks."""
    def __init__(self, indices, img_size, mode='train'):
        self.indices = indices
        self.img_size = img_size
        self.mode = mode

    def __len__(self): return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        name = f'ego_{{i:06d}}'
        rgb  = Image.open(EGOHANDS_RGB  / f'{{name}}.jpg').convert('RGB')
        mask = Image.open(EGOHANDS_MASK / f'{{name}}.jpg').convert('L')
        mask = mask.point(lambda v: 255 if v > 127 else 0)
        rgb  = rgb.resize((self.img_size, self.img_size),  Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        if self.mode == 'train' and AUG == 'heavy':
            rgb = self._aug(rgb)
        x = TF.to_tensor(rgb); x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        m = TF.to_tensor(mask); m = (m > 0.5).float()
        return x, m

    def _aug(self, rgb):
        if random.random() < 0.5:
            rgb = TF.adjust_brightness(rgb, random.uniform(0.65, 1.35))
        if random.random() < 0.3:
            rgb = TF.adjust_gamma(rgb, random.uniform(0.6, 1.5))
        return rgb

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    BG_EXTS = {{'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}}
    bg_paths = [p for p in BG_ROOT.rglob('*') if p.suffix in BG_EXTS]
    assert len(bg_paths) >= 50, f'need backgrounds in {{BG_ROOT}}'

    all_idx = list(range(N_BASE))'''

    # Also inject EgoHands into the DataLoader creation
    old_loaders = '''    tr_ds = HandSegDataset(train_idx, bg_paths, IMG_SIZE, 'train')
    va_ds = HandSegDataset(val_idx,   bg_paths, IMG_SIZE, 'eval')'''

    new_loaders = '''    tr_ds = HandSegDataset(train_idx, bg_paths, IMG_SIZE, 'train')
    va_ds = HandSegDataset(val_idx,   bg_paths, IMG_SIZE, 'eval')

    # Add EgoHands if available
    ego_rgb = Path('../data/egohands/rgb')
    if ego_rgb.exists():
        ego_n = len(list(ego_rgb.glob('*.jpg')))
        ego_idx = list(range(ego_n))
        ego_tr_n = int(ego_n * 0.9)
        random.shuffle(ego_idx)
        ego_tr = EgoHandsDataset(ego_idx[:ego_tr_n], IMG_SIZE, 'train')
        ego_va = EgoHandsDataset(ego_idx[ego_tr_n:], IMG_SIZE, 'eval')
        from torch.utils.data import ConcatDataset
        tr_ds = ConcatDataset([tr_ds, ego_tr])
        va_ds = ConcatDataset([va_ds, ego_va])
        print(f'[{{RUN_TAG}}] EgoHands: +{{ego_tr_n}} train, +{{ego_n - ego_tr_n}} val', flush=True)'''

    if old_main_start not in src:
        print("[patch] hand_seg/train.py structure changed — skipping EgoHands patch")
        return

    src = src.replace(old_main_start, new_main_start)
    if old_loaders in src:
        src = src.replace(old_loaders, new_loaders)
    train_path.write_text(src)
    print(f"[patch] hand_seg/train.py updated to include {n_ego} EgoHands frames")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("DATA DOWNLOAD + SETUP")
    print("="*60)
    print(f"Target: {DATA}")
    print(f"Free space: ", end="")
    os.system("df -h . | tail -1 | awk '{print $4}'")
    print()

    # 1. HaGRID 500k
    download_hagrid_500k()

    # 2. EgoHands
    process_egohands()

    # 3. Patch training scripts
    print("\n[patch] Updating training scripts...")
    patch_gesture_training()
    patch_seg_training()

    # 4. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    hagrid_30k_n = sum(
        len(list((DATA / "hagrid-sample-30k-384p" / "hagrid_30k" / d).glob("*.jpg")))
        for d in (DATA / "hagrid-sample-30k-384p" / "hagrid_30k").iterdir()
        if d.is_dir()
    ) if (DATA / "hagrid-sample-30k-384p" / "hagrid_30k").exists() else 0

    hagrid_500k_dir = DATA / "hagrid-sample-500k-384p"
    hagrid_500k_n = 0
    for sub in ["hagrid_500k", "hagrid_30k"]:
        d = hagrid_500k_dir / sub
        if d.exists():
            hagrid_500k_n = sum(len(list(x.glob("*.jpg"))) for x in d.iterdir() if x.is_dir())
            break

    ego_n = len(list((DATA / "egohands" / "rgb").glob("*.jpg"))) \
            if (DATA / "egohands" / "rgb").exists() else 0
    webcam_n = sum(
        len(list((DATA / "webcam" / c).glob("*.jpg")))
        for c in GESTURE_CLASSES
        if (DATA / "webcam" / c).exists()
    )

    print(f"Gesture training data:")
    print(f"  HaGRID 30k   : {hagrid_30k_n:>8,} images")
    print(f"  HaGRID 500k  : {hagrid_500k_n:>8,} images")
    print(f"  Your webcam  : {webcam_n:>8,} images")
    print(f"  ─────────────────────────")
    print(f"  TOTAL gesture: {hagrid_30k_n + hagrid_500k_n + webcam_n:>8,} images")
    print()
    print(f"Seg training data:")
    print(f"  FreiHAND     : {130240:>8,} images")
    print(f"  EgoHands     : {ego_n:>8,} images")
    print(f"  ─────────────────────────")
    print(f"  TOTAL seg    : {130240 + ego_n:>8,} images")
    print()
    print("Next steps:")
    print("  1. Record your gestures: python3 collect.py")
    print("  2. Train everything:     python3 train_all.py")
    print("  3. Launch app:           python3 -m live_app.app")


if __name__ == "__main__":
    main()
