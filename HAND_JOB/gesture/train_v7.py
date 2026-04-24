"""Gesture v7 — v4 proven config + heavy webcam augmentation.

Changes from v4 (which was best before v6 broke things):
  - IMG_SIZE back to 96 (original winner, not 112)
  - RandomResizedCrop scale 0.4–1.0 (was 0.7–1.0) → simulates hand at different distances
  - Added _webcam_aug: JPEG compression, motion blur, gamma, Gaussian noise
  - Everything else identical to v4: 4-block Wide CNN, AdamW, cosine LR, user split, class weights

Run from HAND_JOB/gesture:
    python train_v7.py
Envs: IMG_SIZE=96 BATCH=128 EPOCHS=40 LR=3e-4 WARMUP=2 PATIENCE=10 WORKERS=6 RUN_TAG=v7
"""
import io, os, json, random, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

# ---------------- CONFIG ----------------
IMG_SIZE  = int(os.environ.get("IMG_SIZE", 96))
BATCH     = int(os.environ.get("BATCH", 128))
EPOCHS    = int(os.environ.get("EPOCHS", 40))
LR        = float(os.environ.get("LR", 3e-4))
WD        = float(os.environ.get("WD", 1e-4))
WARMUP    = int(os.environ.get("WARMUP", 2))
PATIENCE  = int(os.environ.get("PATIENCE", 10))
WORKERS   = int(os.environ.get("WORKERS", 6))
RUN_TAG   = os.environ.get("RUN_TAG", "v7")
SEED      = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# All HaGRID roots — labels guaranteed identical across all (same CLASS_NAMES list)
# Add a new row here whenever a new HaGRID sample is downloaded
_HAGRID_ROOTS_CFG = [
    ("data/hagrid-sample-30k-384p",  "hagrid_30k"),
    ("data/hagrid-sample-500k-384p", "hagrid_500k"),
]
_BASE = Path(__file__).parent.parent
HAGRID_ROOTS = [
    (_BASE / root, img_sub)
    for root, img_sub in _HAGRID_ROOTS_CFG
    if (_BASE / root / "ann_train_val").exists()
]
CLASS_NAMES = ["call","dislike","fist","four","like","ok","one","palm","peace",
               "peace_inverted","rock","stop","stop_inverted","three","three2","two_up",
               "two_up_inverted","middle_finger"]
NUM_CLASSES  = 18
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / f"gesture_{RUN_TAG}.pt"
LOG_PATH  = CKPT_DIR / f"gesture_{RUN_TAG}.log.json"
WEBCAM_DIR = Path(__file__).parent.parent / "data/webcam"


# ---------------- INDEX + SPLIT ----------------
def build_index():
    rows = []
    # HaGRID studio data — all downloaded roots, labels standardized to CLASS_NAMES
    for data_root, img_sub in HAGRID_ROOTS:
        ann_dir_r  = data_root / "ann_train_val"
        img_root_r = data_root / img_sub
        for cls in CLASS_NAMES:
            ann_file = ann_dir_r / f"{cls}.json"
            if not ann_file.exists():
                continue
            with open(ann_file) as f:
                data = json.load(f)
            img_dir = img_root_r / f"train_val_{cls}"
            for image_id, meta in data.items():
                p = img_dir / f"{image_id}.jpg"
                if not p.is_file():
                    continue
                for bbox, label in zip(meta["bboxes"], meta["labels"]):
                    if label == "no_gesture" or label not in CLASS_TO_IDX:
                        continue
                    rows.append((str(p), tuple(bbox), CLASS_TO_IDX[label], meta["user_id"], False))

    # Webcam data (pre-cropped, already gesture images — bbox=full image)
    webcam_count = 0
    if WEBCAM_DIR.exists():
        for cls in CLASS_NAMES:
            cls_dir = WEBCAM_DIR / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.jpg"):
                # bbox=None signals full-image crop in dataset
                rows.append((str(img_path), None, CLASS_TO_IDX[cls], f"webcam_{cls}", True))
                webcam_count += 1
        if webcam_count:
            print(f"[{RUN_TAG}] Loaded {webcam_count} webcam frames from {WEBCAM_DIR}", flush=True)

    return rows


def user_split(rows):
    # Webcam data always goes to train (it's yours — no need to val on it)
    webcam_tr = [r for r in rows if r[4]]
    studio = [r for r in rows if not r[4]]

    users = sorted({r[3] for r in studio})
    rng = random.Random(SEED); rng.shuffle(users)
    n = len(users); n_tr = int(n * 0.8); n_va = int(n * 0.1)
    tr_u, va_u = set(users[:n_tr]), set(users[n_tr:n_tr + n_va])
    te_u = set(users[n_tr + n_va:])

    tr = [r for r in studio if r[3] in tr_u] + webcam_tr
    va = [r for r in studio if r[3] in va_u]
    te = [r for r in studio if r[3] in te_u]
    return tr, va, te


# ---------------- AUGMENTATION ----------------
def _webcam_aug(img: Image.Image, is_webcam: bool = False) -> Image.Image:
    """
    Simulate real-world conditions: outdoor light, pale skin, variable exposure.
    Studio images get the full treatment. Real webcam frames skip re-compression
    but still get color/lighting shifts so the model sees variety even on your data.
    """
    # JPEG compression artifacts — skip for real webcam (already compressed)
    if not is_webcam and random.random() < 0.25:
        q = random.randint(55, 90)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    # Motion blur
    if random.random() < 0.2:
        arr = np.array(img)
        k = random.choice([3, 5])
        kern = np.zeros((k, k), dtype=np.float32)
        if random.random() < 0.5:
            kern[k // 2, :] = 1.0 / k
        else:
            kern[:, k // 2] = 1.0 / k
        arr = cv2.filter2D(arr, -1, kern)
        img = Image.fromarray(arr)

    # Outdoor / autoexposure gamma — wide range, high prob
    # Bright sun overexposes; shade underexposes; webcam autoexposure overcompensates
    if random.random() < 0.7:
        img = TF.adjust_gamma(img, random.uniform(0.45, 1.8))

    # Brightness swing — outdoor light is much more extreme than studio
    if random.random() < 0.6:
        img = TF.adjust_brightness(img, random.uniform(0.5, 1.7))

    # Contrast — harsh sunlight vs flat overcast
    if random.random() < 0.5:
        img = TF.adjust_contrast(img, random.uniform(0.6, 1.8))

    # Saturation — pale/light skin goes low-saturation; warm sun goes high
    if random.random() < 0.6:
        img = TF.adjust_saturation(img, random.uniform(0.2, 2.0))

    # Hue shift — color temperature varies outdoors (warm afternoon, cool shade)
    if random.random() < 0.4:
        img = TF.adjust_hue(img, random.uniform(-0.12, 0.12))

    # Gaussian noise — sensor noise, especially in shade
    if random.random() < 0.3:
        arr = np.array(img, dtype=np.float32)
        arr += np.random.normal(0, random.uniform(3, 12), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return img


# ---------------- DATASET ----------------
def _eval_tf(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _train_tf(img_size):
    return T.Compose([
        # Wide scale range: handles hand at different distances from camera
        T.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=25),
        # Heavy ColorJitter: outdoor brightness/contrast swings + skin tone variance
        T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.7, hue=0.12),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class HagridDS(Dataset):
    def __init__(self, rows, img_size, mode):
        self.rows = rows
        self.mode = mode
        self.img_size = img_size
        self.tf = _train_tf(img_size) if mode == "train" else _eval_tf(img_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, bbox, label, _, is_webcam = self.rows[i]
        img = Image.open(path).convert("RGB")
        if bbox is not None:
            # HaGRID: crop to annotated bbox with padding
            W, H = img.size
            x, y, w, h = bbox
            px, py, pw, ph = x * W, y * H, w * W, h * H
            side = max(pw, ph)
            pad = side * (random.uniform(0.05, 0.25) if self.mode == "train" else 0.15)
            x0 = max(0, int(px - pad));        y0 = max(0, int(py - pad))
            x1 = min(W, int(px + pw + pad));   y1 = min(H, int(py + ph + pad))
            if x1 > x0 and y1 > y0:
                img = img.crop((x0, y0, x1, y1))
        # webcam images are already cropped to the hand — use as-is
        if self.mode == "train":
            img = _webcam_aug(img, is_webcam=is_webcam)
        return self.tf(img), label


# ---------------- MODEL (Wide — same as v1/v4) ----------------
class Wide(nn.Module):
    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Dropout2d(0.1), nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, n),
        )

    def forward(self, x):
        return self.head(self.features(x))


def kaiming_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)


# ---------------- MAIN ----------------
def main():
    print(f"[{RUN_TAG}] device={device} img={IMG_SIZE} batch={BATCH} epochs={EPOCHS} "
          f"lr={LR} warmup={WARMUP} patience={PATIENCE}", flush=True)

    rows = build_index()
    tr, va, te = user_split(rows)
    print(f"[{RUN_TAG}] rows tr/va/te = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    tr_ds = HagridDS(tr, IMG_SIZE, "train")
    va_ds = HagridDS(va, IMG_SIZE, "eval")
    te_ds = HagridDS(te, IMG_SIZE, "eval")
    kw = dict(num_workers=WORKERS, persistent_workers=(WORKERS > 0))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, **kw)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False, **kw)

    cnt = np.zeros(NUM_CLASSES)
    for _, _, li, *_ in tr: cnt[li] += 1
    w = 1.0 / np.sqrt(cnt); w = w / w.mean()
    class_w = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    model = Wide().to(device); kaiming_init(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{RUN_TAG}] params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    def lr_at(ep):
        if ep < WARMUP: return LR * (ep + 1) / WARMUP
        t = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        return 0.5 * LR * (1 + np.cos(np.pi * t))

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        tot_loss, n, P, L = 0.0, 0, [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tot_loss += criterion(logits, y).item() * x.size(0); n += x.size(0)
            P.append(logits.argmax(1).cpu().numpy())
            L.append(y.cpu().numpy())
        P, L = np.concatenate(P), np.concatenate(L)
        return dict(loss=tot_loss / n, acc=accuracy_score(L, P),
                    f1=f1_score(L, P, average="macro", zero_division=0))

    history = []; best_f1 = -1.0; no_improve = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        cur_lr = lr_at(ep)
        for g in opt.param_groups: g["lr"] = cur_lr
        model.train()
        tot_loss, n = 0.0, 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * x.size(0); n += x.size(0)
        tr_loss = tot_loss / n
        val = evaluate(va_ld)

        improved = val["f1"] > best_f1
        if improved:
            best_f1 = val["f1"]; no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "img_size": IMG_SIZE,
                "normalize_mean": IMAGENET_MEAN,
                "normalize_std": IMAGENET_STD,
                "val_macro_f1": best_f1,
                "epoch": ep,
                "model_kind": "wide",
            }, CKPT_PATH)
        else:
            no_improve += 1

        history.append({"epoch": ep, "tr_loss": tr_loss, "lr": cur_lr, **val})
        flag = " *" if improved else ""
        print(f"[{RUN_TAG}] ep {ep:2d} | tr {tr_loss:.4f} | vl {val['loss']:.4f} | "
              f"acc {val['acc']:.4f} | f1 {val['f1']:.4f}{flag} | "
              f"lr {cur_lr:.2e} | t {time.time()-t0:.0f}s", flush=True)

        if no_improve >= PATIENCE:
            print(f"[{RUN_TAG}] early stop at ep {ep}", flush=True); break

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test = evaluate(te_ld)
    print(f"[{RUN_TAG}] TEST | acc {test['acc']:.4f} | f1 {test['f1']:.4f}", flush=True)

    with open(LOG_PATH, "w") as f:
        json.dump({"config": {"img_size": IMG_SIZE, "batch": BATCH, "epochs": EPOCHS,
                              "lr": LR, "wd": WD, "warmup": WARMUP, "patience": PATIENCE},
                   "history": history, "best_val_f1": best_f1, "test": test,
                   "params": n_params}, f, indent=2)
    print(f"[{RUN_TAG}] done. best_val_f1={best_f1:.4f} test_f1={test['f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
