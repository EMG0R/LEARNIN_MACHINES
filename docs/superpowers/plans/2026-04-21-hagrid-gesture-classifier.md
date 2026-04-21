# HaGRID Gesture Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working Jupyter notebook at `HAND_JOB/gesture_classifier.ipynb` that trains an 18-class gesture classifier on the local HaGRID 30k sample, following the design in `docs/superpowers/specs/2026-04-21-hagrid-gesture-classifier-design.md`.

**Architecture:** Single notebook, top-to-bottom runnable on MPS. Flat indexer over 18 per-class annotation JSONs → user-grouped 80/10/10 split → padded-bbox crop dataset with heavy augmentation → 3-block CNN from scratch → CE with class reweighting + Cosine LR → checkpoint on best val macro-F1 → confusion matrix + test eval + `predict_crop` helper.

**Tech Stack:** PyTorch (MPS), torchvision transforms, Pillow, NumPy, scikit-learn (metrics), matplotlib, Jupyter.

**Verification style:** Notebook code. Each task adds one or more cells and runs them. Instead of pytest, "verification" = run the cell and inspect the concrete output described in each step (shapes, counts, plot contents). All verification outputs are listed explicitly so the engineer can confirm without guessing.

---

## File Structure

- **Create:** `HAND_JOB/gesture_classifier.ipynb` — the full training notebook.
- **Create:** `HAND_JOB/checkpoints/` — directory for saved weights. `.gitignore` it.
- **Modify:** `.gitignore` — add `HAND_JOB/checkpoints/` and `*.pt`.

No other files touched. `CONV.ipynb` is left alone (COCO scaffolding, unrelated).

**Constants used across tasks:**

```python
DATA_ROOT = "training data/hagrid-sample-30k-384p"
ANN_DIR   = f"{DATA_ROOT}/ann_train_val"
IMG_ROOT  = f"{DATA_ROOT}/hagrid_30k"
IMG_DIR_PREFIX = "train_val_"   # e.g. train_val_call
CLASS_NAMES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted",
]
NUM_CLASSES = 18
IMG_SIZE = 64
SEED = 42
CKPT_DIR = "checkpoints"
CKPT_PATH = f"{CKPT_DIR}/gesture_classifier_best.pt"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

The engineer should copy this block into the notebook's config cell (Task 2) and reference the names throughout.

All notebook work runs with the working directory set to `HAND_JOB/`. Launch Jupyter from there.

---

## Task 1: Create empty notebook and set up gitignore

**Files:**
- Create: `HAND_JOB/gesture_classifier.ipynb`
- Create: `HAND_JOB/checkpoints/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create checkpoints directory**

Run from repo root:
```bash
mkdir -p HAND_JOB/checkpoints
touch HAND_JOB/checkpoints/.gitkeep
```

- [ ] **Step 2: Add checkpoints to gitignore**

Append to `.gitignore`:
```
HAND_JOB/checkpoints/*
!HAND_JOB/checkpoints/.gitkeep
*.pt
```

- [ ] **Step 3: Create empty notebook**

From `HAND_JOB/`, launch Jupyter and create `gesture_classifier.ipynb`. Or create via CLI:

```bash
cd HAND_JOB
python -c "import nbformat as nbf; nb = nbf.v4.new_notebook(); nb.cells = [nbf.v4.new_markdown_cell('# HaGRID Gesture Classifier\n\nLayer 3b of the vision pipeline. 18 classes, 64x64 hand crops, from-scratch CNN.')]; nbf.write(nb, 'gesture_classifier.ipynb')"
```

Expected: file exists, opens in Jupyter with one markdown title cell.

- [ ] **Step 4: Verify data is where we expect**

```bash
ls "HAND_JOB/training data/hagrid-sample-30k-384p/ann_train_val" | wc -l
ls "HAND_JOB/training data/hagrid-sample-30k-384p/hagrid_30k" | wc -l
```

Expected: both print `18`.

- [ ] **Step 5: Commit**

```bash
git add .gitignore HAND_JOB/checkpoints/.gitkeep HAND_JOB/gesture_classifier.ipynb
git commit -m "scaffold: create gesture_classifier notebook and checkpoints dir"
```

---

## Task 2: Imports, device, seed, config constants

All remaining tasks add cells to `gesture_classifier.ipynb`. Run each cell after adding it.

- [ ] **Step 1: Add imports cell**

```python
import os, json, math, random, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
```

- [ ] **Step 2: Add device + seed cell**

```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")
```

Expected output: `device: mps` (on Apple Silicon).

- [ ] **Step 3: Add config constants cell**

```python
DATA_ROOT = "training data/hagrid-sample-30k-384p"
ANN_DIR   = f"{DATA_ROOT}/ann_train_val"
IMG_ROOT  = f"{DATA_ROOT}/hagrid_30k"
IMG_DIR_PREFIX = "train_val_"

CLASS_NAMES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted",
]
NUM_CLASSES = 18
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_SIZE = 64

CKPT_DIR = "checkpoints"
CKPT_PATH = f"{CKPT_DIR}/gesture_classifier_best.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

assert len(CLASS_NAMES) == NUM_CLASSES
assert Path(ANN_DIR).is_dir(), f"missing {ANN_DIR} — is the working directory HAND_JOB?"
assert Path(IMG_ROOT).is_dir(), f"missing {IMG_ROOT}"
print("config ok")
```

Expected output: `config ok`.

- [ ] **Step 4: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): imports, device, seed, config constants"
```

---

## Task 3: Index builder

Walk the 18 per-class JSONs. One row per bbox. Drop `no_gesture`.

- [ ] **Step 1: Add index builder cell**

```python
def build_index():
    """Returns list of (img_path: str, bbox_xywh_norm: tuple, label_idx: int, user_id: str)."""
    rows = []
    dropped_no_gesture = 0
    dropped_missing_file = 0
    for cls in CLASS_NAMES:
        ann_path = Path(ANN_DIR) / f"{cls}.json"
        img_dir  = Path(IMG_ROOT) / f"{IMG_DIR_PREFIX}{cls}"
        with open(ann_path) as f:
            data = json.load(f)
        for image_id, meta in data.items():
            img_path = img_dir / f"{image_id}.jpg"
            if not img_path.is_file():
                dropped_missing_file += 1
                continue
            for bbox, label in zip(meta["bboxes"], meta["labels"]):
                if label == "no_gesture":
                    dropped_no_gesture += 1
                    continue
                if label not in CLASS_TO_IDX:
                    # stray label we don't know about — skip
                    continue
                rows.append((str(img_path), tuple(bbox), CLASS_TO_IDX[label], meta["user_id"]))
    print(f"rows: {len(rows)} | dropped no_gesture: {dropped_no_gesture} | dropped missing_file: {dropped_missing_file}")
    return rows

rows = build_index()
```

- [ ] **Step 2: Run and verify output**

Expected: a `rows:` line with a count in the 28k–40k range (one row per labeled hand bbox across the ~30k images). `dropped missing_file` should be 0 or tiny.

- [ ] **Step 3: Add per-class count sanity cell**

```python
label_counts = Counter(r[2] for r in rows)
for i, c in enumerate(CLASS_NAMES):
    print(f"  {c:18s} {label_counts[i]:>5}")
print(f"total: {sum(label_counts.values())}")
```

Expected: every class has a nonzero count; counts are roughly balanced (order-of-magnitude similar per class).

- [ ] **Step 4: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): flat index builder over 18 HaGRID annotation JSONs"
```

---

## Task 4: User-grouped 80/10/10 split

- [ ] **Step 1: Add split cell**

```python
def user_split(rows, train_pct=0.8, val_pct=0.1, seed=SEED):
    """Group by user_id. Deterministic assignment via hash+seed."""
    user_ids = sorted({r[3] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(user_ids)
    n = len(user_ids)
    n_train = int(n * train_pct)
    n_val   = int(n * val_pct)
    train_users = set(user_ids[:n_train])
    val_users   = set(user_ids[n_train:n_train + n_val])
    test_users  = set(user_ids[n_train + n_val:])

    train = [r for r in rows if r[3] in train_users]
    val   = [r for r in rows if r[3] in val_users]
    test  = [r for r in rows if r[3] in test_users]

    # sanity
    assert train_users.isdisjoint(val_users)
    assert train_users.isdisjoint(test_users)
    assert val_users.isdisjoint(test_users)
    return train, val, test, (train_users, val_users, test_users)

train_rows, val_rows, test_rows, (train_u, val_u, test_u) = user_split(rows)
print(f"users  train/val/test: {len(train_u)} / {len(val_u)} / {len(test_u)}")
print(f"rows   train/val/test: {len(train_rows)} / {len(val_rows)} / {len(test_rows)}")
```

Expected: three disjoint user counts, three row counts roughly at 80/10/10 ratio of total.

- [ ] **Step 2: Add per-class-per-split count plot**

```python
def split_class_counts(split_rows):
    c = Counter(r[2] for r in split_rows)
    return [c[i] for i in range(NUM_CLASSES)]

tr_c = split_class_counts(train_rows)
va_c = split_class_counts(val_rows)
te_c = split_class_counts(test_rows)

x = np.arange(NUM_CLASSES)
w = 0.28
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(x - w, tr_c, w, label="train")
ax.bar(x,     va_c, w, label="val")
ax.bar(x + w, te_c, w, label="test")
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right")
ax.set_ylabel("rows"); ax.legend(); ax.set_title("per-class split counts")
plt.tight_layout(); plt.show()

for i, c in enumerate(CLASS_NAMES):
    if va_c[i] < 50 or te_c[i] < 50:
        print(f"WARN: {c} has low eval count (val={va_c[i]}, test={te_c[i]})")
```

Expected: bar chart renders, three colors per class, rough 8:1:1 ratio. Warnings only for classes with <50 in val or test — acceptable given 30k sample.

- [ ] **Step 3: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): user-grouped 80/10/10 split with per-class count sanity"
```

---

## Task 5: Augmentation transforms

- [ ] **Step 1: Add transforms cell**

```python
train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomPerspective(distortion_scale=0.15, p=0.5),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
print("transforms built")
```

Expected: `transforms built`.

- [ ] **Step 2: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): train + eval augmentation transforms"
```

---

## Task 6: Dataset class with padded bbox crop

- [ ] **Step 1: Add dataset class cell**

```python
class HagridGestureDataset(Dataset):
    """Crop hand via HaGRID bbox with padding jitter, resize to IMG_SIZE, apply transforms."""
    def __init__(self, rows, mode: str):
        assert mode in ("train", "eval")
        self.rows = rows
        self.mode = mode
        self.tf = train_tf if mode == "train" else eval_tf

    def __len__(self):
        return len(self.rows)

    def _padded_crop(self, img: Image.Image, bbox_norm):
        W, H = img.size
        x, y, w, h = bbox_norm
        px, py = x * W, y * H
        pw, ph = w * W, h * H
        side = max(pw, ph)
        if self.mode == "train":
            pad_frac = random.uniform(0.05, 0.25)
        else:
            pad_frac = 0.15
        pad = side * pad_frac
        x0 = max(0, int(px - pad))
        y0 = max(0, int(py - pad))
        x1 = min(W, int(px + pw + pad))
        y1 = min(H, int(py + ph + pad))
        if x1 <= x0 or y1 <= y0:
            # degenerate bbox — return full image rather than crash
            return img
        return img.crop((x0, y0, x1, y1))

    def __getitem__(self, idx):
        img_path, bbox_norm, label_idx, _ = self.rows[idx]
        img = Image.open(img_path).convert("RGB")
        crop = self._padded_crop(img, bbox_norm)
        return self.tf(crop), label_idx
```

- [ ] **Step 2: Add dataset smoke test cell**

```python
_smoke = HagridGestureDataset(train_rows[:8], mode="train")
_x, _y = _smoke[0]
print("tensor:", _x.shape, _x.dtype, "label:", _y, CLASS_NAMES[_y])
assert tuple(_x.shape) == (3, IMG_SIZE, IMG_SIZE)
```

Expected: `tensor: torch.Size([3, 64, 64]) torch.float32 label: <int> <class name>`.

- [ ] **Step 3: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): HagridGestureDataset with padded bbox crop"
```

---

## Task 7: DataLoaders

- [ ] **Step 1: Add loaders cell**

```python
BATCH = 128

train_ds = HagridGestureDataset(train_rows, mode="train")
val_ds   = HagridGestureDataset(val_rows,   mode="eval")
test_ds  = HagridGestureDataset(test_rows,  mode="eval")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

xb, yb = next(iter(train_loader))
print("batch:", xb.shape, yb.shape, "labels sample:", yb[:8].tolist())
```

Expected: `batch: torch.Size([128, 3, 64, 64]) torch.Size([128])` + 8 ints in [0, 17].

- [ ] **Step 2: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): train/val/test DataLoaders"
```

---

## Task 8: Visualize an augmented batch

- [ ] **Step 1: Add batch viz cell**

```python
def _denorm(t):
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * s + m).clamp(0, 1)

xb, yb = next(iter(train_loader))
fig, axes = plt.subplots(3, 6, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = _denorm(xb[i]).permute(1, 2, 0).numpy()
    ax.imshow(img); ax.set_title(CLASS_NAMES[yb[i].item()], fontsize=8)
    ax.axis("off")
plt.tight_layout(); plt.show()
```

Expected: 3×6 grid of hand crops. Augmentations should be visible: varied rotation, color, occasional erased patches. Class labels should match visible gestures.

- [ ] **Step 2: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): visualize augmented training batch"
```

---

## Task 9: GestureNet model

- [ ] **Step 1: Add model cell**

```python
class GestureNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout2d(0.1), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(0.1), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.1), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.features(x))
```

- [ ] **Step 2: Add instantiation + param count cell**

```python
model = GestureNet().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,}")
with torch.no_grad():
    _out = model(torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device))
print("output shape:", _out.shape)
assert tuple(_out.shape) == (2, NUM_CLASSES)
```

Expected: param count around 130k–180k. Output shape `torch.Size([2, 18])`.

- [ ] **Step 3: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): GestureNet (3 conv blocks + FC head, Kaiming init)"
```

---

## Task 10: Training + eval functions

- [ ] **Step 1: Add class-weighted loss cell**

```python
train_label_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
for _, _, li, _ in train_rows:
    train_label_counts[li] += 1
class_weights = 1.0 / np.sqrt(train_label_counts)
class_weights = class_weights / class_weights.mean()  # center around 1.0
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
print("class weights min/max:", class_weights.min(), class_weights.max())

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
```

Expected: min/max both within ~0.7 to ~1.4 range (HaGRID is close to balanced).

- [ ] **Step 2: Add train/eval helper cell**

```python
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, total_n = 0.0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {
        "loss": total_loss / total_n,
        "acc":  accuracy_score(labels, preds),
        "f1":   f1_score(labels, preds, average="macro", zero_division=0),
        "preds": preds,
        "labels": labels,
    }

def train_one_epoch(model, loader, opt):
    model.train()
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / total_n
```

- [ ] **Step 3: Smoke test one training step**

Add and run:

```python
_opt = torch.optim.Adam(model.parameters(), lr=3e-4)
_xb, _yb = next(iter(train_loader))
_xb, _yb = _xb.to(device), _yb.to(device)
_opt.zero_grad(); _loss = criterion(model(_xb), _yb); _loss.backward(); _opt.step()
print("smoke loss:", _loss.item())
```

Expected: loss prints a finite number roughly in [1.5, 4.0] (random init on 18 classes has ln(18)≈2.89 baseline).

- [ ] **Step 4: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): class-weighted CE + train/eval helpers"
```

---

## Task 11: Training loop with checkpointing and early stopping

- [ ] **Step 1: Add training config + reset-model cell**

```python
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 6

# fresh model for the real run (smoke test above already stepped the old one)
model = GestureNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
best_f1 = -1.0
epochs_since_improve = 0
os.makedirs(CKPT_DIR, exist_ok=True)
```

- [ ] **Step 2: Add training loop cell**

```python
for epoch in range(EPOCHS):
    tr_loss = train_one_epoch(model, train_loader, optimizer)
    val = evaluate(model, val_loader)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val["loss"])
    history["val_acc"].append(val["acc"])
    history["val_f1"].append(val["f1"])

    improved = val["f1"] > best_f1
    if improved:
        best_f1 = val["f1"]
        epochs_since_improve = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "img_size": IMG_SIZE,
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std":  IMAGENET_STD,
            "val_macro_f1": best_f1,
            "epoch": epoch,
        }, CKPT_PATH)
    else:
        epochs_since_improve += 1

    flag = " *" if improved else ""
    print(f"epoch {epoch:2d} | tr_loss {tr_loss:.4f} | val_loss {val['loss']:.4f} | "
          f"val_acc {val['acc']:.4f} | val_f1 {val['f1']:.4f}{flag}")

    if epochs_since_improve >= EARLY_STOP_PATIENCE:
        print(f"early stop at epoch {epoch} (no val_f1 improvement in {EARLY_STOP_PATIENCE} epochs)")
        break

print(f"best val macro-F1: {best_f1:.4f}")
```

Expected: one line per epoch. `val_f1` should rise from near-random (~0.05) toward ≥0.7 within ~15–25 epochs. A `*` appears the first few epochs then intermittently.

- [ ] **Step 3: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): training loop with best-F1 checkpointing and early stop"
```

---

## Task 12: Training curves

- [ ] **Step 1: Add curves cell**

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ep = np.arange(len(history["train_loss"]))
axes[0].plot(ep, history["train_loss"], label="train")
axes[0].plot(ep, history["val_loss"],   label="val")
axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].plot(ep, history["val_acc"], label="val acc")
axes[1].plot(ep, history["val_f1"],  label="val macro-F1")
axes[1].set_title("val metrics"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout(); plt.show()
```

Expected: two side-by-side plots. Train loss declining, val loss declining then plateauing. Val acc + F1 rising.

- [ ] **Step 2: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): loss and metric curves"
```

---

## Task 13: Confusion matrix and classification report (val set)

- [ ] **Step 1: Load best checkpoint cell**

```python
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"loaded checkpoint from epoch {ckpt['epoch']} with val_f1={ckpt['val_macro_f1']:.4f}")
```

Expected: prints an epoch number and F1.

- [ ] **Step 2: Confusion matrix cell**

```python
val_eval = evaluate(model, val_loader)
cm = confusion_matrix(val_eval["labels"], val_eval["preds"], labels=list(range(NUM_CLASSES)))
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right")
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("val confusion matrix (row-normalized)")
plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()
```

Expected: 18×18 heatmap with a strong diagonal.

- [ ] **Step 3: Classification report cell**

```python
print(classification_report(
    val_eval["labels"], val_eval["preds"],
    labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES,
    digits=3, zero_division=0,
))
```

Expected: per-class precision/recall/F1 table with a macro-avg row matching `best_f1` within floating-point rounding.

- [ ] **Step 4: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): val confusion matrix + classification report"
```

---

## Task 14: Held-out test set evaluation

- [ ] **Step 1: Add test eval cell**

```python
test_eval = evaluate(model, test_loader)
print(f"TEST | loss {test_eval['loss']:.4f} | acc {test_eval['acc']:.4f} | macro-F1 {test_eval['f1']:.4f}")

cm = confusion_matrix(test_eval["labels"], test_eval["preds"], labels=list(range(NUM_CLASSES)))
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right")
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("TEST confusion matrix (row-normalized)")
plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()

print(classification_report(
    test_eval["labels"], test_eval["preds"],
    labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES,
    digits=3, zero_division=0,
))
```

Expected: a single summary line followed by confusion matrix and classification report. Test macro-F1 should be close to val macro-F1 (within a few points); a big gap means split leakage or over-fit.

- [ ] **Step 2: Commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): held-out test evaluation"
```

---

## Task 15: predict_crop helper + inference demo

- [ ] **Step 1: Add predict_crop cell**

```python
def predict_crop(np_crop_bgr: np.ndarray) -> tuple[str, float]:
    """Given a BGR hand crop (any size, HxWx3 uint8), return (label, confidence)."""
    assert np_crop_bgr.ndim == 3 and np_crop_bgr.shape[2] == 3
    rgb = np_crop_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb)
    x = eval_tf(pil).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx])
```

- [ ] **Step 2: Add inference demo cell**

```python
def _row_to_bgr_crop(row):
    img_path, bbox, _, _ = row
    img = Image.open(img_path).convert("RGB")
    # use eval-mode padded crop
    W, H = img.size
    x, y, w, h = bbox
    px, py, pw, ph = x * W, y * H, w * W, h * H
    side = max(pw, ph); pad = side * 0.15
    x0 = max(0, int(px - pad)); y0 = max(0, int(py - pad))
    x1 = min(W, int(px + pw + pad)); y1 = min(H, int(py + ph + pad))
    crop_rgb = np.array(img.crop((x0, y0, x1, y1)))
    return crop_rgb[:, :, ::-1]  # BGR

rng = random.Random(0)
samples = rng.sample(val_rows, 9)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for ax, row in zip(axes.flat, samples):
    bgr = _row_to_bgr_crop(row)
    label, conf = predict_crop(bgr)
    truth = CLASS_NAMES[row[2]]
    ok = "✓" if label == truth else "✗"
    ax.imshow(bgr[:, :, ::-1]); ax.axis("off")
    ax.set_title(f"{ok} pred: {label} ({conf:.2f})\ntrue: {truth}", fontsize=9)
plt.tight_layout(); plt.show()
```

Expected: 3×3 grid of val crops with predicted label + confidence vs. true label. Most should be `✓`.

- [ ] **Step 3: Final commit**

```bash
git add HAND_JOB/gesture_classifier.ipynb
git commit -m "feat(notebook): predict_crop helper and inference demo"
```

---

## Final verification checklist

Before declaring done:

- [ ] Notebook runs top-to-bottom with no errors from a fresh kernel (Kernel → Restart & Run All).
- [ ] `HAND_JOB/checkpoints/gesture_classifier_best.pt` exists.
- [ ] Val macro-F1 ≥ 0.70 (red flag if below — augmentations too aggressive, LR wrong, or data issue).
- [ ] Test macro-F1 within ~5 points of val macro-F1 (bigger gap = leakage or overfit).
- [ ] All 18 classes have nonzero diagonal entries in val and test confusion matrices.
- [ ] No user_id appears in more than one split (asserted in Task 4).
