"""Standalone training script — iterate hyperparams here, sync winner to notebook."""
import os, json, random, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

# ---------------- CONFIG (override via env vars) ----------------
IMG_SIZE     = int(os.environ.get("IMG_SIZE", 96))
BATCH        = int(os.environ.get("BATCH", 128))
EPOCHS       = int(os.environ.get("EPOCHS", 40))
LR           = float(os.environ.get("LR", 3e-4))
WD           = float(os.environ.get("WD", 1e-4))
AUG          = os.environ.get("AUG", "light")        # "light" | "medium" | "heavy"
MODEL        = os.environ.get("MODEL", "small")      # "small" | "wide" | "deep"
PATIENCE     = int(os.environ.get("PATIENCE", 8))
SCHED        = os.environ.get("SCHED", "cosine")     # "cosine" | "step"
WORKERS      = int(os.environ.get("WORKERS", 6))
RUN_TAG      = os.environ.get("RUN_TAG", "run")
SEED         = 42

# ---------------- SETUP ----------------
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_ROOT = "../data/hagrid-sample-30k-384p"
ANN_DIR   = f"{DATA_ROOT}/ann_train_val"
IMG_ROOT  = f"{DATA_ROOT}/hagrid_30k"
IMG_DIR_PREFIX = "train_val_"
CLASS_NAMES = ["call","dislike","fist","four","like","mute","ok","one","palm","peace","peace_inverted","rock","stop","stop_inverted","three","three2","two_up","two_up_inverted"]
NUM_CLASSES = 18
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CKPT_DIR = "checkpoints"; os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = f"{CKPT_DIR}/gesture_{RUN_TAG}.pt"
LOG_PATH  = f"{CKPT_DIR}/gesture_{RUN_TAG}.log.json"

# ---------------- INDEX + SPLIT ----------------
def build_index():
    rows = []
    for cls in CLASS_NAMES:
        with open(Path(ANN_DIR)/f"{cls}.json") as f: data = json.load(f)
        img_dir = Path(IMG_ROOT)/f"{IMG_DIR_PREFIX}{cls}"
        for image_id, meta in data.items():
            p = img_dir/f"{image_id}.jpg"
            if not p.is_file(): continue
            for bbox, label in zip(meta["bboxes"], meta["labels"]):
                if label == "no_gesture" or label not in CLASS_TO_IDX: continue
                rows.append((str(p), tuple(bbox), CLASS_TO_IDX[label], meta["user_id"]))
    return rows

def user_split(rows, seed=SEED):
    users = sorted({r[3] for r in rows})
    rng = random.Random(seed); rng.shuffle(users)
    n = len(users); n_tr = int(n*0.8); n_va = int(n*0.1)
    tr_u = set(users[:n_tr]); va_u = set(users[n_tr:n_tr+n_va]); te_u = set(users[n_tr+n_va:])
    tr = [r for r in rows if r[3] in tr_u]
    va = [r for r in rows if r[3] in va_u]
    te = [r for r in rows if r[3] in te_u]
    return tr, va, te

# ---------------- AUGMENTATIONS ----------------
def build_transforms(img_size, mode):
    if mode == "eval":
        return T.Compose([T.Resize((img_size,img_size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if AUG == "light":
        return T.Compose([
            T.Resize((img_size,img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if AUG == "medium":
        return T.Compose([
            T.Resize((img_size,img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomPerspective(distortion_scale=0.1, p=0.3),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.15, scale=(0.02, 0.10)),
        ])
    # heavy (original)
    return T.Compose([
        T.Resize((img_size,img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomPerspective(distortion_scale=0.15, p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1,1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02,0.15)),
    ])

# ---------------- DATASET ----------------
class HagridDS(Dataset):
    def __init__(self, rows, img_size, mode):
        self.rows = rows; self.mode = mode; self.img_size = img_size
        self.tf = build_transforms(img_size, mode)
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        img_path, bbox, label, _ = self.rows[idx]
        img = Image.open(img_path).convert("RGB")
        W,H = img.size; x,y,w,h = bbox
        px,py,pw,ph = x*W, y*H, w*W, h*H
        side = max(pw, ph)
        pad_frac = random.uniform(0.05, 0.25) if self.mode=="train" else 0.15
        pad = side * pad_frac
        x0 = max(0, int(px-pad)); y0 = max(0, int(py-pad))
        x1 = min(W, int(px+pw+pad)); y1 = min(H, int(py+ph+pad))
        if x1 > x0 and y1 > y0: img = img.crop((x0,y0,x1,y1))
        return self.tf(img), label

# ---------------- MODEL ----------------
class Small(nn.Module):
    """Original 131k-param net (3 blocks)."""
    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.1), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.1), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.1), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,n),
        )
    def forward(self, x): return self.head(self.features(x))

class Wide(nn.Module):
    """4 blocks, channels 32->64->128->256. ~450k params."""
    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Dropout2d(0.1), nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3,32), block(32,64), block(64,128), block(128,256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256,512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512,n),
        )
    def forward(self, x): return self.head(self.features(x))

class Deep(nn.Module):
    """MobileNetV3-small pretrained, head retrained. ~1.5M params."""
    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
        self.net = m
    def forward(self, x): return self.net(x)

def make_model(kind):
    if kind == "small": return Small()
    if kind == "wide":  return Wide()
    if kind == "deep":  return Deep()
    raise ValueError(kind)

def kaiming_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

# ---------------- TRAIN ----------------
def main():
    print(f"[{RUN_TAG}] device={device} img={IMG_SIZE} batch={BATCH} epochs={EPOCHS} lr={LR} aug={AUG} model={MODEL} sched={SCHED}", flush=True)

    rows = build_index()
    tr_rows, va_rows, te_rows = user_split(rows)
    print(f"[{RUN_TAG}] rows train/val/test = {len(tr_rows)} / {len(va_rows)} / {len(te_rows)}", flush=True)

    tr_ds = HagridDS(tr_rows, IMG_SIZE, "train")
    va_ds = HagridDS(va_rows, IMG_SIZE, "eval")
    te_ds = HagridDS(te_rows, IMG_SIZE, "eval")

    kw = dict(num_workers=WORKERS, persistent_workers=(WORKERS>0))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, **kw)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False, **kw)

    cnt = np.zeros(NUM_CLASSES)
    for _,_,li,_ in tr_rows: cnt[li] += 1
    w = 1.0/np.sqrt(cnt); w = w/w.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    model = make_model(MODEL).to(device)
    if MODEL != "deep": kaiming_init(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{RUN_TAG}] params: {n_params:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    if SCHED == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    else:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=9, gamma=0.66)

    history = []
    best_f1 = -1.0; no_improve = 0

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        tot_loss, n, preds, labels = 0.0, 0, [], []
        for x,y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            tot_loss += loss.item()*x.size(0); n += x.size(0)
            preds.append(logits.argmax(1).cpu().numpy()); labels.append(y.cpu().numpy())
        P = np.concatenate(preds); L = np.concatenate(labels)
        return dict(loss=tot_loss/n, acc=accuracy_score(L,P), f1=f1_score(L,P,average="macro",zero_division=0))

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        tot_loss, n = 0.0, 0
        for x,y in tr_ld:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
            tot_loss += loss.item()*x.size(0); n += x.size(0)
        tr_loss = tot_loss/n
        val = evaluate(va_ld)
        sched.step()

        improved = val["f1"] > best_f1
        if improved:
            best_f1 = val["f1"]; no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                        "img_size": IMG_SIZE, "normalize_mean": IMAGENET_MEAN, "normalize_std": IMAGENET_STD,
                        "val_macro_f1": best_f1, "epoch": ep, "model_kind": MODEL}, CKPT_PATH)
        else:
            no_improve += 1

        history.append({"epoch": ep, "tr_loss": tr_loss, **val})
        flag = " *" if improved else ""
        print(f"[{RUN_TAG}] ep {ep:2d} | tr {tr_loss:.4f} | vl {val['loss']:.4f} | acc {val['acc']:.4f} | f1 {val['f1']:.4f}{flag} | t {time.time()-t0:.0f}s", flush=True)

        if no_improve >= PATIENCE:
            print(f"[{RUN_TAG}] early stop at epoch {ep}", flush=True); break

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test = evaluate(te_ld)
    print(f"[{RUN_TAG}] TEST | acc {test['acc']:.4f} | f1 {test['f1']:.4f}", flush=True)

    with open(LOG_PATH, "w") as f:
        json.dump({"config": {"img_size":IMG_SIZE,"batch":BATCH,"epochs":EPOCHS,"lr":LR,"wd":WD,"aug":AUG,"model":MODEL,"sched":SCHED},
                   "history": history, "best_val_f1": best_f1, "test": test, "params": n_params}, f, indent=2)
    print(f"[{RUN_TAG}] done. best_val_f1={best_f1:.4f} test_f1={test['f1']:.4f}", flush=True)

if __name__ == "__main__":
    main()
