# FACE_JOB/face_parts/train.py
"""
Face-part segmentation training (5-class on CelebAMask-HQ).

Run from FACE_JOB/:
    python3 -m face_parts.train
Envs: IMG_SIZE=192 BATCH=16 EPOCHS=30 LR=4e-4 WD=1e-4 WORKERS=6 RUN_TAG=v1
"""
import os, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

from FACE_JOB.face_parts.model import FacePartsUNet, NUM_CLASSES, CLASS_NAMES, kaiming_init
from FACE_JOB.shared.datasets.celeba_mask import merge_parts, list_samples, image_path

# -------- CONFIG --------
IMG_SIZE = int(os.environ.get("IMG_SIZE", 192))
BATCH    = int(os.environ.get("BATCH", 16))
EPOCHS   = int(os.environ.get("EPOCHS", 30))
LR       = float(os.environ.get("LR", 4e-4))
WD       = float(os.environ.get("WD", 1e-4))
WARMUP   = int(os.environ.get("WARMUP", 2))
WORKERS  = int(os.environ.get("WORKERS", 6))
PATIENCE = int(os.environ.get("PATIENCE", 6))
RUN_TAG  = os.environ.get("RUN_TAG", "v1")
SEED     = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE = Path(__file__).parent.parent
DATA_ROOT = BASE / "data" / "celeba_mask" / "CelebAMask-HQ"
CKPT_DIR = Path(__file__).parent / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / f"face_parts_{RUN_TAG}.pt"
LOG_PATH  = CKPT_DIR / f"face_parts_{RUN_TAG}.log.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Class weights: eye classes are tiny (~0.5% of pixels), skin is huge (~30%),
# mouth in between. Weight inversely to frequency.
CLASS_WEIGHTS = torch.tensor([0.3, 2.5, 2.5, 1.5, 1.0], dtype=torch.float32)


# -------- DATASET --------
class FacePartsDS(Dataset):
    def __init__(self, sample_ids, img_size, mode="train"):
        self.ids = sample_ids
        self.img_size = img_size
        self.mode = mode

    def __len__(self): return len(self.ids)

    def __getitem__(self, k):
        sid = self.ids[k]
        rgb = Image.open(image_path(DATA_ROOT, sid)).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        mask_np = merge_parts(DATA_ROOT, sid)
        mask = Image.fromarray(mask_np).resize((self.img_size, self.img_size), Image.NEAREST)

        if self.mode == "train":
            if random.random() < 0.5:
                rgb = TF.hflip(rgb); mask = TF.hflip(mask)
                arr = np.array(mask)
                lmask = arr == 1; rmask = arr == 2
                arr[lmask] = 2; arr[rmask] = 1
                mask = Image.fromarray(arr)
            if random.random() < 0.7:
                rgb = TF.adjust_brightness(rgb, random.uniform(0.7, 1.3))
            if random.random() < 0.5:
                rgb = TF.adjust_contrast(rgb, random.uniform(0.7, 1.3))

        x = TF.to_tensor(rgb); x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        y = torch.from_numpy(np.array(mask)).long()
        return x, y


# -------- LOSS --------
def multiclass_dice_loss(logits, targets):
    """logits: (B, C, H, W); targets: (B, H, W) with class ids."""
    C = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum((2, 3))
    denom = probs.sum((2, 3)) + onehot.sum((2, 3)) + 1e-6
    dice = 2 * inter / denom
    return 1.0 - dice.mean()


def compute_loss(main, a3, a2, targets, class_w):
    def one(lo, ta):
        ta_s = F.interpolate(ta.unsqueeze(1).float(), lo.shape[-2:], mode="nearest").squeeze(1).long()
        return 0.5 * F.cross_entropy(lo, ta_s, weight=class_w) + 0.5 * multiclass_dice_loss(lo, ta_s)
    return one(main, targets) + 0.4 * one(a3, targets) + 0.2 * one(a2, targets)


# -------- EVAL --------
@torch.no_grad()
def per_class_iou(model, loader):
    model.eval()
    inter = torch.zeros(NUM_CLASSES); union = torch.zeros(NUM_CLASSES)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        for c in range(NUM_CLASSES):
            pc = p == c; yc = y == c
            inter[c] += (pc & yc).sum().cpu()
            union[c] += (pc | yc).sum().cpu()
    iou = inter / union.clamp(min=1)
    return {CLASS_NAMES[c]: iou[c].item() for c in range(NUM_CLASSES)}


# -------- MAIN --------
def main():
    print(f"[{RUN_TAG}] device={device} img={IMG_SIZE} batch={BATCH} epochs={EPOCHS}", flush=True)

    all_ids = list_samples(DATA_ROOT, max_id=30000)
    random.Random(SEED).shuffle(all_ids)
    n_tr = int(len(all_ids) * 0.9)
    tr_ids, va_ids = all_ids[:n_tr], all_ids[n_tr:]
    print(f"[{RUN_TAG}] train={len(tr_ids)} val={len(va_ids)}", flush=True)

    tr_ds = FacePartsDS(tr_ids, IMG_SIZE, "train")
    va_ds = FacePartsDS(va_ids, IMG_SIZE, "eval")
    kw = dict(num_workers=WORKERS, persistent_workers=(WORKERS > 0))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, **kw)

    model = FacePartsUNet().to(device); kaiming_init(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{RUN_TAG}] params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    class_w = CLASS_WEIGHTS.to(device)

    def lr_at(ep):
        if ep < WARMUP: return LR * (ep + 1) / WARMUP
        t = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        return 0.5 * LR * (1 + np.cos(np.pi * t))

    history = []; best_miou = -1.0; no_improve = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        cur_lr = lr_at(ep)
        for g in opt.param_groups: g["lr"] = cur_lr
        model.train()
        tot_loss, n = 0.0, 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            main, a3, a2 = model(x)
            loss = compute_loss(main, a3, a2, y, class_w)
            loss.backward(); opt.step()
            tot_loss += loss.item() * x.size(0); n += x.size(0)
        tr_loss = tot_loss / max(n, 1)

        iou_by_class = per_class_iou(model, va_ld)
        fg_miou = np.mean([iou_by_class[c] for c in CLASS_NAMES if c != "background"])

        improved = fg_miou > best_miou
        if improved:
            best_miou = fg_miou; no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "img_size": IMG_SIZE,
                "class_names": CLASS_NAMES,
                "normalize_mean": IMAGENET_MEAN,
                "normalize_std": IMAGENET_STD,
                "val_mean_iou_fg": best_miou,
                "val_iou_by_class": iou_by_class,
                "epoch": ep,
            }, CKPT_PATH)
        else:
            no_improve += 1

        history.append({"epoch": ep, "tr_loss": tr_loss, "lr": cur_lr, "iou": iou_by_class, "fg_miou": fg_miou})
        flag = " *" if improved else ""
        print(f"[{RUN_TAG}] ep {ep:2d} | tr {tr_loss:.4f} | fg_mIoU {fg_miou:.3f} "
              f"(eyeL {iou_by_class['eye_L']:.2f} eyeR {iou_by_class['eye_R']:.2f} "
              f"mouth {iou_by_class['mouth']:.2f} skin {iou_by_class['face_skin']:.2f}){flag} | "
              f"lr {cur_lr:.2e} | t {time.time()-t0:.0f}s", flush=True)

        if no_improve >= PATIENCE:
            print(f"[{RUN_TAG}] early stop at ep {ep}", flush=True); break

    with open(LOG_PATH, "w") as f:
        json.dump({"history": history, "best_fg_miou": best_miou, "params": n_params}, f, indent=2)
    print(f"[{RUN_TAG}] done. best_fg_miou={best_miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
