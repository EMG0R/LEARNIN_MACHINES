"""OBJECTIFICATION Layer 1 training. Single-file, env-var configured —
mirrors the HAND_JOB/hand_seg/train.py pattern.

Usage (from OBJECTIFICATION/seg/):
    python train.py
Override via env vars:
    IMG_SIZE=320 BATCH=16 EPOCHS=60 LR=3e-4 WORKERS=6 RUN_TAG=v1 \
        DATA_ROOT=../shared/datasets/openimages_v7 SMOKE=0 python train.py

SMOKE=1 runs 2 batches per epoch over 1 epoch — for plumbing checks only.
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from OBJECTIFICATION.seg.augment import SegTransform
from OBJECTIFICATION.seg.classes import NUM_CLASSES
from OBJECTIFICATION.seg.dataset import OpenImagesSegDataset
from OBJECTIFICATION.seg.eval import ConfusionAccumulator, macro_miou, per_class_iou
from OBJECTIFICATION.seg.losses import ce_dice_loss
from OBJECTIFICATION.seg.model import ObjSegNet


# ---------------- CONFIG ----------------
IMG_SIZE  = int(os.environ.get("IMG_SIZE", 320))
BATCH     = int(os.environ.get("BATCH", 16))
EPOCHS    = int(os.environ.get("EPOCHS", 60))
LR        = float(os.environ.get("LR", 3e-4))
WD        = float(os.environ.get("WD", 5e-4))
WORKERS   = int(os.environ.get("WORKERS", 6))
PATIENCE  = int(os.environ.get("PATIENCE", 8))
RUN_TAG   = os.environ.get("RUN_TAG", "v1")
SMOKE     = bool(int(os.environ.get("SMOKE", 0)))
SEED      = 42

DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    str(Path(__file__).resolve().parent.parent / "shared" / "datasets" / "openimages_v7")
))
TRAIN_ROOT = DATA_ROOT / "train"
VAL_ROOT   = DATA_ROOT / "val"

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / f"obj_seg_{RUN_TAG}.pt"
LOG_PATH  = CKPT_DIR / f"obj_seg_{RUN_TAG}.log.json"


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_tf = SegTransform(img_size=IMG_SIZE, mode="train")
    val_tf   = SegTransform(img_size=IMG_SIZE, mode="eval")
    tr_ds = OpenImagesSegDataset(TRAIN_ROOT, transform=train_tf)
    va_ds = OpenImagesSegDataset(VAL_ROOT,   transform=val_tf)
    assert len(tr_ds) > 0, f"no training images under {TRAIN_ROOT}"
    assert len(va_ds) > 0, f"no validation images under {VAL_ROOT}"

    class_freq = tr_ds.class_freq(NUM_CLASSES).astype(np.float64)
    nonzero = class_freq[class_freq > 0]
    median = float(np.median(nonzero)) if len(nonzero) else 1.0
    cw = np.ones(NUM_CLASSES, dtype=np.float32)
    for c in range(1, NUM_CLASSES):
        if class_freq[c] > 0:
            cw[c] = float(np.clip(median / class_freq[c], 0.5, 5.0))
    class_weights = torch.tensor(cw, device=device)

    sample_weights = tr_ds.sample_weights(NUM_CLASSES)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(tr_ds), replacement=True
    )

    kw = dict(num_workers=WORKERS, persistent_workers=(WORKERS > 0))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler, drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, **kw)

    model = ObjSegNet(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = max(1, EPOCHS * len(tr_ld))
    warmup_steps = min(1000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"[{RUN_TAG}] device={device} params={n_params:,} "
          f"train={len(tr_ds)} val={len(va_ds)} batches/ep={len(tr_ld)}", flush=True)
    print(f"[{RUN_TAG}] img={IMG_SIZE} batch={BATCH} epochs={EPOCHS} lr={LR} "
          f"workers={WORKERS} smoke={SMOKE}", flush=True)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        acc = ConfusionAccumulator(num_classes=NUM_CLASSES)
        tot_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce_dice_loss(logits, y, num_classes=NUM_CLASSES,
                                class_weights=class_weights)
            tot_loss += loss.item() * x.size(0); n += x.size(0)
            acc.update(logits.cpu(), y.cpu())
        iou = per_class_iou(acc.confusion)
        return {"loss": tot_loss / max(1, n), "miou": macro_miou(iou),
                "per_class_iou": iou.tolist()}

    history = []
    best_miou = -1.0; no_improve = 0
    t0 = time.time()
    step = 0
    for ep in range(EPOCHS):
        model.train()
        tot_loss, n = 0.0, 0
        t_ep = time.time()
        for bi, (x, y) in enumerate(tr_ld):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce_dice_loss(model(x), y, num_classes=NUM_CLASSES,
                                class_weights=class_weights)
            loss.backward(); opt.step(); sched.step()
            tot_loss += loss.item() * x.size(0); n += x.size(0); step += 1
            if bi % 50 == 0:
                print(f"[{RUN_TAG}] ep {ep} batch {bi:4d}/{len(tr_ld)} | "
                      f"loss {tot_loss/n:.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                      f"t {time.time()-t_ep:.0f}s", flush=True)
            if SMOKE and bi >= 1:
                break
        tr_loss = tot_loss / max(1, n)

        val = evaluate(va_ld)
        improved = val["miou"] > best_miou
        if improved:
            best_miou = val["miou"]; no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "img_size": IMG_SIZE,
                "num_classes": NUM_CLASSES,
                "val_miou": best_miou,
                "epoch": ep,
            }, CKPT_PATH)
        else:
            no_improve += 1

        history.append({"epoch": ep, "tr_loss": tr_loss,
                        "val_loss": val["loss"], "val_miou": val["miou"]})
        flag = " *NEW BEST*" if improved else ""
        print(f"[{RUN_TAG}] ep {ep:2d} | tr {tr_loss:.4f} | "
              f"vl {val['loss']:.4f} | mIoU {val['miou']:.4f}{flag} | "
              f"total {time.time()-t0:.0f}s", flush=True)

        if SMOKE:
            break
        if no_improve >= PATIENCE:
            print(f"[{RUN_TAG}] early stop at epoch {ep}", flush=True); break

    with open(LOG_PATH, "w") as f:
        json.dump({
            "config": {"img_size": IMG_SIZE, "batch": BATCH, "epochs": EPOCHS,
                       "lr": LR, "wd": WD, "workers": WORKERS},
            "params": n_params, "history": history, "best_val_miou": best_miou,
        }, f, indent=2)
    print(f"[{RUN_TAG}] done. best_val_miou={best_miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
