# FACE_JOB/emotion/train.py
"""
Emotion classifier training on FER+ ∪ RAF-DB ∪ ExpW (7 classes).

Run from FACE_JOB/:
    python3 -m emotion.train
Envs: IMG_SIZE=64 BATCH=96 EPOCHS=35 LR=3e-4 WD=1e-4 WORKERS=6 RUN_TAG=v1
      DATASETS=ferplus,rafdb,expw   # comma-separated, for ablation
"""
import os, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from FACE_JOB.emotion.model import EmotionWide, NUM_CLASSES, CLASS_NAMES, kaiming_init
from FACE_JOB.shared.datasets.emotion_combined import (
    EmotionSample, CombinedEmotionDS, BalancedSourceSampler, SOURCES, SOURCE_LOSS_WEIGHTS,
)
from FACE_JOB.shared.datasets.ferplus import load_ferplus
from FACE_JOB.shared.datasets.rafdb import load_rafdb
from FACE_JOB.shared.datasets.expw import load_expw

# -------- CONFIG --------
IMG_SIZE = int(os.environ.get("IMG_SIZE", 64))
BATCH    = int(os.environ.get("BATCH", 96))  # divisible by 3 (3 sources)
EPOCHS   = int(os.environ.get("EPOCHS", 35))
LR       = float(os.environ.get("LR", 3e-4))
WD       = float(os.environ.get("WD", 1e-4))
WARMUP   = int(os.environ.get("WARMUP", 2))
WORKERS  = int(os.environ.get("WORKERS", 6))
PATIENCE = int(os.environ.get("PATIENCE", 8))
RUN_TAG  = os.environ.get("RUN_TAG", "v1")
DATASETS = os.environ.get("DATASETS", "ferplus,rafdb,expw").split(",")
SEED     = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE = Path(__file__).parent.parent
CKPT_DIR = Path(__file__).parent / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / f"emotion_{RUN_TAG}.pt"
LOG_PATH  = CKPT_DIR / f"emotion_{RUN_TAG}.log.json"


# -------- LOAD ALL SOURCES --------
def _has_ferplus() -> bool:
    return (BASE / "data/fer/fer2013.csv").exists() and (BASE / "data/fer/fer2013new.csv").exists()

def _has_rafdb() -> bool:
    return (BASE / "data/rafdb/list_patition_label.txt").exists()

def _has_expw() -> bool:
    return (BASE / "data/expw/label.lst").exists()


def load_all() -> list[EmotionSample]:
    samples: list[EmotionSample] = []
    if "ferplus" in DATASETS:
        if _has_ferplus():
            fer_csv = BASE / "data/fer/fer2013.csv"
            new_csv = BASE / "data/fer/fer2013new.csv"
            for img, lab in load_ferplus(fer_csv, new_csv):
                samples.append(EmotionSample(image=img, label=lab, source="ferplus"))
        else:
            print("[warn] ferplus requested but data/fer/fer2013.csv not found — skipping", flush=True)
    if "rafdb" in DATASETS:
        if _has_rafdb():
            root = BASE / "data/rafdb"
            for p, lab in load_rafdb(root, "train"):
                samples.append(EmotionSample(image=p, label=lab, source="rafdb"))
        else:
            print("[warn] rafdb requested but data/rafdb/list_patition_label.txt not found — skipping", flush=True)
    if "expw" in DATASETS:
        if _has_expw():
            root = BASE / "data/expw"
            for p, lab in load_expw(root):
                samples.append(EmotionSample(image=p, label=lab, source="expw"))
        else:
            print("[warn] expw requested but data/expw/label.lst not found — skipping", flush=True)
    if not samples:
        raise RuntimeError("No emotion samples loaded. Run python3 FACE_JOB/download_data.py first.")
    return samples


def split_samples(samples: list[EmotionSample], seed: int = SEED):
    """90/10 per-source split — each source contributes proportionally to val."""
    rng = random.Random(seed)
    by_src: dict[str, list[EmotionSample]] = {s: [] for s in SOURCES}
    for s in samples:
        by_src[s.source].append(s)
    tr, va = [], []
    for src, lst in by_src.items():
        rng.shuffle(lst)
        n_va = int(len(lst) * 0.1)
        va.extend(lst[:n_va]); tr.extend(lst[n_va:])
    return tr, va


# -------- MAIN --------
def main():
    print(f"[{RUN_TAG}] device={device} img={IMG_SIZE} batch={BATCH} epochs={EPOCHS} "
          f"datasets={DATASETS}", flush=True)

    samples = load_all()
    counts = {s: 0 for s in SOURCES}
    for x in samples: counts[x.source] += 1
    print(f"[{RUN_TAG}] total={len(samples)}  per-source={counts}", flush=True)

    tr, va = split_samples(samples)
    tr_ds = CombinedEmotionDS(tr, IMG_SIZE, "train")
    va_ds = CombinedEmotionDS(va, IMG_SIZE, "eval")

    active_sources = [s for s in SOURCES if s in DATASETS]
    if len(active_sources) > 1:
        num_batches = (len(tr) // BATCH)
        sampler = BalancedSourceSampler(tr, batch_size=BATCH, num_batches=num_batches, seed=SEED)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler,
                           num_workers=WORKERS, persistent_workers=(WORKERS > 0))
    else:
        tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=True,
                           num_workers=WORKERS, persistent_workers=(WORKERS > 0))
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False,
                       num_workers=WORKERS, persistent_workers=(WORKERS > 0))

    cnt = np.zeros(NUM_CLASSES)
    for s in tr: cnt[s.label] += 1
    w = 1.0 / np.sqrt(cnt.clip(min=1)); w = w / w.mean()
    class_w = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[{RUN_TAG}] class_w={w.round(2).tolist()}", flush=True)

    source_w = torch.tensor(
        [SOURCE_LOSS_WEIGHTS[s] for s in SOURCES], dtype=torch.float32, device=device
    )

    model = EmotionWide().to(device); kaiming_init(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{RUN_TAG}] params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    def lr_at(ep):
        if ep < WARMUP: return LR * (ep + 1) / WARMUP
        t = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        return 0.5 * LR * (1 + np.cos(np.pi * t))

    ce = nn.CrossEntropyLoss(weight=class_w, reduction="none")

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        P, L = [], []; tot_loss, n = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y).mean()
            tot_loss += loss.item() * x.size(0); n += x.size(0)
            P.append(logits.argmax(1).cpu().numpy()); L.append(y.cpu().numpy())
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
        for x, y, src_idx in tr_ld:
            x, y, src_idx = x.to(device), y.to(device), src_idx.to(device)
            opt.zero_grad()
            logits = model(x)
            per_sample = ce(logits, y)              # (B,)
            weights = source_w[src_idx]             # (B,)
            loss = (per_sample * weights).mean()
            loss.backward(); opt.step()
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
                "val_macro_f1": best_f1,
                "epoch": ep,
                "datasets": DATASETS,
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

    with open(LOG_PATH, "w") as f:
        json.dump({"history": history, "best_val_f1": best_f1, "params": n_params,
                   "datasets": DATASETS}, f, indent=2)
    print(f"[{RUN_TAG}] done. best_val_f1={best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
