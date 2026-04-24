# FACE_JOB/face_det/train.py
"""
Face detector training.

Run from FACE_JOB/:
    python3 -m face_det.train
Envs: IMG_SIZE=320 BATCH=16 EPOCHS=25 LR=3e-4 WD=1e-4 WORKERS=6 RUN_TAG=v1
"""
import os, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

from FACE_JOB.face_det.model import FaceDetector, kaiming_init
from FACE_JOB.face_det.losses import build_targets, fcos_loss
from FACE_JOB.face_det.postprocess import decode, nms
from FACE_JOB.shared.datasets.wider_face import load_filtered

# -------- CONFIG --------
IMG_SIZE = int(os.environ.get("IMG_SIZE", 320))
BATCH    = int(os.environ.get("BATCH", 16))
EPOCHS   = int(os.environ.get("EPOCHS", 25))
LR       = float(os.environ.get("LR", 3e-4))
WD       = float(os.environ.get("WD", 1e-4))
WARMUP   = int(os.environ.get("WARMUP", 2))
WORKERS  = int(os.environ.get("WORKERS", 6))
PATIENCE = int(os.environ.get("PATIENCE", 6))
RUN_TAG  = os.environ.get("RUN_TAG", "v1")
SEED     = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE = Path(__file__).parent.parent
DATA_ROOT = BASE / "data" / "wider_face"
CKPT_DIR = Path(__file__).parent / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / f"face_det_{RUN_TAG}.pt"
LOG_PATH  = CKPT_DIR / f"face_det_{RUN_TAG}.log.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -------- DATASET --------
class WiderDS(Dataset):
    def __init__(self, rows, img_size, mode="train"):
        self.rows = rows
        self.img_size = img_size
        self.mode = mode

    def __len__(self): return len(self.rows)

    def __getitem__(self, k):
        path, bboxes = self.rows[k]
        img = Image.open(path).convert("RGB")
        W, H = img.size
        scale = self.img_size / max(W, H)
        new_w = int(W * scale); new_h = int(H * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        canvas.paste(img, (0, 0))
        scaled = [(x * scale, y * scale, w * scale, h * scale) for (x, y, w, h) in bboxes]

        if self.mode == "train" and random.random() < 0.5:
            canvas = TF.hflip(canvas)
            scaled = [(self.img_size - (x + w), y, w, h) for (x, y, w, h) in scaled]
        if self.mode == "train":
            canvas = TF.adjust_brightness(canvas, random.uniform(0.7, 1.3))
            canvas = TF.adjust_contrast(canvas, random.uniform(0.7, 1.3))

        x_t = TF.to_tensor(canvas)
        x_t = TF.normalize(x_t, IMAGENET_MEAN, IMAGENET_STD)
        return x_t, scaled


def collate(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    bboxes_list = [b[1] for b in batch]
    return xs, bboxes_list


# -------- EVAL --------
@torch.no_grad()
def val_objectness_f1(model, loader, score_thr=0.3, iou_thr=0.5):
    model.eval()
    tp = fp = fn = 0
    for xs, bboxes_list in loader:
        xs = xs.to(device)
        obj, bbox, ctr = model(xs)
        decoded = decode(obj, bbox, ctr, stride=FaceDetector.STRIDE, score_thr=score_thr)
        for (boxes, scores), gts in zip(decoded, bboxes_list):
            if len(boxes):
                keep = nms(boxes.cpu(), scores.cpu(), iou_thr=iou_thr)
                pred = boxes.cpu()[keep].tolist()
            else:
                pred = []
            gt_xyxy = [(x, y, x + w, y + h) for (x, y, w, h) in gts]
            matched = [False] * len(gt_xyxy)
            for pb in pred:
                best_j = -1; best_iou = 0
                for j, gb in enumerate(gt_xyxy):
                    if matched[j]: continue
                    iou = _iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thr and best_j >= 0:
                    tp += 1; matched[best_j] = True
                else:
                    fp += 1
            fn += matched.count(False)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1e-9)


# -------- MAIN --------
def main():
    print(f"[{RUN_TAG}] device={device} img={IMG_SIZE} batch={BATCH} epochs={EPOCHS}", flush=True)

    tr_rows = load_filtered(DATA_ROOT, "train")
    va_rows = load_filtered(DATA_ROOT, "val")
    print(f"[{RUN_TAG}] train={len(tr_rows)} val={len(va_rows)}", flush=True)

    tr_ds = WiderDS(tr_rows, IMG_SIZE, "train")
    va_ds = WiderDS(va_rows, IMG_SIZE, "eval")
    kw = dict(num_workers=WORKERS, persistent_workers=(WORKERS > 0), collate_fn=collate)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, **kw)

    model = FaceDetector().to(device); kaiming_init(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{RUN_TAG}] params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    feat_hw = (IMG_SIZE // FaceDetector.STRIDE, IMG_SIZE // FaceDetector.STRIDE)

    def lr_at(ep):
        if ep < WARMUP: return LR * (ep + 1) / WARMUP
        t = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        return 0.5 * LR * (1 + np.cos(np.pi * t))

    history = []; best_f1 = -1.0; no_improve = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        cur_lr = lr_at(ep)
        for g in opt.param_groups: g["lr"] = cur_lr
        model.train()
        tot_loss, n = 0.0, 0
        for xs, bboxes_list in tr_ld:
            xs = xs.to(device)
            B = xs.size(0)
            obj_t = torch.zeros((B, 1, *feat_hw))
            bbox_t = torch.zeros((B, 4, *feat_hw))
            ctr_t = torch.zeros((B, 1, *feat_hw))
            for i, faces in enumerate(bboxes_list):
                a, b_, c = build_targets(faces, feat_hw, FaceDetector.STRIDE, radius_cells=1)
                obj_t[i], bbox_t[i], ctr_t[i] = a, b_, c
            obj_t, bbox_t, ctr_t = obj_t.to(device), bbox_t.to(device), ctr_t.to(device)

            obj_p, bbox_p, ctr_p = model(xs)
            _, _, _, loss = fcos_loss(obj_p, bbox_p, ctr_p, obj_t, bbox_t, ctr_t)

            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * B; n += B
        tr_loss = tot_loss / max(n, 1)

        val = val_objectness_f1(model, va_ld)
        improved = val["f1"] > best_f1
        if improved:
            best_f1 = val["f1"]; no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "img_size": IMG_SIZE,
                "stride": FaceDetector.STRIDE,
                "normalize_mean": IMAGENET_MEAN,
                "normalize_std": IMAGENET_STD,
                "val_f1": best_f1,
                "epoch": ep,
            }, CKPT_PATH)
        else:
            no_improve += 1

        history.append({"epoch": ep, "tr_loss": tr_loss, "lr": cur_lr, **val})
        flag = " *" if improved else ""
        print(f"[{RUN_TAG}] ep {ep:2d} | tr {tr_loss:.4f} | "
              f"val f1 {val['f1']:.3f} (p {val['precision']:.3f} r {val['recall']:.3f}){flag} | "
              f"lr {cur_lr:.2e} | t {time.time()-t0:.0f}s", flush=True)

        if no_improve >= PATIENCE:
            print(f"[{RUN_TAG}] early stop at ep {ep}", flush=True); break

    with open(LOG_PATH, "w") as f:
        json.dump({"history": history, "best_val_f1": best_f1, "params": n_params}, f, indent=2)
    print(f"[{RUN_TAG}] done. best_val_f1={best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
