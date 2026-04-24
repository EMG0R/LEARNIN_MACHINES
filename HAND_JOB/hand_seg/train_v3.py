"""Hand-seg v3 — from-scratch U-Net, tuned for max quality in one night.

Additions over v1/v2:
  - Focal Tversky + BCE loss (β=0.7 weights false-negatives)
  - Deep supervision (aux heads on u2, u3 decoders)
  - EMA of model weights (eval on shadow weights)
  - Linear warmup (2 ep) + cosine LR
  - Gradient clipping (1.0)
  - MPS fp16 autocast (optional, AMP=1 default)
  - Wider scale range (0.15–1.0) so model sees small/far hands (webcam regime)
  - Heavy webcam degradations (JPEG, motion blur, gamma, noise)

Run from HAND_JOB/hand_seg:
    python train_v3.py
Envs: IMG_SIZE=256 BATCH=16 EPOCHS=25 LR=4e-4 WARMUP=2 EMA=0.999 AMP=1 RUN_TAG=v3 RESUME=1

Resume: auto-resumes from `checkpoints/hand_seg_{RUN_TAG}.last.pt` if present.
To force fresh: RESUME=0 (or delete the last.pt). Config mismatch (img_size etc)
also forces fresh with a warning — different config = different run.
"""
import os, io, json, random, time, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import cv2

# reuse existing dataset class (we'll subclass for wider scale range)
from train import HandSegDataset as _BaseDS, HandUNet as _BaseUNet

# ---------------- CONFIG ----------------
IMG_SIZE = int(os.environ.get('IMG_SIZE', 256))
BATCH    = int(os.environ.get('BATCH', 16))
EPOCHS   = int(os.environ.get('EPOCHS', 25))
LR       = float(os.environ.get('LR', 4e-4))
WD       = float(os.environ.get('WD', 1e-4))
WORKERS  = int(os.environ.get('WORKERS', 6))
PATIENCE = int(os.environ.get('PATIENCE', 6))
WARMUP   = int(os.environ.get('WARMUP', 2))
EMA_DECAY= float(os.environ.get('EMA', 0.999))
AMP      = os.environ.get('AMP', '1') == '1'
RESUME   = os.environ.get('RESUME', '1') == '1'
RUN_TAG  = os.environ.get('RUN_TAG', 'v3')
SEED     = 42

FREIHAND_ROOT = Path('../data/freihand/training')
BG_ROOT  = Path('../data/backgrounds')
CKPT_DIR = Path('checkpoints'); CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / f'hand_seg_{RUN_TAG}.pt'         # best-so-far (EMA weights, for inference)
LAST_PATH = CKPT_DIR / f'hand_seg_{RUN_TAG}.last.pt'    # full state for resuming
LOG_PATH  = CKPT_DIR / f'hand_seg_{RUN_TAG}.log.json'
N_BASE = 32560
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------- DATASET (wide scale + heavy webcam degradations) ----------------
class HandSegDSv3(_BaseDS):
    """Hand can be 15%-100% of canvas; adds JPEG/blur/gamma/noise on train."""
    def _scale_place(self, rgb, mask):
        target = self.img_size
        s = random.uniform(0.15, 1.0)          # wider than base (0.3–1.0)
        new_w = new_h = max(8, int(target * s))
        rgb_s  = rgb.resize((new_w, new_h), Image.BILINEAR)
        mask_s = mask.resize((new_w, new_h), Image.NEAREST)
        try:
            canvas = Image.open(random.choice(self.bg_paths)).convert('RGB').resize((target, target), Image.BILINEAR)
        except Exception:
            canvas = Image.new('RGB', (target, target), (127, 127, 127))
        mask_canvas = Image.new('L', (target, target), 0)
        x = random.randint(0, target - new_w); y = random.randint(0, target - new_h)
        canvas.paste(rgb_s, (x, y), mask_s)
        mask_canvas.paste(mask_s, (x, y))
        return canvas, mask_canvas

    def _webcam_deg(self, rgb):
        if random.random() < 0.5:
            q = random.randint(35, 85)
            buf = io.BytesIO(); rgb.save(buf, format='JPEG', quality=q); buf.seek(0)
            rgb = Image.open(buf).convert('RGB')
        if random.random() < 0.3:
            arr = np.array(rgb); k = random.choice([3, 5, 7])
            kern = np.zeros((k, k), dtype=np.float32)
            if random.random() < 0.5: kern[k // 2, :] = 1.0 / k
            else:                     kern[:, k // 2] = 1.0 / k
            rgb = Image.fromarray(cv2.filter2D(arr, -1, kern))
        if random.random() < 0.5:
            rgb = TF.adjust_gamma(rgb, random.uniform(0.6, 1.5))
        if random.random() < 0.3:
            arr = np.array(rgb, dtype=np.float32)
            arr = arr + np.random.normal(0, random.uniform(3, 12), arr.shape)
            rgb = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return rgb

    def _augment(self, rgb, mask):
        rgb = self._skin_tone(rgb, mask)
        if random.random() < 0.7:
            rgb, mask = self._scale_place(rgb, mask)
        else:
            rgb = self._paste_bg(rgb, mask)
            rgb  = rgb.resize((self.img_size, self.img_size),  Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        if random.random() < 0.5:
            rgb  = TF.hflip(rgb); mask = TF.hflip(mask)
        angle = random.uniform(-25, 25)
        rgb  = TF.rotate(rgb,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        rgb = TF.adjust_brightness(rgb, random.uniform(0.6, 1.4))
        rgb = TF.adjust_contrast(rgb,   random.uniform(0.6, 1.4))
        rgb = TF.adjust_saturation(rgb, random.uniform(0.7, 1.3))
        rgb = TF.adjust_hue(rgb,        random.uniform(-0.06, 0.06))
        rgb = self._webcam_deg(rgb)
        return rgb, mask

# ---------------- MODEL (U-Net with deep supervision) ----------------
class HandUNetDS(_BaseUNet):
    """Adds auxiliary 1×1 heads on u2 and u3 decoder levels for deep supervision."""
    def __init__(self):
        super().__init__()
        self.aux3 = nn.Conv2d(128, 1, 1)
        self.aux2 = nn.Conv2d(64,  1, 1)

    def forward(self, x):
        c1 = self.d1(x); c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2)); c4 = self.d4(self.pool(c3))
        u3 = self.u3(torch.cat([self.up3(c4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        main = self.out(u1)
        if self.training:
            return main, self.aux3(u3), self.aux2(u2)
        return main

# ---------------- LOSS ----------------
def focal_tversky(logits, targets, alpha=0.3, beta=0.7, gamma=0.75, eps=1e-6):
    """α weights FP, β weights FN. β>α pushes toward catching more hand pixels."""
    p = torch.sigmoid(logits)
    tp = (p * targets).sum(dim=(2, 3))
    fn = ((1 - p) * targets).sum(dim=(2, 3))
    fp = (p * (1 - targets)).sum(dim=(2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - tversky) ** gamma).mean()

bce = nn.BCEWithLogitsLoss()
def seg_loss(logits, targets):
    return 0.5 * bce(logits, targets) + 0.5 * focal_tversky(logits, targets)

def deep_sup_loss(main, aux3, aux2, targets):
    """Downsample targets to aux resolution, weighted sum (1.0 / 0.4 / 0.2)."""
    t_aux3 = F.interpolate(targets, size=aux3.shape[-2:], mode='nearest')
    t_aux2 = F.interpolate(targets, size=aux2.shape[-2:], mode='nearest')
    return seg_loss(main, targets) + 0.4 * seg_loss(aux3, t_aux3) + 0.2 * seg_loss(aux2, t_aux2)

def iou_score(logits, targets, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * targets).sum(dim=(2, 3))
    union = ((p + targets) > 0).float().sum(dim=(2, 3))
    return ((inter + eps) / (union + eps)).mean().item()

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

# ---------------- MAIN ----------------
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    BG_EXTS = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}
    bg_paths = [p for p in BG_ROOT.rglob('*') if p.suffix in BG_EXTS]
    assert len(bg_paths) >= 50, f'need backgrounds in {BG_ROOT}'

    idx = list(range(N_BASE)); random.Random(SEED).shuffle(idx)
    n_tr = int(N_BASE * 0.90); n_va = int(N_BASE * 0.05)
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:n_tr + n_va]

    tr_ds = HandSegDSv3(tr_idx, bg_paths, IMG_SIZE, 'train')
    va_ds = HandSegDSv3(va_idx, bg_paths, IMG_SIZE, 'eval')
    kw = dict(num_workers=WORKERS, persistent_workers=(WORKERS > 0))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, **kw)

    model = HandUNetDS().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ema   = EMA(model, EMA_DECAY)

    def lr_at(ep):
        if ep < WARMUP: return LR * (ep + 1) / (WARMUP + 1)
        t = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        return 0.5 * LR * (1 + np.cos(np.pi * t))

    print(f'[{RUN_TAG}] device={device} params={n_params:,} batches/ep tr={len(tr_ld)} va={len(va_ld)} amp={AMP}', flush=True)
    print(f'[{RUN_TAG}] img={IMG_SIZE} batch={BATCH} epochs={EPOCHS} lr={LR} warmup={WARMUP} ema={EMA_DECAY}', flush=True)

    # ---- resume ----
    start_ep = 0; best_iou_init = -1.0; no_improve_init = 0; history_init = []
    if RESUME and LAST_PATH.exists():
        ck = torch.load(LAST_PATH, map_location=device, weights_only=False)
        cfg_now = {'img_size': IMG_SIZE}
        cfg_ck  = {'img_size': ck.get('img_size')}
        if cfg_ck != cfg_now:
            print(f'[{RUN_TAG}] RESUME skipped — config changed {cfg_ck} → {cfg_now}. Starting fresh.', flush=True)
        else:
            model.load_state_dict(ck['model'])
            ema.shadow.load_state_dict(ck['ema_shadow'])
            opt.load_state_dict(ck['opt'])
            start_ep        = ck['epoch'] + 1
            best_iou_init   = ck['best_iou']
            no_improve_init = ck['no_improve']
            history_init    = ck.get('history', [])
            print(f'[{RUN_TAG}] RESUMED from ep {ck["epoch"]} (best iou {best_iou_init:.4f}) → starting ep {start_ep}', flush=True)
    elif RESUME:
        print(f'[{RUN_TAG}] no resume file at {LAST_PATH}, starting fresh.', flush=True)

    @torch.no_grad()
    def evaluate(net):
        net.eval()
        tot_loss, tot_iou, n = 0.0, 0.0, 0
        for x, m in va_ld:
            x, m = x.to(device), m.to(device)
            logits = net(x)
            tot_loss += seg_loss(logits, m).item() * x.size(0)
            tot_iou  += iou_score(logits, m) * x.size(0)
            n += x.size(0)
        return {'loss': tot_loss / n, 'iou': tot_iou / n}

    history = history_init; best_iou = best_iou_init; no_improve = no_improve_init
    t0 = time.time()
    if start_ep >= EPOCHS:
        print(f'[{RUN_TAG}] already trained {start_ep} epochs (>= EPOCHS={EPOCHS}). Nothing to do. Bump EPOCHS to continue.', flush=True)
        return
    for ep in range(start_ep, EPOCHS):
        cur_lr = lr_at(ep)
        for g in opt.param_groups: g['lr'] = cur_lr
        model.train()
        tot_loss, n = 0.0, 0
        t_ep = time.time()
        for bi, (x, m) in enumerate(tr_ld):
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            if AMP and device.type == 'mps':
                with torch.autocast(device_type='mps', dtype=torch.bfloat16):
                    main, a3, a2 = model(x)
                    loss = deep_sup_loss(main, a3, a2, m)
            else:
                main, a3, a2 = model(x)
                loss = deep_sup_loss(main, a3, a2, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            tot_loss += loss.item() * x.size(0); n += x.size(0)
            if bi % 50 == 0:
                print(f'[{RUN_TAG}] ep {ep} batch {bi:4d}/{len(tr_ld)} | run {tot_loss/n:.4f} | lr {cur_lr:.2e} | t {time.time()-t_ep:.0f}s', flush=True)

        val_ema = evaluate(ema.shadow)
        improved = val_ema['iou'] > best_iou
        if improved:
            best_iou = val_ema['iou']; no_improve = 0
            torch.save({'model_state_dict': ema.shadow.state_dict(),
                        'img_size': IMG_SIZE, 'normalize_mean': IMAGENET_MEAN,
                        'normalize_std': IMAGENET_STD, 'val_iou': best_iou, 'epoch': ep,
                        'deep_sup': True, 'ema': EMA_DECAY}, CKPT_PATH)
        else:
            no_improve += 1

        history.append({'epoch': ep, 'tr_loss': tot_loss / n, **val_ema})
        flag = ' *NEW BEST*' if improved else ''
        print(f"[{RUN_TAG}] ep {ep:2d} | tr {tot_loss/n:.4f} | vl_ema {val_ema['loss']:.4f} | iou_ema {val_ema['iou']:.4f}{flag} | total {time.time()-t0:.0f}s", flush=True)

        torch.save({'model': model.state_dict(), 'ema_shadow': ema.shadow.state_dict(),
                    'opt': opt.state_dict(), 'epoch': ep, 'best_iou': best_iou,
                    'no_improve': no_improve, 'history': history,
                    'img_size': IMG_SIZE}, LAST_PATH)

        if no_improve >= PATIENCE:
            print(f'[{RUN_TAG}] early stop at ep {ep}', flush=True); break

    with open(LOG_PATH, 'w') as f:
        json.dump({'config': {'img_size': IMG_SIZE, 'batch': BATCH, 'epochs': EPOCHS,
                              'lr': LR, 'wd': WD, 'warmup': WARMUP, 'ema': EMA_DECAY, 'amp': AMP},
                   'history': history, 'best_val_iou_ema': best_iou, 'params': n_params}, f, indent=2)
    print(f'[{RUN_TAG}] done. best_val_iou_ema={best_iou:.4f}', flush=True)

if __name__ == '__main__':
    main()
