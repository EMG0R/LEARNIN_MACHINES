"""Standalone hand-seg training — iterate here, run from terminal.

Usage (from HAND_JOB/hand_seg):
    python train.py
Override via env vars:
    IMG_SIZE=192 BATCH=32 EPOCHS=30 LR=3e-4 WORKERS=6 RUN_TAG=v1 AUG=default python train.py

AUG=heavy adds webcam-like degradations (JPEG compression, motion blur, gamma jitter,
gaussian noise) on top of the default augmentations — targets domain gap vs. live webcam.
"""
import os, io, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import cv2

# ---------------- CONFIG ----------------
IMG_SIZE = int(os.environ.get('IMG_SIZE', 192))
BATCH    = int(os.environ.get('BATCH', 32))
EPOCHS   = int(os.environ.get('EPOCHS', 30))
LR       = float(os.environ.get('LR', 3e-4))
WD       = float(os.environ.get('WD', 1e-4))
WORKERS  = int(os.environ.get('WORKERS', 6))
PATIENCE = int(os.environ.get('PATIENCE', 6))
RUN_TAG  = os.environ.get('RUN_TAG', 'v1')
AUG      = os.environ.get('AUG', 'default')  # 'default' | 'heavy'
SEED     = 42

FREIHAND_ROOT = Path('../data/freihand/training')
RGB_DIR  = FREIHAND_ROOT / 'rgb'
MASK_DIR = FREIHAND_ROOT / 'mask'
BG_ROOT  = Path('../data/backgrounds')
CKPT_DIR = Path('checkpoints'); CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / f'hand_seg_{RUN_TAG}.pt'
LOG_PATH  = CKPT_DIR / f'hand_seg_{RUN_TAG}.log.json'

N_BASE = 32560
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------- DATASET ----------------
class HandSegDataset(Dataset):
    def __init__(self, indices, bg_paths, img_size, mode='train'):
        self.indices = indices
        self.bg_paths = bg_paths
        self.img_size = img_size
        self.mode = mode

    def __len__(self): return len(self.indices)

    def _load(self, i):
        rgb  = Image.open(RGB_DIR  / f'{i:08d}.jpg').convert('RGB')
        mask = Image.open(MASK_DIR / f'{i:08d}.jpg').convert('L')
        mask = mask.point(lambda v: 255 if v > 127 else 0)
        mask = mask.filter(ImageFilter.MinFilter(3))
        return rgb, mask

    def _skin_tone(self, rgb, mask):
        arr = np.array(rgb, dtype=np.float32)
        m = (np.array(mask, dtype=np.float32) / 255.0)[..., None]
        factor = random.uniform(0.35, 1.0)
        desat = 1.0 - 0.4 * (1.0 - factor)
        mean = arr.mean(axis=-1, keepdims=True)
        recolored = (arr * desat + mean * (1 - desat)) * factor
        arr = arr * (1.0 - m) + recolored * m
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def _paste_bg(self, rgb, mask):
        try:
            bg = Image.open(random.choice(self.bg_paths)).convert('RGB')
        except Exception:
            return rgb
        W, H = rgb.size; bw, bh = bg.size
        scale = max(W/bw, H/bh) * random.uniform(1.0, 1.5)
        nbw, nbh = int(bw*scale), int(bh*scale)
        bg = bg.resize((nbw, nbh), Image.BILINEAR)
        x0 = random.randint(0, max(0, nbw - W)); y0 = random.randint(0, max(0, nbh - H))
        bg = bg.crop((x0, y0, x0 + W, y0 + H))
        if random.random() < 0.3:
            bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        return Image.composite(rgb, bg, mask)

    def _scale_place(self, rgb, mask):
        target = self.img_size
        s = random.uniform(0.3, 1.0)
        new_w = new_h = max(8, int(target*s))
        rgb_s  = rgb.resize((new_w, new_h), Image.BILINEAR)
        mask_s = mask.resize((new_w, new_h), Image.NEAREST)
        try:
            canvas = Image.open(random.choice(self.bg_paths)).convert('RGB').resize((target, target), Image.BILINEAR)
        except Exception:
            canvas = Image.new('RGB', (target, target), (127,127,127))
        mask_canvas = Image.new('L', (target, target), 0)
        x = random.randint(0, target - new_w); y = random.randint(0, target - new_h)
        canvas.paste(rgb_s, (x, y), mask_s)
        mask_canvas.paste(mask_s, (x, y))
        return canvas, mask_canvas

    def _heavy_webcam_aug(self, rgb):
        """Degradations that simulate webcam feed: JPEG, motion blur, gamma, noise."""
        if random.random() < 0.5:
            q = random.randint(35, 85)
            buf = io.BytesIO()
            rgb.save(buf, format='JPEG', quality=q)
            buf.seek(0); rgb = Image.open(buf).convert('RGB')
        if random.random() < 0.3:
            arr = np.array(rgb)
            k = random.choice([3, 5, 7])
            kern = np.zeros((k, k), dtype=np.float32)
            if random.random() < 0.5: kern[k // 2, :] = 1.0 / k    # horizontal motion
            else:                     kern[:, k // 2] = 1.0 / k    # vertical motion
            arr = cv2.filter2D(arr, -1, kern)
            rgb = Image.fromarray(arr)
        if random.random() < 0.5:
            rgb = TF.adjust_gamma(rgb, random.uniform(0.6, 1.5))
        if random.random() < 0.3:
            arr = np.array(rgb, dtype=np.float32)
            arr = arr + np.random.normal(0, random.uniform(3, 12), arr.shape)
            rgb = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return rgb

    def _augment(self, rgb, mask):
        rgb = self._skin_tone(rgb, mask)
        if random.random() < 0.6:
            rgb, mask = self._scale_place(rgb, mask)
        else:
            rgb = self._paste_bg(rgb, mask)
            rgb  = rgb.resize((self.img_size, self.img_size),  Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        if random.random() < 0.5:
            rgb  = TF.hflip(rgb); mask = TF.hflip(mask)
        angle = random.uniform(-20, 20)
        rgb  = TF.rotate(rgb,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        if AUG == 'heavy':
            rgb = TF.adjust_brightness(rgb, random.uniform(0.65, 1.35))
            rgb = TF.adjust_contrast(rgb,   random.uniform(0.65, 1.35))
            rgb = TF.adjust_saturation(rgb, random.uniform(0.7, 1.25))
            rgb = TF.adjust_hue(rgb,        random.uniform(-0.06, 0.06))
            rgb = self._heavy_webcam_aug(rgb)
        else:
            rgb = TF.adjust_brightness(rgb, random.uniform(0.8, 1.2))
            rgb = TF.adjust_contrast(rgb,   random.uniform(0.8, 1.2))
            rgb = TF.adjust_saturation(rgb, random.uniform(0.8, 1.1))
            rgb = TF.adjust_hue(rgb,        random.uniform(-0.04, 0.04))
        return rgb, mask

    def __getitem__(self, k):
        i = self.indices[k]
        rgb, mask = self._load(i)
        if self.mode == 'train':
            rgb, mask = self._augment(rgb, mask)
        else:
            rgb, mask = self._scale_place(rgb, mask)
        rgb  = rgb.resize((self.img_size, self.img_size),  Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        x = TF.to_tensor(rgb); x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        m = TF.to_tensor(mask); m = (m > 0.5).float()
        return x, m

# ---------------- MODEL ----------------
def conv_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )

class HandUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = conv_block(3, 32); self.d2 = conv_block(32, 64)
        self.d3 = conv_block(64, 128); self.d4 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.u3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128,  64, 2, stride=2); self.u2 = conv_block(128,  64)
        self.up1 = nn.ConvTranspose2d(64,   32, 2, stride=2); self.u1 = conv_block(64,   32)
        self.out = nn.Conv2d(32, 1, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        c1 = self.d1(x); c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2)); c4 = self.d4(self.pool(c3))
        u3 = self.u3(torch.cat([self.up3(c4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out(u1)

# ---------------- LOSS ----------------
def dice_loss(logits, targets, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * targets).sum(dim=(2,3)) + eps
    den = p.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps
    return 1 - (num / den).mean()

bce = nn.BCEWithLogitsLoss()
def seg_loss(logits, targets): return bce(logits, targets) + dice_loss(logits, targets)

def iou_score(logits, targets, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * targets).sum(dim=(2,3))
    union = ((p + targets) > 0).float().sum(dim=(2,3))
    return ((inter + eps) / (union + eps)).mean().item()

# ---------------- MAIN ----------------
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    BG_EXTS = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}
    bg_paths = [p for p in BG_ROOT.rglob('*') if p.suffix in BG_EXTS]
    assert len(bg_paths) >= 50, f'need backgrounds in {BG_ROOT}'

    all_idx = list(range(N_BASE))
    rng = random.Random(SEED); rng.shuffle(all_idx)
    n_tr = int(N_BASE*0.90); n_va = int(N_BASE*0.05)
    train_idx = all_idx[:n_tr]; val_idx = all_idx[n_tr:n_tr+n_va]; test_idx = all_idx[n_tr+n_va:]

    tr_ds = HandSegDataset(train_idx, bg_paths, IMG_SIZE, 'train')
    va_ds = HandSegDataset(val_idx,   bg_paths, IMG_SIZE, 'eval')

    kw = dict(num_workers=WORKERS, persistent_workers=(WORKERS>0))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, **kw)

    model = HandUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    print(f'[{RUN_TAG}] device={device} params={n_params:,} batches/ep train={len(tr_ld)} val={len(va_ld)}', flush=True)
    print(f'[{RUN_TAG}] img={IMG_SIZE} batch={BATCH} epochs={EPOCHS} lr={LR} workers={WORKERS} aug={AUG}', flush=True)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        tot_loss, tot_iou, n = 0.0, 0.0, 0
        for x, m in loader:
            x, m = x.to(device), m.to(device)
            logits = model(x)
            tot_loss += seg_loss(logits, m).item() * x.size(0)
            tot_iou  += iou_score(logits, m) * x.size(0)
            n += x.size(0)
        return {'loss': tot_loss/n, 'iou': tot_iou/n}

    history = []; best_iou = -1.0; no_improve = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        tot_loss, n = 0.0, 0
        t_ep = time.time()
        for bi, (x, m) in enumerate(tr_ld):
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            loss = seg_loss(model(x), m)
            loss.backward(); opt.step()
            tot_loss += loss.item()*x.size(0); n += x.size(0)
            if bi % 50 == 0:
                print(f'[{RUN_TAG}] ep {ep} batch {bi:4d}/{len(tr_ld)} | running_loss {tot_loss/n:.4f} | t {time.time()-t_ep:.0f}s', flush=True)
        tr_loss = tot_loss/n
        val = evaluate(va_ld)
        sched.step()

        improved = val['iou'] > best_iou
        if improved:
            best_iou = val['iou']; no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'img_size': IMG_SIZE,
                        'normalize_mean': IMAGENET_MEAN, 'normalize_std': IMAGENET_STD,
                        'val_iou': best_iou, 'epoch': ep}, CKPT_PATH)
        else:
            no_improve += 1

        history.append({'epoch': ep, 'tr_loss': tr_loss, **val})
        flag = ' *NEW BEST*' if improved else ''
        print(f"[{RUN_TAG}] ep {ep:2d} | tr {tr_loss:.4f} | vl {val['loss']:.4f} | iou {val['iou']:.4f}{flag} | lr {opt.param_groups[0]['lr']:.2e} | total {time.time()-t0:.0f}s", flush=True)

        if no_improve >= PATIENCE:
            print(f'[{RUN_TAG}] early stop at epoch {ep}', flush=True); break

    with open(LOG_PATH, 'w') as f:
        json.dump({'config': {'img_size': IMG_SIZE, 'batch': BATCH, 'epochs': EPOCHS, 'lr': LR, 'wd': WD, 'workers': WORKERS},
                   'history': history, 'best_val_iou': best_iou, 'params': n_params}, f, indent=2)
    print(f'[{RUN_TAG}] done. best_val_iou={best_iou:.4f}', flush=True)

if __name__ == '__main__':
    main()
