"""
Hand-Seg v6: The "Production Grade" Retrain.
Goal: Stop "random" hand detection by mixing high-quality hand shapes (FreiHAND) 
with messy real-world backgrounds (HaGRID) and forced negative samples (Backgrounds).

Run from project root:
    python3 hand_seg/train_v6.py
"""
import os, io, json, random, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image, ImageFilter, ImageDraw
import torchvision.transforms.functional as TF
import cv2

# --- CONFIG ---
IMG_SIZE  = int(os.environ.get('IMG_SIZE', 256))
BATCH     = int(os.environ.get('BATCH', 16))
EPOCHS    = int(os.environ.get('EPOCHS', 30))
LR        = float(os.environ.get('LR', 4e-4))
WD        = float(os.environ.get('WD', 1e-4))
WARMUP    = int(os.environ.get('WARMUP', 2))
EMA_DECAY = 0.999
AMP       = True
RUN_TAG   = "v6"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Path resolution
BASE = Path(__file__).parent.parent
FREIHAND_ROOT = BASE / "data/freihand/training"
HAGRID_ROOT   = BASE / "data/hagrid-sample-30k-384p"
BG_ROOT       = BASE / "data/backgrounds"
CKPT_DIR      = Path(__file__).parent / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH     = CKPT_DIR / f"hand_seg_{RUN_TAG}.pt"

# --- AUGMENTATION ENGINE ---
class Augmentor:
    @staticmethod
    def skin_tone(rgb, mask):
        arr = np.array(rgb, dtype=np.float32)
        m = (np.array(mask, dtype=np.float32) / 255.0)[..., None]
        factor = random.uniform(0.4, 1.0)
        desat = 1.0 - 0.3 * (1.0 - factor)
        mean = arr.mean(axis=-1, keepdims=True)
        recolored = (arr * desat + mean * (1 - desat)) * factor
        arr = arr * (1.0 - m) + recolored * m
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    @staticmethod
    def webcam_deg(rgb):
        if random.random() < 0.4:
            q = random.randint(30, 80)
            buf = io.BytesIO(); rgb.save(buf, format='JPEG', quality=q); buf.seek(0)
            rgb = Image.open(buf).convert('RGB')
        if random.random() < 0.3:
            arr = np.array(rgb); k = random.choice([3, 5, 7])
            kern = np.zeros((k, k), dtype=np.float32)
            if random.random() < 0.5: kern[k // 2, :] = 1.0 / k
            else:                     kern[:, k // 2] = 1.0 / k
            rgb = Image.fromarray(cv2.filter2D(arr, -1, kern))
        if random.random() < 0.4:
            rgb = TF.adjust_gamma(rgb, random.uniform(0.7, 1.4))
        if random.random() < 0.3:
            arr = np.array(rgb, dtype=np.float32)
            arr += np.random.normal(0, random.uniform(2, 10), arr.shape)
            rgb = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return rgb

# --- DATASETS ---
class BaseHandDS(Dataset):
    def __init__(self, bg_paths, mode='train'):
        self.bg_paths = bg_paths
        self.mode = mode
        self.aug = Augmentor()

    def _scale_place(self, rgb, mask):
        s = random.uniform(0.1, 1.0)
        new_w = new_h = max(16, int(IMG_SIZE * s))
        rgb_s  = rgb.resize((new_w, new_h), Image.BILINEAR)
        mask_s = mask.resize((new_w, new_h), Image.NEAREST)
        try:
            canvas = Image.open(random.choice(self.bg_paths)).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        except:
            canvas = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        mask_canvas = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
        x = random.randint(0, IMG_SIZE - new_w); y = random.randint(0, IMG_SIZE - new_h)
        canvas.paste(rgb_s, (x, y), mask_s)
        mask_canvas.paste(mask_s, (x, y))
        return canvas, mask_canvas

    def _finalize(self, rgb, mask):
        if self.mode == 'train':
            rgb = self.aug.webcam_deg(rgb)
            if random.random() < 0.5:
                rgb = TF.hflip(rgb); mask = TF.hflip(mask)
            angle = random.uniform(-25, 25)
            rgb = TF.rotate(rgb, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        x = TF.to_tensor(rgb.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        m = TF.to_tensor(mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        return x, (m > 0.5).float()

class FreiHandDS(BaseHandDS):
    def __init__(self, indices, bg_paths, mode='train'):
        super().__init__(bg_paths, mode)
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, k):
        i = self.indices[k]
        rgb = Image.open(FREIHAND_ROOT / "rgb" / f"{i:08d}.jpg").convert('RGB')
        mask = Image.open(FREIHAND_ROOT / "mask" / f"{i:08d}.jpg").convert('L')
        mask = mask.point(lambda v: 255 if v > 127 else 0)
        if self.mode == 'train':
            rgb = self.aug.skin_tone(rgb, mask)
            rgb, mask = self._scale_place(rgb, mask)
        return self._finalize(rgb, mask)

class HagridDS(BaseHandDS):
    def __init__(self, rows, bg_paths, mode='train'):
        super().__init__(bg_paths, mode)
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, k):
        path, bboxes = self.rows[k]
        img = Image.open(path).convert('RGB')
        W, H = img.size
        mask = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask)
        for (bx, by, bw, bh) in bboxes:
            draw.rectangle([bx*W, by*H, (bx+bw)*W, (by+bh)*H], fill=255)
        if self.mode == 'train' and random.random() < 0.7:
            img, mask = self._scale_place(img, mask)
        return self._finalize(img, mask)

class NegativeDS(Dataset):
    def __init__(self, bg_paths):
        self.bg_paths = bg_paths
    def __len__(self): return len(self.bg_paths)
    def __getitem__(self, k):
        img = Image.open(self.bg_paths[k]).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        mask = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
        if random.random() < 0.5: img = Augmentor.webcam_deg(img)
        x = TF.to_tensor(img); x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        return x, torch.zeros((1, IMG_SIZE, IMG_SIZE), dtype=torch.float32)

class HandUNetDS(nn.Module):
    def __init__(self):
        super().__init__()
        def cb(ci, co): return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
            nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True))
        self.d1=cb(3,32); self.d2=cb(32,64); self.d3=cb(64,128); self.d4=cb(128,256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256,128,2,2); self.u3=cb(256,128)
        self.up2 = nn.ConvTranspose2d(128,64,2,2);  self.u2=cb(128,64)
        self.up1 = nn.ConvTranspose2d(64,32,2,2);   self.u1=cb(64,32)
        self.out = nn.Conv2d(32,1,1)
        self.aux3 = nn.Conv2d(128,1,1); self.aux2 = nn.Conv2d(64,1,1)
    def forward(self, x):
        c1=self.d1(x); c2=self.d2(self.pool(c1)); c3=self.d3(self.pool(c2)); c4=self.d4(self.pool(c3))
        u3=self.u3(torch.cat([self.up3(c4), c3], 1)); u2=self.u2(torch.cat([self.up2(u3), c2], 1))
        u1=self.u1(torch.cat([self.up1(u2), c1], 1))
        main = self.out(u1)
        if self.training: return main, self.aux3(u3), self.aux2(u2)
        return main

def focal_tversky(logits, targets, alpha=0.3, beta=0.7, gamma=0.75):
    p = torch.sigmoid(logits)
    tp = (p * targets).sum((2,3)); fn = ((1-p)*targets).sum((2,3)); fp = (p*(1-targets)).sum((2,3))
    t = (tp + 1e-6) / (tp + alpha*fp + beta*fn + 1e-6)
    return ((1-t)**gamma).mean()

def compute_loss(main, a3, a2, targets):
    def l(lo, ta): 
        ta_s = F.interpolate(ta, lo.shape[-2:], mode='nearest')
        return 0.5 * F.binary_cross_entropy_with_logits(lo, ta_s) + 0.5 * focal_tversky(lo, ta_s)
    return l(main, targets) + 0.4 * l(a3, targets) + 0.2 * l(a2, targets)

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    bg_paths = sorted([p for p in BG_ROOT.rglob('*') if p.suffix.lower() in ('.jpg','.png')])
    f_idx = list(range(32560)); random.shuffle(f_idx)
    tr_f = FreiHandDS(f_idx[:30000], bg_paths, 'train')
    va_f = FreiHandDS(f_idx[30000:], bg_paths, 'eval')
    h_rows = []
    for f in (HAGRID_ROOT / "ann_train_val").glob("*.json"):
        with open(f) as j: data = json.load(j)
        cls = f.stem
        for img_id, meta in data.items():
            p = HAGRID_ROOT / "hagrid_30k" / f"train_val_{cls}" / f"{img_id}.jpg"
            if p.exists(): h_rows.append((p, meta['bboxes']))
    random.shuffle(h_rows)
    tr_h = HagridDS(h_rows[:25000], bg_paths, 'train')
    va_h = HagridDS(h_rows[25000:27000], bg_paths, 'eval')
    tr_neg = NegativeDS(bg_paths)
    tr_ld = DataLoader(ConcatDataset([tr_f, tr_h, tr_neg]), batch_size=BATCH, shuffle=True, num_workers=4)
    va_ld = DataLoader(ConcatDataset([va_f, va_h]), batch_size=BATCH, shuffle=False)
    model = HandUNetDS().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ema_model = copy.deepcopy(model).eval()
    print(f"[{RUN_TAG}] Device: {device} | Samples: {len(tr_ld.dataset)}")
    best_iou = 0.0
    for ep in range(EPOCHS):
        model.train()
        cur_lr = LR * (ep+1)/(WARMUP+1) if ep < WARMUP else 0.5*LR*(1+np.cos(np.pi*(ep-WARMUP)/(EPOCHS-WARMUP)))
        for g in opt.param_groups: g['lr'] = cur_lr
        for i, (x, m) in enumerate(tr_ld):
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=='mps' else torch.float16):
                main_out, a3, a2 = model(x)
                loss = compute_loss(main_out, a3, a2, m)
            loss.backward(); opt.step()
            with torch.no_grad():
                for s, p in zip(ema_model.parameters(), model.parameters()):
                    s.data.mul_(EMA_DECAY).add_(p.data, alpha=1-EMA_DECAY)
            if i % 100 == 0: print(f"Ep {ep} [{i}/{len(tr_ld)}] Loss: {loss.item():.4f}")
        ema_model.eval(); tot_iou = 0; n = 0
        with torch.no_grad():
            for x, m in va_ld:
                x, m = x.to(device), m.to(device)
                pred = (torch.sigmoid(ema_model(x)) > 0.5).float()
                inter = (pred * m).sum((2,3)); union = ((pred+m)>0).float().sum((2,3))
                tot_iou += ((inter+1e-6)/(union+1e-6)).mean().item() * x.size(0)
                n += x.size(0)
        avg_iou = tot_iou / n
        print(f"Epoch {ep} Val IoU: {avg_iou:.4f}")
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save({'model_state_dict': ema_model.state_dict(), 'img_size': IMG_SIZE, 
                        'normalize_mean': IMAGENET_MEAN, 'normalize_std': IMAGENET_STD}, CKPT_PATH)
            print("  *** New Best ***")
if __name__ == "__main__":
    main()
