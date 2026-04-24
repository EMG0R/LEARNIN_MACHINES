"""Gesture v3 — from-scratch Wide CNN, tuned for max quality in one night.

Additions over v1/v2:
  - Label smoothing (0.1) in CrossEntropy
  - MixUp (α=0.2) on 50% of batches
  - EMA of model weights (eval on shadow)
  - Linear warmup (2 ep) + cosine LR
  - Gradient clipping (1.0)
  - MPS fp16 autocast (optional, AMP=1 default)
  - RandAugment on top of existing aug pipeline
  - Larger input (128×128 default)

Run from HAND_JOB/gesture:
    python train_v3.py
Envs: IMG_SIZE=128 BATCH=96 EPOCHS=35 LR=4e-4 WARMUP=2 EMA=0.999 MIXUP=0.2 AMP=1 RUN_TAG=v3 RESUME=1

Resume: auto-resumes from `checkpoints/gesture_{RUN_TAG}.last.pt` if present.
To force fresh: RESUME=0 (or delete the last.pt). Config mismatch (img_size etc)
also forces fresh with a warning.
"""
import os, json, random, time, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

# ---------------- CONFIG ----------------
IMG_SIZE     = int(os.environ.get('IMG_SIZE', 128))
BATCH        = int(os.environ.get('BATCH', 96))
EPOCHS       = int(os.environ.get('EPOCHS', 35))
LR           = float(os.environ.get('LR', 4e-4))
WD           = float(os.environ.get('WD', 1e-4))
WARMUP       = int(os.environ.get('WARMUP', 2))
EMA_DECAY    = float(os.environ.get('EMA', 0.999))
MIXUP_ALPHA  = float(os.environ.get('MIXUP', 0.2))
LABEL_SMOOTH = float(os.environ.get('LS', 0.1))
AMP          = os.environ.get('AMP', '1') == '1'
RESUME       = os.environ.get('RESUME', '1') == '1'
PATIENCE     = int(os.environ.get('PATIENCE', 8))
WORKERS      = int(os.environ.get('WORKERS', 6))
RUN_TAG      = os.environ.get('RUN_TAG', 'v3')
SEED         = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

DATA_ROOT  = '../data/hagrid-sample-30k-384p'
ANN_DIR    = f'{DATA_ROOT}/ann_train_val'
IMG_ROOT   = f'{DATA_ROOT}/hagrid_30k'
IMG_PREFIX = 'train_val_'
CLASS_NAMES = ['call','dislike','fist','four','like','mute','ok','one','palm','peace','peace_inverted','rock','stop','stop_inverted','three','three2','two_up','two_up_inverted']
NUM_CLASSES = 18
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CKPT_DIR = 'checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = f'{CKPT_DIR}/gesture_{RUN_TAG}.pt'         # best-so-far (EMA)
LAST_PATH = f'{CKPT_DIR}/gesture_{RUN_TAG}.last.pt'    # full state for resuming
LOG_PATH  = f'{CKPT_DIR}/gesture_{RUN_TAG}.log.json'

# ---------------- INDEX + SPLIT ----------------
def build_index():
    rows = []
    for cls in CLASS_NAMES:
        with open(Path(ANN_DIR) / f'{cls}.json') as f: data = json.load(f)
        img_dir = Path(IMG_ROOT) / f'{IMG_PREFIX}{cls}'
        for image_id, meta in data.items():
            p = img_dir / f'{image_id}.jpg'
            if not p.is_file(): continue
            for bbox, label in zip(meta['bboxes'], meta['labels']):
                if label == 'no_gesture' or label not in CLASS_TO_IDX: continue
                rows.append((str(p), tuple(bbox), CLASS_TO_IDX[label], meta['user_id']))
    return rows

def user_split(rows):
    users = sorted({r[3] for r in rows})
    rng = random.Random(SEED); rng.shuffle(users)
    n = len(users); n_tr = int(n * 0.8); n_va = int(n * 0.1)
    tr_u, va_u = set(users[:n_tr]), set(users[n_tr:n_tr + n_va])
    te_u = set(users[n_tr + n_va:])
    return ([r for r in rows if r[3] in tr_u],
            [r for r in rows if r[3] in va_u],
            [r for r in rows if r[3] in te_u])

# ---------------- TRANSFORMS (RandAugment + heavy) ----------------
def build_tf(img_size, mode):
    if mode == 'eval':
        return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                          T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.RandAugment(num_ops=2, magnitude=8),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

class HagridDS(Dataset):
    def __init__(self, rows, img_size, mode):
        self.rows, self.mode, self.img_size = rows, mode, img_size
        self.tf = build_tf(img_size, mode)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, bbox, label, _ = self.rows[i]
        img = Image.open(path).convert('RGB')
        W, H = img.size; x, y, w, h = bbox
        px, py, pw, ph = x * W, y * H, w * W, h * H
        side = max(pw, ph)
        pad = side * (random.uniform(0.0, 0.25) if self.mode == 'train' else 0.10)
        x0, y0 = max(0, int(px - pad)), max(0, int(py - pad))
        x1, y1 = min(W, int(px + pw + pad)), min(H, int(py + ph + pad))
        if x1 > x0 and y1 > y0: img = img.crop((x0, y0, x1, y1))
        return self.tf(img), label

# ---------------- MODEL (Wide, same as v1) ----------------
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
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n),
        )
    def forward(self, x): return self.head(self.features(x))

def kaiming_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

# ---------------- MIXUP ----------------
def mixup(x, y, alpha, n_classes):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    y1 = F.one_hot(y, n_classes).float()
    y_mix = lam * y1 + (1 - lam) * y1[perm]
    return x_mix, y_mix

def soft_ce(logits, targets_soft, label_smooth=0.0):
    n = logits.size(-1)
    if label_smooth > 0:
        targets_soft = targets_soft * (1 - label_smooth) + label_smooth / n
    logp = F.log_softmax(logits, dim=-1)
    return -(targets_soft * logp).sum(dim=-1).mean()

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
    print(f'[{RUN_TAG}] device={device} img={IMG_SIZE} batch={BATCH} epochs={EPOCHS} lr={LR} mixup={MIXUP_ALPHA} ls={LABEL_SMOOTH} ema={EMA_DECAY} amp={AMP}', flush=True)

    rows = build_index()
    tr, va, te = user_split(rows)
    print(f'[{RUN_TAG}] rows tr/va/te = {len(tr)}/{len(va)}/{len(te)}', flush=True)

    tr_ds = HagridDS(tr, IMG_SIZE, 'train')
    va_ds = HagridDS(va, IMG_SIZE, 'eval')
    te_ds = HagridDS(te, IMG_SIZE, 'eval')
    kw = dict(num_workers=WORKERS, persistent_workers=(WORKERS > 0))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, **kw)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False, **kw)

    cnt = np.zeros(NUM_CLASSES)
    for _, _, li, _ in tr: cnt[li] += 1
    w = 1.0 / np.sqrt(cnt); w = w / w.mean()
    class_w = torch.tensor(w, dtype=torch.float32, device=device)

    model = Wide().to(device); kaiming_init(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[{RUN_TAG}] params: {n_params:,}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ema = EMA(model, EMA_DECAY)

    def lr_at(ep):
        if ep < WARMUP: return LR * (ep + 1) / WARMUP
        t = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        return 0.5 * LR * (1 + np.cos(np.pi * t))

    @torch.no_grad()
    def evaluate(net, loader):
        net.eval()
        tot_loss, n, P, L = 0.0, 0, [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            y_soft = F.one_hot(y, NUM_CLASSES).float()
            tot_loss += soft_ce(logits, y_soft, LABEL_SMOOTH).item() * x.size(0)
            n += x.size(0)
            P.append(logits.argmax(1).cpu().numpy()); L.append(y.cpu().numpy())
        P, L = np.concatenate(P), np.concatenate(L)
        return dict(loss=tot_loss / n, acc=accuracy_score(L, P),
                    f1=f1_score(L, P, average='macro', zero_division=0))

    # ---- resume ----
    start_ep = 0; best_f1 = -1.0; no_improve = 0; history = []
    if RESUME and os.path.exists(LAST_PATH):
        ck = torch.load(LAST_PATH, map_location=device, weights_only=False)
        if ck.get('img_size') != IMG_SIZE:
            print(f'[{RUN_TAG}] RESUME skipped — img_size changed {ck.get("img_size")} → {IMG_SIZE}. Starting fresh.', flush=True)
        else:
            model.load_state_dict(ck['model'])
            ema.shadow.load_state_dict(ck['ema_shadow'])
            opt.load_state_dict(ck['opt'])
            start_ep   = ck['epoch'] + 1
            best_f1    = ck['best_f1']
            no_improve = ck['no_improve']
            history    = ck.get('history', [])
            print(f'[{RUN_TAG}] RESUMED from ep {ck["epoch"]} (best f1 {best_f1:.4f}) → starting ep {start_ep}', flush=True)
    elif RESUME:
        print(f'[{RUN_TAG}] no resume file at {LAST_PATH}, starting fresh.', flush=True)

    t0 = time.time()
    if start_ep >= EPOCHS:
        print(f'[{RUN_TAG}] already trained {start_ep} epochs (>= EPOCHS={EPOCHS}). Bump EPOCHS to continue.', flush=True)
        return
    for ep in range(start_ep, EPOCHS):
        cur_lr = lr_at(ep)
        for g in opt.param_groups: g['lr'] = cur_lr
        model.train()
        tot_loss, n = 0.0, 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            use_mixup = MIXUP_ALPHA > 0 and random.random() < 0.5
            if use_mixup:
                x_in, y_soft = mixup(x, y, MIXUP_ALPHA, NUM_CLASSES)
            else:
                x_in = x; y_soft = F.one_hot(y, NUM_CLASSES).float()

            opt.zero_grad()
            if AMP and device.type == 'mps':
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    logits = model(x_in)
                    if use_mixup:
                        loss = soft_ce(logits, y_soft, LABEL_SMOOTH)
                    else:
                        loss = F.cross_entropy(logits, y, weight=class_w, label_smoothing=LABEL_SMOOTH)
            else:
                logits = model(x_in)
                if use_mixup:
                    loss = soft_ce(logits, y_soft, LABEL_SMOOTH)
                else:
                    loss = F.cross_entropy(logits, y, weight=class_w, label_smoothing=LABEL_SMOOTH)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            tot_loss += loss.item() * x.size(0); n += x.size(0)

        tr_loss = tot_loss / n
        val = evaluate(ema.shadow, va_ld)
        improved = val['f1'] > best_f1
        if improved:
            best_f1 = val['f1']; no_improve = 0
            torch.save({'model_state_dict': ema.shadow.state_dict(), 'class_names': CLASS_NAMES,
                        'img_size': IMG_SIZE, 'normalize_mean': IMAGENET_MEAN,
                        'normalize_std': IMAGENET_STD, 'val_macro_f1': best_f1,
                        'epoch': ep, 'model_kind': 'wide', 'ema': EMA_DECAY,
                        'mixup': MIXUP_ALPHA, 'label_smooth': LABEL_SMOOTH}, CKPT_PATH)
        else:
            no_improve += 1

        history.append({'epoch': ep, 'tr_loss': tr_loss, 'lr': cur_lr, **val})
        flag = ' *' if improved else ''
        print(f"[{RUN_TAG}] ep {ep:2d} | tr {tr_loss:.4f} | vl_ema {val['loss']:.4f} | acc {val['acc']:.4f} | f1 {val['f1']:.4f}{flag} | lr {cur_lr:.2e} | t {time.time()-t0:.0f}s", flush=True)

        torch.save({'model': model.state_dict(), 'ema_shadow': ema.shadow.state_dict(),
                    'opt': opt.state_dict(), 'epoch': ep, 'best_f1': best_f1,
                    'no_improve': no_improve, 'history': history,
                    'img_size': IMG_SIZE}, LAST_PATH)

        if no_improve >= PATIENCE:
            print(f'[{RUN_TAG}] early stop at ep {ep}', flush=True); break

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test = evaluate(model, te_ld)
    print(f"[{RUN_TAG}] TEST | acc {test['acc']:.4f} | f1 {test['f1']:.4f}", flush=True)

    with open(LOG_PATH, 'w') as f:
        json.dump({'config': {'img_size': IMG_SIZE, 'batch': BATCH, 'epochs': EPOCHS,
                              'lr': LR, 'wd': WD, 'warmup': WARMUP, 'ema': EMA_DECAY,
                              'mixup': MIXUP_ALPHA, 'label_smooth': LABEL_SMOOTH, 'amp': AMP},
                   'history': history, 'best_val_f1_ema': best_f1, 'test': test,
                   'params': n_params}, f, indent=2)
    print(f"[{RUN_TAG}] done. best_val_f1_ema={best_f1:.4f} test_f1={test['f1']:.4f}", flush=True)

if __name__ == '__main__':
    main()
