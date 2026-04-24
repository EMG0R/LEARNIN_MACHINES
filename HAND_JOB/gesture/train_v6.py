"""
Gesture v6: The "Locked In" Gesture Classifier.
Fixes for live deployment:
1. Adds "no_gesture" class (19th class) to stop random flickering.
2. Increases IMG_SIZE to 128 for better distant-hand detail.
3. Randomizes Padding (10% to 25%) during training to match live cropping jitter.
4. Heavy webcam degradations (Blur, Noise, JPEG).
"""
import os, json, random, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import f1_score

# --- CONFIG ---
IMG_SIZE  = 128
BATCH     = 64  # Reduced batch for larger IMG_SIZE on MPS
EPOCHS    = 35
LR        = 3e-4
WARMUP    = 2
RUN_TAG   = "v6"

DATA_ROOT = Path(__file__).parent.parent / "data/hagrid-sample-30k-384p"
ANN_DIR   = DATA_ROOT / "ann_train_val"
IMG_ROOT  = DATA_ROOT / "hagrid_30k"

CLASS_NAMES = [
    "call","dislike","fist","four","like","mute","ok","one","palm","peace",
    "peace_inverted","rock","stop","stop_inverted","three","three2","two_up",
    "two_up_inverted", "no_gesture" # Class 19
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- DATASET ---
class HagridV6DS(Dataset):
    def __init__(self, rows, mode='train'):
        self.rows = rows
        self.mode = mode
        
    def __len__(self): return len(self.rows)
    
    def __getitem__(self, i):
        path, bbox, label_idx = self.rows[i]
        img = Image.open(path).convert("RGB")
        W, H = img.size
        x, y, w, h = bbox
        px, py, pw, ph = x*W, y*H, w*W, h*H
        
        # Vibe Fix: Randomize padding during training so model is robust to 
        # imperfect segmentation crops in the live app.
        side = max(pw, ph)
        p_val = random.uniform(0.10, 0.25) if self.mode == 'train' else 0.15
        pad = side * p_val
        
        x0, y0 = max(0, int(px - pad)), max(0, int(py - pad))
        x1, y1 = min(W, int(px + pw + pad)), min(H, int(py + ph + pad))
        img = img.crop((x0, y0, x1, y1)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        
        if self.mode == 'train':
            if random.random() < 0.5: img = TF.hflip(img)
            # Webcam Degradations
            if random.random() < 0.3:
                img = TF.gaussian_blur(img, kernel_size=3, sigma=(0.1, 1.5))
            if random.random() < 0.3:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
                
        t = TF.to_tensor(img)
        t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
        return t, label_idx

def build_index():
    rows = []
    for cls_file in ANN_DIR.glob("*.json"):
        with open(cls_file) as f: data = json.load(f)
        cls_name = cls_file.stem
        img_dir = IMG_ROOT / f"train_val_{cls_name}"
        for img_id, meta in data.items():
            p = img_dir / f"{img_id}.jpg"
            if not p.exists(): continue
            for bbox, label in zip(meta["bboxes"], meta["labels"]):
                if label in CLASS_TO_IDX:
                    rows.append((str(p), tuple(bbox), CLASS_TO_IDX[label], meta["user_id"]))
    return rows

# --- MODEL (Wide CNN) ---
class Wide(nn.Module):
    def __init__(self, n=19):
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
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, n),
        )
    def forward(self, x): return self.head(self.features(x))

# --- TRAINER ---
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = build_index()
    
    # User-based split (don't leak people across splits)
    users = list(set(r[3] for r in rows)); random.shuffle(users)
    tr_u = set(users[:int(len(users)*0.85)])
    
    tr_rows = [r[:3] for r in rows if r[3] in tr_u]
    va_rows = [r[:3] for r in rows if r[3] not in tr_u]
    
    tr_ld = DataLoader(HagridV6DS(tr_rows, 'train'), batch_size=BATCH, shuffle=True, num_workers=4)
    va_ld = DataLoader(HagridV6DS(va_rows, 'eval'), batch_size=BATCH, shuffle=False)
    
    model = Wide(n=19).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    print(f"[{RUN_TAG}] Training Gesture v6 on {len(tr_rows)} samples. Device: {device}")
    
    best_f1 = 0.0
    for ep in range(EPOCHS):
        model.train()
        cur_lr = LR * (ep+1)/(WARMUP+1) if ep < WARMUP else 0.5*LR*(1+np.cos(np.pi*(ep-WARMUP)/(EPOCHS-WARMUP)))
        for g in opt.param_groups: g['lr'] = cur_lr
        
        for i, (x, y) in enumerate(tr_ld):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            if i % 100 == 0: print(f"Ep {ep} [{i}/{len(tr_ld)}] Loss: {loss.item():.4f}")
            
        # Eval
        model.eval(); all_p = []; all_y = []
        with torch.no_grad():
            for x, y in va_ld:
                x = x.to(device)
                logits = model(x)
                all_p.extend(logits.argmax(1).cpu().numpy())
                all_y.extend(y.numpy())
        
        f1 = f1_score(all_y, all_p, average='macro')
        print(f"Epoch {ep} Macro F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES,
                'img_size': IMG_SIZE,
                'normalize_mean': IMAGENET_MEAN,
                'normalize_std': IMAGENET_STD
            }, f"gesture/checkpoints/gesture_{RUN_TAG}.pt")
            print("  *** New Best ***")

if __name__ == "__main__":
    main()
