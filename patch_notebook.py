#!/usr/bin/env python3.14
"""Patch gesture_classifier.ipynb with winning config and add Tasks 12-15 cells."""

import json
import copy

NB_PATH = "HAND_JOB/gesture_classifier.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# Helper to find cell index by partial source match
def find_cell(substr):
    for i, c in enumerate(cells):
        if substr in "".join(c["source"]):
            return i
    return None

# ---- 1. Title cell: replace 64x64 with 96x96 ----
title_idx = find_cell("64x64 hand crops")
assert title_idx is not None, "Title cell not found"
cells[title_idx]["source"] = ["# HaGRID Gesture Classifier\n", "\n",
    "Layer 3b of the vision pipeline. 18 classes, 96x96 hand crops, from-scratch CNN."]
print(f"[1] Patched title cell [{title_idx}]")

# ---- 2. Config cell: IMG_SIZE 64 -> 96 ----
config_idx = find_cell("IMG_SIZE = 64")
assert config_idx is not None, "Config cell not found"
src = "".join(cells[config_idx]["source"])
src = src.replace("IMG_SIZE = 64", "IMG_SIZE = 96")
cells[config_idx]["source"] = src
print(f"[2] Patched config cell [{config_idx}]: IMG_SIZE=96")

# ---- 3. Transforms cell: replace train_tf with light aug ----
tf_idx = find_cell("RandomPerspective")
assert tf_idx is not None, "Transforms cell not found"
old_src = "".join(cells[tf_idx]["source"])
# Find where eval_tf starts to preserve it
eval_tf_start = old_src.find("eval_tf")
eval_tf_part = old_src[eval_tf_start:]
new_train_tf = """train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

"""
cells[tf_idx]["source"] = new_train_tf + eval_tf_part
print(f"[3] Patched transforms cell [{tf_idx}]")

# ---- 4. DataLoaders cell: num_workers=0 -> num_workers=6, persistent_workers=True ----
loaders_idx = find_cell("num_workers=0")
assert loaders_idx is not None, "DataLoaders cell not found"
src = "".join(cells[loaders_idx]["source"])
# Replace all three loaders
src = src.replace(
    "DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, drop_last=True)",
    "DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=6, persistent_workers=True, drop_last=True)"
)
src = src.replace(
    "DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)",
    "DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=6, persistent_workers=True)"
)
src = src.replace(
    "DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)",
    "DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=6, persistent_workers=True)"
)
cells[loaders_idx]["source"] = src
print(f"[4] Patched DataLoaders cell [{loaders_idx}]")

# ---- 5. GestureNet cell: replace with 4-block Wide CNN ----
gesturenet_idx = find_cell("class GestureNet")
assert gesturenet_idx is not None, "GestureNet cell not found"
cells[gesturenet_idx]["source"] = '''class GestureNet(nn.Module):
    """4-block CNN, channels 32->64->128->256, two convs per block. ~1.3M params."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Dropout2d(0.1), nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.features(x))
'''
print(f"[5] Patched GestureNet cell [{gesturenet_idx}]")

# ---- 7. Task 11 Step 1 training config cell ----
train_config_idx = find_cell("EPOCHS = 100")
assert train_config_idx is not None, "Training config cell not found"
src = "".join(cells[train_config_idx]["source"])
src = src.replace("EPOCHS = 100", "EPOCHS = 40")
src = src.replace("EARLY_STOP_PATIENCE = 6", "EARLY_STOP_PATIENCE = 8")
src = src.replace(
    "scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=9, gamma=0.66)",
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)"
)
cells[train_config_idx]["source"] = src
print(f"[7] Patched training config cell [{train_config_idx}]")

# ---- 9. Training loop cell: clear outputs, set execution_count=None ----
training_loop_idx = find_cell("for epoch in range(EPOCHS):")
assert training_loop_idx is not None, "Training loop cell not found"
cells[training_loop_idx]["outputs"] = []
cells[training_loop_idx]["execution_count"] = None
print(f"[9] Cleared training loop cell [{training_loop_idx}] outputs")

# ---- 10. Add Tasks 12-15 cells ----
# Task 12: loss/metric curves from log.json
task12_cell = {
    "cell_type": "code",
    "id": "task12_curves",
    "metadata": {},
    "source": '''## Task 12: Training curves (from log file)
import json as _json
with open(f"{CKPT_DIR}/gesture_v1_wide96.log.json") as f:
    log = _json.load(f)
hist = log["history"]
ep = [h["epoch"] for h in hist]
tr = [h["tr_loss"] for h in hist]
vl = [h["loss"] for h in hist]
va = [h["acc"] for h in hist]
vf = [h["f1"] for h in hist]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep, tr, label="train"); axes[0].plot(ep, vl, label="val")
axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].plot(ep, va, label="val acc"); axes[1].plot(ep, vf, label="val macro-F1")
axes[1].set_title("val metrics"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout(); plt.show()
''',
    "outputs": [],
    "execution_count": None,
}

# Task 13: load best ckpt, val confusion matrix + classification report
task13_cell = {
    "cell_type": "code",
    "id": "task13_val_eval",
    "metadata": {},
    "source": '''## Task 13: Load best checkpoint, val confusion matrix + classification report
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
print(f"loaded checkpoint from epoch {ckpt['epoch']} with val_f1={ckpt['val_macro_f1']:.4f}")

val_eval = evaluate(model, val_loader)
cm = confusion_matrix(val_eval["labels"], val_eval["preds"], labels=list(range(NUM_CLASSES)))
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right")
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("val confusion matrix (row-normalized)")
plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()

print(classification_report(
    val_eval["labels"], val_eval["preds"],
    labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES,
    digits=3, zero_division=0,
))
''',
    "outputs": [],
    "execution_count": None,
}

# Task 14: test set eval + confusion matrix + classification report
task14_cell = {
    "cell_type": "code",
    "id": "task14_test_eval",
    "metadata": {},
    "source": '''## Task 14: Held-out test set evaluation
test_eval = evaluate(model, test_loader)
print(f"TEST | loss {test_eval['loss']:.4f} | acc {test_eval['acc']:.4f} | macro-F1 {test_eval['f1']:.4f}")

cm = confusion_matrix(test_eval["labels"], test_eval["preds"], labels=list(range(NUM_CLASSES)))
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right")
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("TEST confusion matrix (row-normalized)")
plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()

print(classification_report(
    test_eval["labels"], test_eval["preds"],
    labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES,
    digits=3, zero_division=0,
))
''',
    "outputs": [],
    "execution_count": None,
}

# Task 15: predict_crop helper + 3x3 demo grid
task15_cell = {
    "cell_type": "code",
    "id": "task15_predict_crop",
    "metadata": {},
    "source": '''## Task 15: predict_crop helper + inference demo
def predict_crop(np_crop_bgr: np.ndarray) -> tuple[str, float]:
    """Given a BGR hand crop (any size, HxWx3 uint8), return (label, confidence)."""
    assert np_crop_bgr.ndim == 3 and np_crop_bgr.shape[2] == 3
    rgb = np_crop_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb)
    x = eval_tf(pil).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx])

def _row_to_bgr_crop(row):
    img_path, bbox, _, _ = row
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    x, y, w, h = bbox
    px, py, pw, ph = x * W, y * H, w * W, h * H
    side = max(pw, ph); pad = side * 0.15
    x0 = max(0, int(px - pad)); y0 = max(0, int(py - pad))
    x1 = min(W, int(px + pw + pad)); y1 = min(H, int(py + ph + pad))
    crop_rgb = np.array(img.crop((x0, y0, x1, y1)))
    return crop_rgb[:, :, ::-1]  # BGR

rng = random.Random(0)
samples = rng.sample(val_rows, 9)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for ax, row in zip(axes.flat, samples):
    bgr = _row_to_bgr_crop(row)
    label, conf = predict_crop(bgr)
    truth = CLASS_NAMES[row[2]]
    ok = "✓" if label == truth else "✗"
    ax.imshow(bgr[:, :, ::-1]); ax.axis("off")
    ax.set_title(f"{ok} pred: {label} ({conf:.2f})\\ntrue: {truth}", fontsize=9)
plt.tight_layout(); plt.show()
''',
    "outputs": [],
    "execution_count": None,
}

# Append Tasks 12-15 cells
cells.extend([task12_cell, task13_cell, task14_cell, task15_cell])
print(f"[10] Added Tasks 12-15 cells (cells {len(cells)-4} to {len(cells)-1})")

# Save patched notebook
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)
print(f"\nSaved patched notebook to {NB_PATH}")
print(f"Total cells: {len(nb['cells'])}")
